# utils/review_process.py
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from utils.datamodels.db import get_connection, DB_PATH
import logging

logger = logging.getLogger("review_process")

# Try to import HybridScheduler if present
try:
    from utils.scheduler.hybrid_scheduler import HybridScheduler
    hybrid = HybridScheduler(DB_PATH)
except Exception:
    hybrid = None

# Helper: parse datetime strings from SQLite (handles NULL)
def _to_dt(val):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        # fallback for "YYYY-MM-DD HH:MM:SS" format
        try:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def get_due_cards(user_id: int, deck_id: Optional[int] = None, limit: int = 10) -> List[int]:
    """Get cards due for review, with fallback to simple query"""
    # Try hybrid scheduler first
    try:
        if hybrid:
            if deck_id:
                cards = hybrid.select_cards_for_review(deck_id, user_id, limit)
                return [card.card_id for card in cards]
            else:
                # For now, get cards from all user's decks
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT deck_id FROM decks WHERE user_id = ?", (user_id,))
                deck_ids = [row[0] for row in cur.fetchall()]
                conn.close()
                
                all_due_cards = []
                for d_id in deck_ids:
                    cards = hybrid.select_cards_for_review(d_id, user_id, limit // len(deck_ids) + 1)
                    all_due_cards.extend([card.card_id for card in cards])
                return all_due_cards[:limit]
    except Exception as e:
        logger.exception("Hybrid scheduler failed, falling back to simple due query: %s", e)

    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    
    # Check if scheduler_state table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scheduler_state'")
    table_exists = cur.fetchone() is not None
    
    if not table_exists:
        # If table doesn't exist, return cards that haven't been reviewed recently
        if deck_id:
            cur.execute("""
                SELECT c.card_id
                FROM cards c
                WHERE c.deck_id = ? AND (c.last_reviewed IS NULL OR c.last_reviewed < datetime('now', '-1 day'))
                ORDER BY c.last_reviewed ASC NULLS FIRST
                LIMIT ?
            """, (deck_id, limit))
        else:
            cur.execute("""
                SELECT c.card_id
                FROM cards c
                JOIN decks d ON c.deck_id = d.deck_id
                WHERE d.user_id = ? AND (c.last_reviewed IS NULL OR c.last_reviewed < datetime('now', '-1 day'))
                ORDER BY c.last_reviewed ASC NULLS FIRST
                LIMIT ?
            """, (user_id, limit))
    else:
        # Use scheduler_state table
        if deck_id:
            cur.execute("""
                SELECT s.card_id
                FROM scheduler_state s
                JOIN cards c ON c.card_id = s.card_id
                WHERE c.deck_id = ? AND (s.next_due_at IS NULL OR s.next_due_at <= ?)
                ORDER BY s.next_due_at ASC
                LIMIT ?
            """, (deck_id, now, limit))
        else:
            cur.execute("""
                SELECT s.card_id
                FROM scheduler_state s
                JOIN cards c ON c.card_id = s.card_id
                JOIN decks d ON c.deck_id = d.deck_id
                WHERE d.user_id = ? AND (s.next_due_at IS NULL OR s.next_due_at <= ?)
                ORDER BY s.next_due_at ASC
                LIMIT ?
            """, (user_id, now, limit))

    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows] if rows else []

def _fallback_scheduler_update(card_id: int, success: bool) -> Tuple[float, datetime]:
    """
    Minimal fallback update rule:
      - maintain alpha/beta in scheduler_state (Beta posterior)
      - simple interval update:
          success -> interval *= 1.5 (min 1 day)
          failure -> interval = 1 day
      - compute next_due_at = now + interval_days
    Returns (new_interval_days, next_due_at)
    """
    conn = get_connection()
    cur = conn.cursor()
    # Read current
    cur.execute("""
        SELECT alpha, beta, interval_days, repetitions
        FROM scheduler_state WHERE card_id = ?
    """, (card_id,))
    row = cur.fetchone()
    if not row:
        # create a default scheduler row if missing
        try:
            cur.execute("""
                INSERT OR IGNORE INTO scheduler_state (card_id, alpha, beta, interval_days, repetitions, next_due_at)
                VALUES (?, 1.0, 1.0, 0.0, 0, datetime('now'))
            """, (card_id,))
            conn.commit()
            cur.execute("SELECT alpha, beta, interval_days, repetitions FROM scheduler_state WHERE card_id = ?", (card_id,))
            row = cur.fetchone()
        except Exception:
            conn.close()
            raise

    alpha, beta, interval_days, repetitions = row[0] or 1.0, row[1] or 1.0, row[2] or 0.0, row[3] or 0
    # Update posterior
    if success:
        alpha += 1.0
    else:
        beta += 1.0

    # Interval logic
    if success:
        new_interval = max(1.0, interval_days * 1.5)  # simple growth
        if repetitions == 0:
            new_interval = 1.0
    else:
        new_interval = 1.0

    next_due = datetime.now() + timedelta(days=new_interval)

    # Update DB
    cur.execute("""
        UPDATE scheduler_state
        SET alpha = ?, beta = ?, interval_days = ?, repetitions = ?, last_result = ?, last_reviewed_at = ?, next_due_at = ?
        WHERE card_id = ?
    """, (alpha, beta, new_interval, repetitions + 1, ('good' if success else 'again'), datetime.now().isoformat(), next_due.isoformat(), card_id))

    conn.commit()
    conn.close()
    return new_interval, next_due

def record_review(card_id: int, user_id: int, grade: int, response_time: Optional[float] = None):
    """
    Record review, update scheduler_state either via HybridScheduler or fallback,
    and insert immutable review_history row.
    grade: 0=again,1=hard,2=good,3=easy
    """
    success = True if grade >= 2 else False

    # If hybrid available, delegate update
    if hybrid:
        try:
            # hybrid.update_after_review handles DB updates and logging (as in hybrid_scheduler)
            hybrid.update_after_review(card_id, user_id, success, response_time, confidence=grade)
            
            # Update adaptive learning parameters based on recent performance
            try:
                # Get recent performance data
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("""
                    SELECT AVG(CAST(user_response as FLOAT)) as accuracy
                    FROM review_history 
                    WHERE user_id = ? AND timestamp > datetime('now', '-7 days')
                """, (user_id,))
                result = cur.fetchone()
                conn.close()
                
                if result and result[0] is not None:
                    performance_data = {'accuracy': result[0]}
                    hybrid.adapt_learning_parameters(user_id, performance_data)
            except Exception as e:
                logger.exception("Failed to update adaptive learning parameters: %s", e)
            
            return
        except Exception as e:
            logger.exception("Hybrid update failed, falling back: %s", e)

    # Read pre-review scheduler snapshot
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT alpha, beta, interval_days FROM scheduler_state WHERE card_id = ?", (card_id,))
    row = cur.fetchone()
    prev_alpha = prev_beta = prev_interval = None
    if row:
        prev_alpha, prev_beta, prev_interval = row[0], row[1], row[2]
    conn.close()

    # Fallback update
    new_interval, next_due = _fallback_scheduler_update(card_id, success)

    # Insert immutable review_history row
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO review_history (
            card_id, user_id, user_response, response_time,
            confidence_level, pre_review_knowledge_state, post_review_knowledge_state, sampled_theta, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (card_id, user_id, int(success), response_time, grade, prev_interval or 0.0, new_interval, None))
    conn.commit()
    conn.close()

    logger.info(f"Recorded review for card {card_id}: success={success}, next_due={next_due.isoformat()}")

