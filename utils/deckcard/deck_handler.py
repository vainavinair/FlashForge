# utils/deck_handler.py
from typing import List, Tuple
from utils.datamodels.db import get_connection

def create_deck(user_id: int, name: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO decks (user_id, name, created_at) VALUES (?, ?, datetime('now'))", (user_id, name))
    deck_id = cur.lastrowid
    conn.commit()
    conn.close()
    return deck_id

def list_decks(user_id: int) -> List[Tuple[int, str]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT deck_id, name FROM decks WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]

def create_card_with_scheduler(deck_id: int, question: str, answer: str) -> int:
    """
    Insert a card into `cards` and create a default scheduler_state row in one transaction.
    Returns the created card_id.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Insert card
        cur.execute(
            "INSERT INTO cards (deck_id, question, answer, created_at) VALUES (?, ?, ?, datetime('now'))",
            (deck_id, question, answer)
        )
        card_id = cur.lastrowid

        # Insert scheduler_state defaults (if table exists)
        try:
            cur.execute("""
                INSERT OR IGNORE INTO scheduler_state (
                    card_id, alpha, beta, interval_days, repetitions, last_result, last_reviewed_at, next_due_at
                ) VALUES (?, 1.0, 1.0, 0.0, 0, NULL, NULL, datetime('now'))
            """, (card_id,))
        except Exception:
            # If scheduler_state table doesn't exist, ignore silently (MVP may not have it)
            pass

        conn.commit()
        return card_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
