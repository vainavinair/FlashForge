# utils/scheduler.py

from utils.db import get_connection

def ensure_scheduler_row(card_id):
    """
    Ensure there's a scheduler_state row for the given card_id.
    - If a row already exists, does nothing and returns False.
    - If missing, inserts a default row and returns True.

    Default values: alpha=1.0, beta=1.0, interval_days=0.0, repetitions=0, next_due_at = now
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # fast existence check
        cur.execute("SELECT 1 FROM scheduler_state WHERE card_id = ?", (card_id,))
        if cur.fetchone():
            return False

        # insert default scheduler row
        cur.execute("""
            INSERT INTO scheduler_state (
                card_id,
                alpha,
                beta,
                interval_days,
                repetitions,
                last_result,
                last_reviewed_at,
                next_due_at
            ) VALUES (?, 1.0, 1.0, 0.0, 0, NULL, NULL, datetime('now'))
        """, (card_id,))
        conn.commit()
        return True
    finally:
        conn.close()
