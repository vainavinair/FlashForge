
import sqlite3
from typing import List, Tuple, Optional, Any
from utils.datamodels.db import get_connection

# Small helpers to centralize DB interactions used by scheduler/evaluator

def fetch_deck_cards_basic(deck_id: int) -> List[Tuple[Any, ...]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT card_id, question, answer, alpha_param, beta_param, knowledge_state,
               learning_rate, slip_probability, guess_probability, forgetting_rate,
               last_reviewed, last_knowledge_update, review_count
        FROM cards WHERE deck_id = ?
    """, (deck_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_user_params_row(user_id: int) -> Optional[Tuple[Any, ...]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_learning_params WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def insert_user_params_row(user_id: int, params) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_learning_params
        (user_id, base_learning_rate, base_forgetting_rate, exploration_weight, knowledge_weight)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, params.base_learning_rate, params.base_forgetting_rate, params.exploration_weight, params.knowledge_weight))
    conn.commit()
    conn.close()


def update_card_row(card_id: int, updates: dict) -> None:
    conn = get_connection()
    cur = conn.cursor()
    sets = ", ".join([f"{k} = ?" for k in updates.keys()])
    params = list(updates.values()) + [card_id]
    cur.execute(f"UPDATE cards SET {sets} WHERE card_id = ?", params)
    conn.commit()
    conn.close()


def upsert_scheduler_state(card_id: int, alpha: float, beta: float, interval_days: float, repetitions: int, last_result: Optional[str], last_reviewed_at: Optional[str], next_due_at: Optional[str]) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO scheduler_state
        (card_id, alpha, beta, interval_days, repetitions, last_result, last_reviewed_at, next_due_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (card_id, alpha, beta, interval_days, repetitions, last_result, last_reviewed_at, next_due_at))
    conn.commit()
    conn.close()


def insert_review_history(card_id: int, user_id: int, user_response: int, response_time: Optional[float], confidence_level: Optional[int], pre_k: float, post_k: float, sampled_theta: Optional[float]) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO review_history (card_id, user_id, user_response, response_time, confidence_level,
                                    pre_review_knowledge_state, post_review_knowledge_state, sampled_theta, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (card_id, user_id, user_response, response_time, confidence_level, pre_k, post_k, sampled_theta))
    conn.commit()
    conn.close()


def fetch_recent_reviews(user_id: int, days: int = 7):
    conn = get_connection()
    cur = conn.cursor()
    start_date = ("%s" % ("now",))
    cur.execute("""
        SELECT COUNT(*) as total_reviews,
               AVG(CAST(user_response as FLOAT)) as accuracy_rate,
               AVG(response_time) as avg_response_time,
               AVG(post_review_knowledge_state) as avg_knowledge
        FROM review_history
        WHERE user_id = ? AND timestamp > datetime('now', '-? days')
    """, (user_id, days))
    # Note: this helper is intentionally minimal — callers may run more tailored queries.
    rows = cur.fetchone()
    conn.close()
    return rows