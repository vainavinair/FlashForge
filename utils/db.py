"""
utils/db.py

SQLite connection helper and schema initializer for FlashForge.

- Ensures database lives in ./database/flashforge.db
- Creates minimal tables:
    - users
    - decks
    - cards
    - scheduler_state
    - reviews

Usage:
    from utils.db import init_db, get_connection
    init_db()  # call at app startup to ensure tables exist
    conn = get_connection()
"""

import os
import sqlite3
from datetime import datetime

DB_DIR = "database"
DB_FILENAME = "flashforge.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)

def get_connection():
    """
    Returns a sqlite3.Connection with foreign keys enabled and a row factory.
    Use check_same_thread=False for Streamlit multi-threaded usage.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    # Provide dict-like access to rows
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all required tables and indexes (no-op if they already exist)."""
    conn = get_connection()
    cur = conn.cursor()

    # users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT (datetime('now'))
    );
    """)

    # decks
    cur.execute("""
    CREATE TABLE IF NOT EXISTS decks (
        deck_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # cards
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        card_id INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (deck_id) REFERENCES decks(deck_id) ON DELETE CASCADE
    );
    """)

    # scheduler_state (one row per card)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scheduler_state (
        card_id INTEGER PRIMARY KEY,  -- also FK to cards(card_id)
        alpha REAL NOT NULL DEFAULT 1.0,
        beta REAL NOT NULL DEFAULT 1.0,
        interval_days REAL NOT NULL DEFAULT 0.0,
        repetitions INTEGER NOT NULL DEFAULT 0,
        last_result TEXT,            -- e.g. 'again'|'hard'|'good'|'easy'
        last_reviewed_at DATETIME,
        next_due_at DATETIME,
        FOREIGN KEY (card_id) REFERENCES cards(card_id) ON DELETE CASCADE
    );
    """)

    # reviews (immutable log)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        reviewed_at DATETIME NOT NULL DEFAULT (datetime('now')),
        grade INTEGER NOT NULL,      -- 0=again,1=hard,2=good,3=easy
        recalled INTEGER NOT NULL,   -- 0 or 1
        response_time_ms INTEGER,
        delta_t_seconds INTEGER,     -- time since last review in seconds
        prev_alpha REAL,
        prev_beta REAL,
        prev_interval_days REAL,
        new_alpha REAL,
        new_beta REAL,
        new_interval_days REAL,
        next_due_at DATETIME,
        FOREIGN KEY (card_id) REFERENCES cards(card_id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # Indexes to speed up common queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decks_user ON decks(user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scheduler_next_due ON scheduler_state(next_due_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_card_time ON reviews(card_id, reviewed_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_user_time ON reviews(user_id, reviewed_at);")

    conn.commit()
    conn.close()


# Convenience: initialize on import if DB file doesn't exist.
# This makes it simple for Streamlit apps to call init_db() at startup as well.
if not os.path.exists(DB_PATH):
    init_db()
