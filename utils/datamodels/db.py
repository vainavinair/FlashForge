# db.py - Updated schema for Hybrid Scheduler
import sqlite3
import os

DB_PATH = os.path.join("database", "flashforge.db")

def get_connection():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Decks table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS decks (
        deck_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    # Enhanced Cards table with ML parameters
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        card_id INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        -- Thompson Sampling parameters
        alpha_param REAL DEFAULT 1.0,
        beta_param REAL DEFAULT 1.0,
        
        -- Knowledge Tracing parameters
        knowledge_state REAL DEFAULT 0.1,
        learning_rate REAL DEFAULT 0.3,
        slip_probability REAL DEFAULT 0.1,
        guess_probability REAL DEFAULT 0.2,
        forgetting_rate REAL DEFAULT 0.1,
        
        -- Review tracking
        last_reviewed DATETIME,
        last_knowledge_update DATETIME DEFAULT CURRENT_TIMESTAMP,
        review_count INTEGER DEFAULT 0,
        
        FOREIGN KEY (deck_id) REFERENCES decks(deck_id)
    )
    """)

    # User learning parameters
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_learning_params (
        user_id INTEGER PRIMARY KEY,
        base_learning_rate REAL DEFAULT 0.3,
        base_forgetting_rate REAL DEFAULT 0.1,
        exploration_weight REAL DEFAULT 0.6,
        knowledge_weight REAL DEFAULT 0.4,
        optimal_session_length INTEGER DEFAULT 20,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    # Review history for analytics and learning
    cur.execute("""
    CREATE TABLE IF NOT EXISTS review_history (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        -- User response data
        user_response INTEGER NOT NULL,  -- 1=correct, 0=incorrect
        response_time REAL,              -- seconds
        confidence_level INTEGER,        -- 1-5 scale (1=hard, 5=easy)
        
        -- ML state tracking
        pre_review_knowledge_state REAL,
        post_review_knowledge_state REAL,
        sampled_theta REAL,  -- Thompson sampling probability
        
        FOREIGN KEY (card_id) REFERENCES cards(card_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    # Create indexes for better performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_deck_id ON cards(deck_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_last_reviewed ON cards(last_reviewed)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_knowledge_state ON cards(knowledge_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_review_history_card_id ON review_history(card_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_review_history_user_id ON review_history(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_review_history_timestamp ON review_history(timestamp)")

    conn.commit()
    conn.close()

def get_user_stats(user_id):
    """Get comprehensive user statistics"""
    conn = get_connection()
    cur = conn.cursor()
    
    # Basic stats
    cur.execute("""
    SELECT 
        COUNT(DISTINCT d.deck_id) as total_decks,
        COUNT(c.card_id) as total_cards,
        AVG(c.knowledge_state) as avg_knowledge,
        COUNT(CASE WHEN c.knowledge_state > 0.7 THEN 1 END) as mastered_cards
    FROM decks d
    LEFT JOIN cards c ON d.deck_id = c.deck_id
    WHERE d.user_id = ?
    """, (user_id,))
    
    stats = cur.fetchone()
    
    # Recent activity
    cur.execute("""
    SELECT COUNT(*) as reviews_last_7_days
    FROM review_history
    WHERE user_id = ? 
    AND timestamp > datetime('now', '-7 days')
    """, (user_id,))
    
    recent_activity = cur.fetchone()[0]
    
    conn.close()
    
    return {
        'total_decks': stats[0] or 0,
        'total_cards': stats[1] or 0,
        'avg_knowledge': stats[2] or 0.0,
        'mastered_cards': stats[3] or 0,
        'reviews_last_7_days': recent_activity
    }
