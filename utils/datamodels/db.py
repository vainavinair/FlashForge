# db.py - Updated schema for Hybrid Scheduler
import sqlite3
import os

# Use environment variable for database path, fallback to local development path
DB_PATH = os.getenv('DATABASE_PATH', os.path.join("database", "flashforge.db"))

def get_connection():
    # Ensure directory exists for database file
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:  # Only create directory if path has a directory component
        os.makedirs(db_dir, exist_ok=True)
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
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        -- New fields for adaptive learning
        adaptation_count INTEGER DEFAULT 0,
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

    # Scheduler state table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scheduler_state (
        state_id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        alpha REAL DEFAULT 1.0,
        beta REAL DEFAULT 1.0,
        interval_days REAL DEFAULT 0.0,
        repetitions INTEGER DEFAULT 0,
        next_due_at DATETIME,
        last_result TEXT,
        last_reviewed_at DATETIME,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    )
    """)

    # Evaluation results table for research metrics
    cur.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_results (
        evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id INTEGER,
        user_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        -- Source text metadata
        source_text_length INTEGER,
        num_flashcards INTEGER,
        
        -- BERTScore metrics
        bert_precision REAL,
        bert_recall REAL,
        bert_f1 REAL,
        question_f1 REAL,
        answer_f1 REAL,
        
        -- Keyword coverage metrics
        keyword_coverage_score REAL,
        total_keywords INTEGER,
        covered_keywords INTEGER,
        
        -- Full evaluation data as JSON
        evaluation_data TEXT,
        
        FOREIGN KEY (deck_id) REFERENCES decks(deck_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    # Source texts table to store original texts for evaluation
    cur.execute("""
    CREATE TABLE IF NOT EXISTS source_texts (
        source_id INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        filename TEXT,
        content TEXT NOT NULL,
        content_hash TEXT,
        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (deck_id) REFERENCES decks(deck_id),
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_deck_id ON evaluation_results(deck_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_user_id ON evaluation_results(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp ON evaluation_results(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source_texts_deck_id ON source_texts(deck_id)")

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
