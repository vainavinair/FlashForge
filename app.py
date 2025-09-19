from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Your existing utils will be used here
from utils.auth.auth import register_user as auth_register_user, login_user as auth_login_user
from utils.datamodels.db import init_db, get_connection, get_user_stats
from utils.deckcard.deck_handler import create_deck, list_decks, create_card_with_scheduler
from utils.deckcard.get_fc import get_deck_cards
from utils.extraction.pdf_handler import load_pdf, extract_text_or_ocr
from utils.extraction.text_handler import load_txt, load_docx
from utils.extraction.text_cleaner import clean_text_for_llm
from utils.generation.flashcard_gen import generate_flashcards
from utils.reviewer.review_process import get_due_cards, record_review
from werkzeug.utils import secure_filename
import tempfile


# Optional ML analytics (safe import)
try:
    from utils.scheduler.hybrid_scheduler import HybridScheduler, SchedulerEvaluator
    from utils.datamodels.db import DB_PATH
    _hybrid = HybridScheduler(DB_PATH)
    _evaluator = SchedulerEvaluator(_hybrid)
except Exception:
    _hybrid = None
    _evaluator = None

# Initialize the database
init_db()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, username FROM users WHERE user_id = ?", (user_id,))
    user_data = cur.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1])
    return None

# --- Helper Functions ---
def _table_exists(name: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,))
    ok = bool(cur.fetchone())
    conn.close()
    return ok

def load_dashboard_data(user_id: int):
    """
    Collects all data needed for the Dashboard quickly.
    Returns a dict of dataframes/values.
    """
    conn = get_connection()
    cur = conn.cursor()

    # High-level stats (already available)
    stats = get_user_stats(user_id)

    # Reviews last 30 days (counts + accuracy per day)
    cur.execute("""
        SELECT DATE(timestamp) as day,
               COUNT(*) as reviews,
               AVG(CAST(user_response as FLOAT)) as accuracy,
               AVG(response_time) as avg_time
        FROM review_history
        WHERE user_id = ? AND timestamp >= datetime('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
    """, (user_id,))
    rows = cur.fetchall()
    hist_df = pd.DataFrame(rows, columns=["day", "reviews", "accuracy", "avg_time"]) if rows else pd.DataFrame(columns=["day","reviews","accuracy","avg_time"])

    # Deck health: cards per deck, avg knowledge, due today
    due_clause = ""
    if _table_exists("scheduler_state"):
        due_clause = """
            , SUM(CASE WHEN s.next_due_at <= datetime('now', 'localtime') OR s.next_due_at IS NULL THEN 1 ELSE 0 END) as due_today
            FROM decks d
            LEFT JOIN cards c ON c.deck_id = d.deck_id
            LEFT JOIN scheduler_state s ON s.card_id = c.card_id
        """
    else:
        due_clause = """
            , SUM(CASE WHEN c.last_reviewed IS NULL OR c.last_reviewed < datetime('now', '-1 day') THEN 1 ELSE 0 END) as due_today
            FROM decks d
            LEFT JOIN cards c ON c.deck_id = d.deck_id
        """

    cur.execute(f"""
        SELECT d.deck_id, d.name,
               COUNT(c.card_id) as cards,
               AVG(c.knowledge_state) as avg_knowledge
               {due_clause}
        WHERE d.user_id = ?
        GROUP BY d.deck_id, d.name
        ORDER BY d.name
    """, (user_id,))
    deck_rows = cur.fetchall()
    deck_df = pd.DataFrame(deck_rows, columns=["deck_id", "deck_name", "cards", "avg_knowledge", "due_today"]) if deck_rows else pd.DataFrame(columns=["deck_id","deck_name","cards","avg_knowledge","due_today"])

    # Recent reviews (last 15)
    cur.execute("""
        SELECT rh.timestamp, c.question,
               CASE rh.user_response WHEN 1 THEN 'Correct' ELSE 'Incorrect' END as result,
               rh.confidence_level, rh.response_time
        FROM review_history rh
        JOIN cards c ON c.card_id = rh.card_id
        WHERE rh.user_id = ?
        ORDER BY rh.timestamp DESC
        LIMIT 15
    """, (user_id,))
    rr = cur.fetchall()
    recent_df = pd.DataFrame(rr, columns=["time","question","result","confidence","resp_time"]) if rr else pd.DataFrame(columns=["time","question","result","confidence","resp_time"])

    # Convert timestamp strings to datetime objects
    if not recent_df.empty:
        recent_df['time'] = pd.to_datetime(recent_df['time'])

    conn.close()

    # Optional ML analytics
    ml_analytics = None
    ml_suggestions = {}
    ml_pred_acc = None
    if _hybrid is not None:
        try:
            ml_analytics = _hybrid.get_learning_analytics(user_id, days=30)
        except Exception:
            ml_analytics = None
        if _evaluator is not None:
            try:
                ml_suggestions = _evaluator.suggest_parameter_adjustments(user_id)
            except Exception:
                ml_suggestions = {}
            try:
                ml_pred_acc = _evaluator.evaluate_prediction_accuracy(user_id, days=30)
            except Exception:
                ml_pred_acc = None

    return {
        "stats": stats,
        "hist_df": hist_df,
        "deck_df": deck_df,
        "recent_df": recent_df,
        "ml_analytics": ml_analytics,
        "ml_suggestions": ml_suggestions,
        "ml_pred_acc": ml_pred_acc,
    }

def save_flashcards(deck_id, flashcards):
    """
    Save flashcards using create_card_with_scheduler for atomic card+scheduler creation.
    Returns list of card_ids.
    """
    inserted_ids = []
    for card in flashcards:
        q = (card.get("question") or card.get("q") or "").strip()
        a = (card.get("answer") or card.get("a") or "").strip()
        if not q or not a:
            continue
        try:
            cid = create_card_with_scheduler(deck_id, q, a)
            inserted_ids.append(cid)
        except Exception as e:
            # Consider logging this error
            pass
    return inserted_ids

# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if auth_register_user(username, password):
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_id = auth_login_user(username, password)
        if user_id:
            user = load_user(user_id)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    data = load_dashboard_data(current_user.id)
    return render_template('dashboard.html', data=data)

@app.route('/decks', methods=['GET', 'POST'])
@login_required
def decks():
    if request.method == 'POST':
        deck_name = request.form['deck_name']
        if deck_name:
            create_deck(current_user.id, deck_name)
            flash(f"Deck '{deck_name}' created successfully!", 'success')
        return redirect(url_for('decks'))
    
    user_decks = list_decks(current_user.id)
    return render_template('decks.html', decks=user_decks)

@app.route('/deck/<int:deck_id>')
@login_required
def deck_view(deck_id):
    # Ensure the user owns this deck
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM decks WHERE deck_id = ? AND user_id = ?", (deck_id, current_user.id))
    deck = cur.fetchone()
    conn.close()

    if deck:
        deck_name = deck[0]
        cards = get_deck_cards(deck_id)
        return render_template('deck_view.html', cards=cards, deck_name=deck_name)
    else:
        flash("Deck not found or you don't have permission to view it.", 'danger')
        return redirect(url_for('decks'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    user_decks = list_decks(current_user.id)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        deck_id = request.form.get('deck_id')
        if not deck_id:
            flash('No deck selected', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            # Use a temporary file to handle the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            full_text = ""
            try:
                if filename.lower().endswith('.pdf'):
                    pdf = load_pdf(tmp_path)
                    if pdf:
                        full_text = extract_text_or_ocr(pdf)
                        pdf.close()
                elif filename.lower().endswith('.txt'):
                    full_text = load_txt(tmp_path)
                elif filename.lower().endswith('.docx'):
                    full_text = load_docx(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path) # Clean up the temporary file

            if full_text:
                cleaned_text = clean_text_for_llm(full_text)
                flashcards = generate_flashcards(cleaned_text)
                if flashcards:
                    saved_ids = save_flashcards(deck_id, flashcards)
                    flash(f'Successfully generated and saved {len(saved_ids)} flashcards!', 'success')
                    return redirect(url_for('deck_view', deck_id=deck_id))
                else:
                    flash('Could not generate flashcards from the document.', 'warning')
            else:
                flash('Could not extract text from the file.', 'warning')
        
        return redirect(url_for('upload'))

    return render_template('upload.html', decks=user_decks)

@app.route('/review', methods=['GET'])
@login_required
def review():
    # If a review session is in progress, show the current card
    if 'review_cards' in session and session['review_idx'] < len(session['review_cards']):
        card_id = session['review_cards'][session['review_idx']]
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT question, answer FROM cards WHERE card_id = ?", (card_id,))
        card_data = cur.fetchone()
        conn.close()
        
        if card_data:
            card = {'question': card_data[0], 'answer': card_data[1]}
            return render_template('review.html', card=card)
        else:
            # Card not found, advance session
            session['review_idx'] += 1
            return redirect(url_for('review'))

    # If session is over, clear it and show summary
    summary = {}
    if 'review_cards' in session:
        grades = session.get('review_grades', [])
        total_cards = len(grades)
        if total_cards > 0:
            num_correct = sum(1 for g in grades if g > 0)
            accuracy = (num_correct / total_cards) * 100
            summary = {
                'total_cards': total_cards,
                'num_correct': num_correct,
                'accuracy': f"{accuracy:.1f}"
            }

        session.pop('review_cards', None)
        session.pop('review_idx', None)
        session.pop('card_start_time', None)
        session.pop('review_grades', None)

    # Show setup for a new session
    user_decks = list_decks(current_user.id)
    return render_template('review.html', decks=user_decks, summary=summary)

@app.route('/review/start', methods=['POST'])
@login_required
def review_start():
    deck_ids = request.form.getlist('deck_ids')
    limit = int(request.form.get('limit', 10))
    
    if not deck_ids:
        flash('Please select at least one deck.', 'warning')
        return redirect(url_for('review'))

    all_due_cards = []
    for deck_id in deck_ids:
        due_cards = get_due_cards(current_user.id, deck_id, limit=limit)
        all_due_cards.extend(due_cards)
    
    # Simple shuffle and limit
    np.random.shuffle(all_due_cards)
    review_cards = all_due_cards[:limit]
    
    if review_cards:
        session['review_cards'] = review_cards
        session['review_idx'] = 0
        session['card_start_time'] = datetime.utcnow().isoformat()
        session['review_grades'] = []
    else:
        flash('No cards due for review in the selected deck(s).', 'info')

    return redirect(url_for('review'))

@app.route('/review/process', methods=['POST'])
@login_required
def review_process():
    card_id = int(request.form.get('card_id'))
    grade = int(request.form.get('grade'))
    
    start_time_str = session.get('card_start_time')
    response_time = 10.0 # Default
    if start_time_str:
        start_time = datetime.fromisoformat(start_time_str)
        response_time = (datetime.utcnow() - start_time).total_seconds()

    record_review(card_id, current_user.id, grade, response_time)
    
    # Store grade in session for summary
    if 'review_grades' in session:
        session['review_grades'].append(grade)

    session['review_idx'] += 1
    session['card_start_time'] = datetime.utcnow().isoformat()
    
    return redirect(url_for('review'))

# --- API Routes ---
@app.route('/api/chart_data')
@login_required
def chart_data():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DATE(timestamp) as day,
               COUNT(*) as reviews,
               AVG(CAST(user_response as FLOAT)) as accuracy
        FROM review_history
        WHERE user_id = ? AND timestamp >= datetime('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
    """, (current_user.id,))
    rows = cur.fetchall()
    conn.close()
    
    data = {
        "labels": [row[0] for row in rows],
        "reviews": [row[1] for row in rows],
        "accuracy": [row[2] for row in rows]
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
