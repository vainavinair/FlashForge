# app.py
import streamlit as st
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ---- App modules
from utils.datamodels.db import init_db, get_connection, get_user_stats
from utils.auth.auth import register_user, login_user, logout
from utils.deckcard.deck_handler import create_deck, list_decks, create_card_with_scheduler
from utils.deckcard.get_fc import get_deck_cards
from utils.extraction.pdf_handler import load_pdf, extract_text_or_ocr
from utils.extraction.text_handler import load_txt, load_docx
from utils.extraction.text_cleaner import clean_text_for_llm
from utils.generation.flashcard_gen import generate_flashcards
from utils.reviewer.review_process import get_due_cards, record_review

# Optional ML analytics (safe import)
try:
    from utils.scheduler.hybrid_scheduler import HybridScheduler, SchedulerEvaluator
    from utils.datamodels.db import DB_PATH
    _hybrid = HybridScheduler(DB_PATH)
    _evaluator = SchedulerEvaluator(_hybrid)
except Exception:
    _hybrid = None
    _evaluator = None

# ---- Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

init_db()
st.set_page_config(page_title="FlashForge", page_icon="📚", layout="wide")
st.title("FlashForge: Smart Flashcards with Spaced Repetition 📚")

# =====================================================
# Helpers
# =====================================================

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
            logger.exception("Failed to insert card: %s", e)
    return inserted_ids

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
            , NULL as due_today
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

# =====================================================
# AUTH
# =====================================================
if "user_id" not in st.session_state:
    st.subheader("Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Logged in successfully ✅")
                st.rerun()
            else:
                st.error("Invalid username or password ❌")

    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists ❌")

# =====================================================
# MAIN APP
# =====================================================
else:
    user_id = st.session_state.user_id
    with st.sidebar:
        st.success(f"Logged in as user_id={user_id}")
        if st.button("Logout"):
            logout()
            st.rerun()

        st.markdown("### Quick Actions")
        new_deck_name = st.text_input("New Deck Name", key="sidebar_deck_name")
        if st.button("Create Deck", key="sidebar_create_deck"):
            if new_deck_name.strip():
                create_deck(user_id, new_deck_name.strip())
                st.rerun()

    # Top-level tabs
    tab_dash, tab_decks, tab_upload, tab_review = st.tabs(["📊 Dashboard", "🗂️ Decks", "📄 Upload & Generate", "🧠 Review"])

    # ======================= DASHBOARD =======================
    with tab_dash:
        data = load_dashboard_data(user_id)
        stats = data["stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Decks", stats["total_decks"])
        c2.metric("Total Cards", stats["total_cards"])
        c3.metric("Avg Knowledge", f"{(stats['avg_knowledge'] or 0.0)*100:.1f}%")
        c4.metric("Mastered Cards (≥0.7)", stats["mastered_cards"])

        c5, c6 = st.columns(2)
        c5.metric("Reviews (7d)", stats["reviews_last_7_days"])
        if data["ml_pred_acc"] is not None:
            c6.metric("ML Prediction Calibration (↑ better)", f"{data['ml_pred_acc']*100:.1f}%")
        elif data["ml_analytics"] and "accuracy_rate" in data["ml_analytics"]:
            c6.metric("Accuracy (30d)", f"{data['ml_analytics']['accuracy_rate']*100:.1f}%")
        else:
            c6.metric("Accuracy (30d)", "—")

        st.markdown("---")
        left, right = st.columns([2,1])

        with left:
            st.subheader("Activity — Last 30 Days")
            hist_df = data["hist_df"]
            if len(hist_df):
                hist_df_display = hist_df.copy()
                hist_df_display["day"] = pd.to_datetime(hist_df_display["day"])
                st.line_chart(hist_df_display.set_index("day")[["reviews"]])
                st.line_chart(hist_df_display.set_index("day")[["accuracy"]])
            else:
                st.info("No review activity yet. Start a session to see charts!")

        with right:
            st.subheader("Deck Health")
            deck_df = data["deck_df"]
            if len(deck_df):
                show_df = deck_df.copy()
                show_df["avg_knowledge"] = (show_df["avg_knowledge"].fillna(0) * 100).round(1)
                st.dataframe(
                    show_df.rename(columns={
                        "deck_name":"Deck",
                        "cards":"Cards",
                        "avg_knowledge":"Avg Knowledge (%)",
                        "due_today":"Due Now"
                    })[["Deck","Cards","Avg Knowledge (%)","Due Now"]],
                    use_container_width=True,
                    height=360
                )
            else:
                st.info("No decks yet. Create one from the sidebar.")

        st.markdown("---")
        colL, colR = st.columns([2,1])

        with colL:
            st.subheader("Recent Reviews")
            recent_df = data["recent_df"]
            if len(recent_df):
                nice = recent_df.copy()
                nice["time"] = pd.to_datetime(nice["time"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(nice, use_container_width=True, height=340)
            else:
                st.info("No recent reviews found.")

        with colR:
            st.subheader("ML Insights")
            if data["ml_analytics"]:
                a = data["ml_analytics"]
                st.write(f"• Total reviews (30d): **{a.get('total_reviews', 0)}**")
                st.write(f"• Accuracy: **{(a.get('accuracy_rate',0.0)*100):.1f}%**")
                st.write(f"• Avg response time: **{a.get('avg_response_time',0.0):.1f}s**")
                st.write(f"• Deck avg knowledge: **{a.get('avg_deck_knowledge',0.0):.3f}**")
                st.write(f"• Mastery rate: **{(a.get('mastery_rate',0.0)*100):.1f}%**")
            else:
                st.caption("ML analytics will appear after you have some reviews.")

            if data["ml_suggestions"]:
                st.markdown("**Parameter Suggestions**")
                for k, v in data["ml_suggestions"].items():
                    st.write(f"• {k.replace('_',' ').title()}: **{v}**")
            else:
                st.caption("No parameter suggestions yet.")

    # ======================= DECKS =======================
    with tab_decks:
        st.subheader("Your Decks")
        decks = list_decks(user_id)
        deck_names = {deck_id: name for deck_id, name in decks}

        if decks:
            selected_deck_id = st.selectbox(
                "Select a deck:",
                options=list(deck_names.keys()),
                format_func=lambda x: deck_names[x],
                key="deck_select_view"
            )
            st.write(f"**Deck:** {deck_names[selected_deck_id]}")

            if st.button("View Flashcards in Deck"):
                cards = get_deck_cards(selected_deck_id)
                if cards:
                    for i, (question, answer) in enumerate(cards):
                        with st.expander(f"Q{i+1}: {question}"):
                            st.markdown(f"**Answer:** {answer}")
                else:
                    st.info("No flashcards in this deck yet.")
        else:
            st.info("No decks yet. Create one from the sidebar.")

    # ======================= UPLOAD & GENERATE =======================
    with tab_upload:
        st.subheader("Upload Notes to Generate Flashcards")
        decks_for_save = list_decks(user_id)
        deck_names_save = {deck_id: name for deck_id, name in decks_for_save}
        if decks_for_save:
            selected_deck_id_save = st.selectbox(
                "Choose a deck to save into:",
                options=list(deck_names_save.keys()),
                format_func=lambda x: deck_names_save[x],
                key="deck_select_save"
            )
        else:
            selected_deck_id_save = None
            st.warning("Create a deck first (see sidebar).")

        uploaded_file = st.file_uploader("Choose PDF, TXT, DOCX", type=["pdf", "txt", "docx"])
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            full_text = ""

            if filename.endswith(".pdf"):
                pdf = load_pdf(uploaded_file)
                if pdf:
                    if len(pdf.pages) > 20:
                        st.error("PDF has more than 20 pages ❌")
                    else:
                        with st.spinner("Extracting from PDF (OCR if needed)…"):
                            full_text = extract_text_or_ocr(pdf)
                            full_text = clean_text_for_llm(full_text)

            elif filename.endswith(".txt"):
                with st.spinner("Processing TXT…"):
                    full_text = clean_text_for_llm(load_txt(uploaded_file))

            elif filename.endswith(".docx"):
                with st.spinner("Processing DOCX…"):
                    full_text = clean_text_for_llm(load_docx(uploaded_file))

            if full_text.strip():
                st.subheader("Generated Flashcards")
                flashcards = generate_flashcards(full_text)
                if flashcards:
                    for i, card in enumerate(flashcards):
                        with st.expander(f"Q{i+1}: {card['question']}"):
                            st.markdown(f"**Answer:** {card['answer']}")
                    if selected_deck_id_save:
                        if st.button("Save Flashcards to Deck"):
                            ids = save_flashcards(selected_deck_id_save, flashcards)
                            st.success(f"Saved {len(ids)} flashcards to '{deck_names_save[selected_deck_id_save]}' ✅")
                    else:
                        st.warning("Select a deck above to save.")
                else:
                    st.error("No flashcards generated from the text.")
            else:
                st.error("No extractable text found in the uploaded file.")

    # ======================= REVIEW =======================
    with tab_review:
        st.subheader("Review / Study Mode")
        # Choose deck to review
        decks_rv = list_decks(user_id)
        deck_names_rv = {deck_id: name for deck_id, name in decks_rv}
        if decks_rv:
            selected_deck_id_rv = st.selectbox(
                "Select a deck to review:",
                options=list(deck_names_rv.keys()),
                format_func=lambda x: deck_names_rv[x],
                key="deck_select_review"
            )
        else:
            selected_deck_id_rv = None
            st.info("Create a deck first to start reviewing.")

        session_size = st.number_input("Cards per session", min_value=1, max_value=100, value=10, step=1)
        if st.button("Start Review Session"):
            if selected_deck_id_rv:
                due_card_ids = get_due_cards(user_id, selected_deck_id_rv, limit=session_size)
                if not due_card_ids:
                    st.info("No cards due for review.")
                else:
                    st.session_state.review_cards = due_card_ids
                    st.session_state.review_idx = 0
                    st.rerun()
            else:
                st.warning("Pick a deck to review.")

        # Review flow
        if "review_cards" in st.session_state and st.session_state.get("review_cards"):
            idx = st.session_state.get("review_idx", 0)
            review_ids = st.session_state["review_cards"]
            if idx < len(review_ids):
                card_id = review_ids[idx]
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT question, answer FROM cards WHERE card_id = ?", (card_id,))
                r = cur.fetchone()
                conn.close()

                if r:
                    question, answer = r[0], r[1]
                    st.markdown(f"### Card {idx+1} / {len(review_ids)}")
                    st.markdown(f"**Q:** {question}")
                    show_ans = st.checkbox("Show Answer", key=f"show_ans_{card_id}")
                    if show_ans:
                        st.markdown(f"**A:** {answer}")

                    col1, col2, col3, col4 = st.columns(4)
                    if col1.button("Again"):
                        record_review(card_id, user_id, grade=0, response_time=None)
                        st.session_state.review_idx += 1
                        st.rerun()
                    if col2.button("Hard"):
                        record_review(card_id, user_id, grade=1, response_time=None)
                        st.session_state.review_idx += 1
                        st.rerun()
                    if col3.button("Good"):
                        record_review(card_id, user_id, grade=2, response_time=None)
                        st.session_state.review_idx += 1
                        st.rerun()
                    if col4.button("Easy"):
                        record_review(card_id, user_id, grade=3, response_time=None)
                        st.session_state.review_idx += 1
                        st.rerun()
            else:
                st.success("Session complete ✅")
                st.session_state.pop("review_cards", None)
                st.session_state.pop("review_idx", None)
