import streamlit as st
from utils.db import init_db, get_connection
from utils.auth import register_user, login_user, logout
from utils.deck_handler import create_deck, list_decks
from utils.pdf_handler import load_pdf, extract_text_or_ocr
from utils.scheduler import ensure_scheduler_row
from utils.text_handler import load_txt, load_docx
from utils.text_cleaner import clean_text_for_llm
from utils.flashcard_gen import generate_flashcards

# Initialize DB
init_db()

st.set_page_config(page_title="FlashForge", page_icon="📚", layout="wide")
st.title("FlashForge: Smart Flashcards with Spaced Repetition 📚")

# --- Helper: Save flashcards into DB ---
def save_flashcards(deck_id, flashcards):
    conn = get_connection()
    cur = conn.cursor()
    for card in flashcards:
        cur.execute(
            "INSERT INTO cards (deck_id, question, answer) VALUES (?, ?, ?)",
            (deck_id, card["question"], card["answer"])
        )
        ensure_scheduler_row(card["card_id"])
    conn.commit()
    conn.close()

# ------------------ AUTHENTICATION ------------------
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

# ------------------ MAIN APP ------------------
else:
    st.success(f"Logged in as user_id={st.session_state.user_id}")
    if st.button("Logout"):
        logout()
        st.rerun()

    # --- Deck Management ---
    st.subheader("Your Decks")
    decks = list_decks(st.session_state.user_id)

    deck_names = {deck_id: name for deck_id, name in decks}
    if decks:
        selected_deck_id = st.selectbox(
            "Select a deck to work with:",
            options=list(deck_names.keys()),
            format_func=lambda x: deck_names[x]
        )
    else:
        st.info("No decks yet. Create one below.")
        selected_deck_id = None

    new_deck_name = st.text_input("New Deck Name")
    if st.button("Create Deck"):
        if new_deck_name.strip():
            create_deck(st.session_state.user_id, new_deck_name.strip())
            st.rerun()

    # --- File Upload & Flashcard Generation ---
    st.subheader("Upload Notes to Generate Flashcards")
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
                    full_text = extract_text_or_ocr(pdf)
                    full_text = clean_text_for_llm(full_text)

        elif filename.endswith(".txt"):
            full_text = clean_text_for_llm(load_txt(uploaded_file))

        elif filename.endswith(".docx"):
            full_text = clean_text_for_llm(load_docx(uploaded_file))

        if full_text.strip():
            st.subheader("Generated Flashcards:")
            flashcards = generate_flashcards(full_text)

            for i, card in enumerate(flashcards):
                with st.expander(f"Q{i+1}: {card['question']}"):
                    st.markdown(f"**Answer:** {card['answer']}")

            # Save flashcards into deck
            if selected_deck_id:
                if st.button("Save Flashcards to Deck"):
                    save_flashcards(selected_deck_id, flashcards)
                    st.success(f"{len(flashcards)} flashcards saved to deck '{deck_names[selected_deck_id]}' ✅")
            else:
                st.warning("Select a deck to save flashcards.")
        else:
            st.error("No text extracted ❌")
