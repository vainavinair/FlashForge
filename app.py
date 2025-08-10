import streamlit as st
from utils.pdf_handler import load_pdf, extract_text_or_ocr
from utils.text_handler import load_txt, load_docx
from utils. text_cleaner import clean_text_for_llm
from utils.flashcard_gen import generate_flashcards

st.title("Universal File Uploader with Flashcards 📄✨")

uploaded_file = st.file_uploader(
    "Choose a file (PDF, TXT, DOCX - max 20 pages for PDFs)", 
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    full_text = ""

    if filename.endswith(".pdf"):
        pdf = load_pdf(uploaded_file)
        if pdf:
            num_pages = len(pdf.pages)
            st.info(f"Number of pages: {num_pages}")

            if num_pages > 20:
                st.error("PDF has more than 20 pages. Please upload a shorter PDF.")
            else:
                with st.spinner("Extracting and processing text..."):
                    full_text = extract_text_or_ocr(pdf)
                    full_text = clean_text_for_llm(full_text)

    elif filename.endswith(".txt"):
        with st.spinner("Loading and processing TXT..."):
            full_text = load_txt(uploaded_file)
            full_text = clean_text_for_llm(full_text)

    elif filename.endswith(".docx"):
        with st.spinner("Loading and processing DOCX..."):
            full_text = load_docx(uploaded_file)
            full_text = clean_text_for_llm(full_text)

    if full_text.strip():
        st.subheader("Generated Flashcards:")
        flashcards = generate_flashcards(full_text)

        if flashcards:
            for i, card in enumerate(flashcards):
                with st.expander(f"Q{i+1}: {card['question']}"):
                    st.markdown(f"**Answer:** {card['answer']}")
        else:
            st.error("No flashcards generated from the text.")
    else:
        st.error("No extractable text found in the uploaded file.")
