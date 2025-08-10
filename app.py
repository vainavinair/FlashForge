import streamlit as st
from utils.pdf_handler import load_pdf, extract_text_or_ocr
from utils.text_handler import load_txt, load_docx
from utils.text_cleaner import clean_text_for_llm

st.title("Universal File Uploader 📄")

uploaded_file = st.file_uploader(
    "Choose a file (PDF, TXT, DOCX - max 20 pages for PDFs)", 
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        pdf = load_pdf(uploaded_file)

        if pdf:
            num_pages = len(pdf.pages)
            st.info(f"Number of pages: {num_pages}")

            if num_pages > 20:
                st.error("PDF has more than 20 pages. Please upload a shorter PDF.")
            else:
                full_text = extract_text_or_ocr(pdf)
                full_text = clean_text_for_llm(full_text)
                if full_text.strip():
                    st.subheader("Extracted Text:")
                    st.text(full_text)
                else:
                    st.error("No text could be extracted from this PDF, even after OCR.")

    elif filename.endswith(".txt"):
        text = load_txt(uploaded_file)
        text = clean_text_for_llm(text)
        if text:
            st.subheader("Extracted Text:")
            st.text(text)
        else:
            st.error("Failed to read .txt file")

    elif filename.endswith(".docx"):
        text = load_docx(uploaded_file)
        text = clean_text_for_llm(text)
        if text:
            st.subheader("Extracted Text:")
            st.text(text)
        else:
            st.error("Failed to read .docx file")
