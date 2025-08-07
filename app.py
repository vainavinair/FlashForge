import streamlit as st
from utils.pdf_handler import load_pdf, extract_text_all_pages

st.title("PDF Uploader 📄")

uploaded_file = st.file_uploader("Choose a PDF file (max 20 pages)", type="pdf")

if uploaded_file is not None:
    pdf = load_pdf(uploaded_file)

    if pdf:
        num_pages = len(pdf.pages)
        st.info(f"Number of pages: {num_pages}")

        if num_pages > 20:
            st.error("PDF has more than 20 pages. Please upload a shorter PDF.")
        else:
            full_text = extract_text_all_pages(pdf)
            st.subheader("Extracted Text:")
            st.text(full_text)
