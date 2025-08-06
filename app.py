import streamlit as st
import pdfplumber

st.title("PDF Uploader 📄")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    
    with pdfplumber.open(uploaded_file) as pdf:
        num_pages = len(pdf.pages)
        st.info(f"Number of pages: {num_pages}")

        # display text from first page
        first_page_text = pdf.pages[0].extract_text()
        if first_page_text:
            st.subheader("First Page Content:")
            st.text(first_page_text)
        else:
            st.warning("No text found on the first page.")
