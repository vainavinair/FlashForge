import pdfplumber

def load_pdf(uploaded_file):
    try:
        return pdfplumber.open(uploaded_file)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def extract_text_all_pages(pdf):
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    return all_text.strip()