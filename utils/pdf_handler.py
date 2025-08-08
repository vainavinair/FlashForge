import pdfplumber
from utils.ocr_handler import ocr_pdf_pages

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

def extract_text_or_ocr(pdf, threshold=50):
    """
    Extract text from PDF pages.
    If extracted text length < threshold, use OCR on PDF pages.
    """
    extracted_text = extract_text_all_pages(pdf)
    if len(extracted_text) < threshold:
        # Likely scanned PDF with no selectable text, fallback to OCR
        extracted_text = ocr_pdf_pages(pdf)
    return extracted_text
