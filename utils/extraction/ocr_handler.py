# ocr_handler.py

import pytesseract
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
TESS_PATH = os.getenv("TESS_PATH")
# Only set tesseract_cmd if TESS_PATH is provided, otherwise use system default
if TESS_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

def ocr_image(image: Image.Image) -> str:
    """
    Perform OCR on a PIL Image and return extracted text.
    """
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def ocr_pdf_pages(pdf):
    """
    Perform OCR on each page image of a pdfplumber PDF object.
    Returns concatenated text from all pages.
    """
    full_text = ""
    for page in pdf.pages:
        pil_image = page.to_image(resolution=300).original  # pdfplumber to PIL image
        page_text = ocr_image(pil_image)
        full_text += page_text + "\n"
    return full_text.strip()
