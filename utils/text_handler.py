# text_handler.py

import docx

def load_txt(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        print(f"Error loading TXT: {e}")
        return None

def load_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error loading DOCX: {e}")
        return None
    