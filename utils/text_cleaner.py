import re
import unicodedata

def clean_text_for_llm(text: str) -> str:
    if not text:
        return ""

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Fix common OCR ligatures
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
        "ﬃ": "ffi", "ﬄ": "ffl",
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)

    # Remove OCR error lines
    text = re.sub(r"OCR error:.*", "", text)

    # Remove stray symbols
    text = re.sub(r"[~_=—–•<>]+", " ", text)

    # Normalize MCQ option markers
    text = re.sub(r"\b([a-dA-D])[\s\.\-–~\"'“”]+", r"\1. ", text)
    text = re.sub(r"\b[Cc][cC]?\.\s*", "", text)

    # Merge lines into paragraphs — keep breaks before numbered questions
    lines = text.splitlines()
    merged_lines = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # If line starts with question number (e.g., "1. ", "15.")
        if re.match(r"^\d+\.", stripped):
            if buffer:
                merged_lines.append(buffer.strip())
                buffer = ""
            merged_lines.append(stripped)
        else:
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped

    if buffer:
        merged_lines.append(buffer.strip())

    # Join all lines with a single newline
    text = "\n".join(merged_lines)

    # Remove multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
