import os
from dotenv import load_dotenv

# Lazy loading - only import when actually generating flashcards
_model = None

def _get_model():
    """Lazy load the Gemini model to avoid slow imports on app startup."""
    global _model
    if _model is None:
        import google.generativeai as genai
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel('gemini-2.0-flash')
    return _model

def generate_flashcards(text: str) -> list[dict]:
    """
    Generate flashcards from input text using Gemini LLM.

    Returns:
        List of dicts: [{"question": str, "answer": str}, ...]
    """
    prompt = f"""
You are an expert at creating high‑quality flashcards based on cognitive
science and active‑recall learning principles.

I will give you text from a textbook or study notes.
Follow this process:

1. Break the text into clear meaning units without losing essential ideas,
   abbreviations, or acronyms.
2. From these meaning units, generate flashcards that each test one atomic
   concept.
3. Write questions that promote active recall, including a mix of factual
   and conceptual questions. Do not generate recognition‑style questions
   (no \"Which of the following\" or simple true/false).

Produce flashcards in the **exact** format (no bullets, numbering, or extra text):

Q: [concise, unambiguous question]
A: [short answer, ideally 5‑12 words]

Important content rules (optimize for semantic coverage of the source):
- Reuse key technical terms and important phrases from the original text
  whenever they are clear. Prefer preserving terminology rather than
  paraphrasing it away.
- Each flashcard must cover exactly one important fact, relationship,
  definition, formula, or acronym expansion.
- Answers should usually contain the main term or phrase that appears in
  the source text, so that the meaning is very close to the original.
- Let the number of flashcards be appropriate to the content; for a
  typical 2–5 page document, generate **12–25** high‑quality cards.
- Prefer high‑quality, concept‑dense flashcards over quantity.
- Answers must still be short enough to fit on one side of a flashcard.
- Avoid long explanations, examples, or commentary in either question or answer.
- Skip irrelevant or unclear parts of the text.
- Output only flashcards, one Q/A pair after another, one card per line or
  per two lines, with no extra prose before or after.

Here is the text:

{text}
"""

    

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_flashcards = response.text.strip().split("\n")
        
        flashcards = []
        for line in raw_flashcards:
            if line.startswith("Q: ") and "A: " in line:
                # Sometimes Q and A might be on same line, rare but handle
                qa = line.split("A:")
                question = qa[0][3:].strip()
                answer = qa[1].strip()
                flashcards.append({"question": question, "answer": answer})
            elif line.startswith("Q: "):
                # Expect next line to be answer - handle if output differs
                question = line[3:].strip()
                # Find next line index safely
                idx = raw_flashcards.index(line)
                if idx + 1 < len(raw_flashcards) and raw_flashcards[idx + 1].startswith("A: "):
                    answer = raw_flashcards[idx + 1][3:].strip()
                    flashcards.append({"question": question, "answer": answer})
        return flashcards

    except Exception as e:
        print(f"Error generating flashcards: {e}")
        return []
