import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')

def generate_flashcards(text: str) -> list[dict]:
    """
    Generate flashcards from input text using Gemini LLM.

    Returns:
        List of dicts: [{"question": str, "answer": str}, ...]
    """
    prompt = f"""
You are an expert at creating flashcards for active recall learning.  
I will give you text from a textbook or study notes.  
Create flashcards in the exact format:

Q: [clear, concise question]  
A: [short answer, max 8 words]  

Rules:
- Answers must be short enough to fit on one side of a flashcard.
- Avoid full sentences unless necessary — prefer key terms or short phrases.
- Each card should test only one fact or concept.
- Skip unclear or irrelevant parts of the text.
- Do not include explanations, examples, or commentary.
- Output only the flashcards, one per line, no extra formatting.

Here is the text:

{text}

Generate only the flashcards now.
"""

    try:
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
