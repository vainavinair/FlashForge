import os
import google.generativeai as genai
import nltk
from bert_score import score as bert_scorer
import json
import re
import numpy as np
from dotenv import load_dotenv
import sys
import subprocess

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

load_dotenv()

# --- Configuration ---

# 1. Set up your Gemini API Key
# Note: It's recommended to use environment variables for security.
# Example: os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
try:
    # Attempt to get the key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise KeyError
except KeyError:
    print("API Key not found in environment variables.")
    # Fallback for demonstration purposes. Replace with your actual key if not using env vars.
    api_key = "YOUR_GEMINI_API_KEY_HERE" 
    if api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("ERROR: Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key.")
        exit()

genai.configure(api_key=api_key)

# --- 1. Flashcard Generation ---

def generate_flashcards_from_text(text: str):
    """Generates flashcards from text using the Gemini API."""
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Based on the text below, generate a list of 5-10 flashcards.
    Each flashcard should be a JSON object with a "question" and an "answer" key.
    The question should be a clear, concise question testing a key concept from the text.
    The answer should be a direct and accurate answer to that question.
    Return ONLY a valid JSON array of these objects, without any markdown formatting or surrounding text.

    TEXT:
    ---
    {text}
    ---
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = re.search(r'\[.*\]', response.text, re.DOTALL)
        if cleaned_response:
            return json.loads(cleaned_response.group(0))
        else:
            print("Warning: Could not parse flashcards from LLM response.")
            return []
    except Exception as e:
        print(f"An error occurred during flashcard generation: {e}")
        return []

# --- 2. Evaluation Metrics ---

def evaluate_bert_score(source_text: str, generated_flashcards: list):
    """
    Calculates BERTScore for generated flashcards against the source text.
    
    Compares each question and answer to the full source text.
    """
    if not generated_flashcards:
        return None, None, None

    questions = [card['question'] for card in generated_flashcards]
    answers = [card['answer'] for card in generated_flashcards]
    
    # We need to provide a list of source texts for each candidate
    source_texts = [source_text] * len(questions)
    
    # Score questions
    P_q, R_q, F1_q = bert_scorer(questions, source_texts, lang='en', verbose=False)
    
    # Score answers
    P_a, R_a, F1_a = bert_scorer(answers, source_texts, lang='en', verbose=False)
    
    # Average the scores for a single, overall metric
    avg_precision = (P_q.mean() + P_a.mean()) / 2
    avg_recall = (R_q.mean() + R_a.mean()) / 2
    avg_f1 = (F1_q.mean() + F1_a.mean()) / 2
    
    return avg_precision.item(), avg_recall.item(), avg_f1.item()


def evaluate_keyword_coverage(source_text: str, generated_flashcards: list):
    """
    Calculates the percentage of key words from the source text
    that are covered by the generated flashcards using simple regex extraction.
    """
    if not generated_flashcards:
        return 0.0, []

    # Simple regex-based keyword extraction (capitalized words and longer words)
    # This avoids the NLTK POS tagging issue entirely
    import re
    
    # Extract words that are either:
    # 1. Capitalized (likely proper nouns or important terms)
    # 2. Longer than 4 characters (likely content words)
    # 3. Common scientific/academic terms
    words = re.findall(r'\b[A-Za-z]+\b', source_text)
    
    source_keywords = set()
    for word in words:
        word_lower = word.lower()
        # Include capitalized words, longer words, and filter out common stop words
        if (word[0].isupper() or len(word) > 4) and len(word) > 2:
            if word_lower not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'then', 'them', 'these', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'over', 'such', 'take', 'than', 'they', 'well', 'were'}:
                source_keywords.add(word_lower)
    
    if not source_keywords:
        print("Warning: No keywords found in source text for coverage analysis.")
        return 0.0, []

    # Combine all text from the flashcards
    flashcards_text = " ".join([f"{card['question']} {card['answer']}" for card in generated_flashcards]).lower()
    
    # Count how many source keywords are present in the flashcards
    covered_keywords = 0
    for keyword in source_keywords:
        # Simple substring check
        if keyword in flashcards_text:
            covered_keywords += 1
            
    coverage_score = covered_keywords / len(source_keywords)
    return coverage_score, sorted(list(source_keywords))


# --- Main Execution ---

if __name__ == "__main__":
    # Sample text for demonstration. You can replace this with any text.
    sample_source_text = """
    The powerhouse of the cell, the mitochondrion, is responsible for generating most of the cell's supply of adenosine triphosphate (ATP), 
    used as a source of chemical energy. This process is called cellular respiration. Mitochondria are composed of two membranes: an outer membrane 
    and an inner membrane, which is folded into structures called cristae. These cristae increase the surface area for ATP production. 
    The citric acid cycle, also known as the Krebs cycle, occurs in the mitochondrial matrix, while the electron transport chain is located on the inner membrane. 
    Mitochondria are unique in that they have their own small circular DNA, known as mitochondrial DNA or mtDNA.
    """
    
    print("--- 1. Generating Flashcards ---")
    flashcards = generate_flashcards_from_text(sample_source_text)
    
    if not flashcards:
        print("Flashcard generation failed. Exiting.")
    else:
        print(f"Successfully generated {len(flashcards)} flashcards.")
        for i, card in enumerate(flashcards, 1):
            print(f"  {i}. Q: {card['question']}")
            print(f"     A: {card['answer']}")
        print("-" * 30)

        print("\n--- 2. Calculating Evaluation Metrics ---")
        
        # BERTScore Evaluation
        print("\nCalculating BERTScore (this may take a moment)...")
        precision, recall, f1 = evaluate_bert_score(sample_source_text, flashcards)
        if f1 is not None:
            print(f"  - Average BERT F1-Score: {f1:.4f}")
            print(f"  - Average BERT Precision: {precision:.4f}")
            print(f"  - Average BERT Recall: {recall:.4f}")
        
        # Keyword/Entity Coverage Evaluation
        print("\nCalculating Keyword/Entity Coverage...")
        coverage, keywords = evaluate_keyword_coverage(sample_source_text, flashcards)
        if keywords:
            print(f"\n  - Identified {len(keywords)} key keywords (nouns) in source text:")
            print(f"    {keywords}")
            print(f"\n  - Keyword Coverage Score: {coverage:.2%}")
            
        print("\n--- Evaluation Complete ---")
