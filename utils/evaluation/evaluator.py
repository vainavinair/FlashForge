"""
Comprehensive Evaluation System for FlashForge

This module integrates the BERTScore and Keyword Coverage metrics from the evaluation script
into the main application for systematic evaluation of flashcard generation quality.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
import warnings

# Suppress expected warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

# Core dependencies
from bert_score import score as bert_scorer

# Application imports
from utils.datamodels.db import get_connection

logger = logging.getLogger(__name__)


class FlashcardEvaluator:
    """
    Comprehensive evaluator for flashcard generation quality using:
    1. BERTScore: Semantic similarity between generated flashcards and source text
    2. Keyword Coverage: Percentage of key terms from source text covered in flashcards
    """
    
    def __init__(self):
        self.bert_model = 'bert-base-uncased'  # Default BERT model for scoring
        
    def evaluate_bert_score(self, source_text: str, generated_flashcards: List[Dict[str, str]]) -> Optional[Dict[str, float]]:
        """
        Calculate BERTScore for generated flashcards against the source text.
        
        Args:
            source_text: Original text used for flashcard generation
            generated_flashcards: List of dicts with 'question' and 'answer' keys
            
        Returns:
            Dict with precision, recall, and f1 scores, or None if evaluation fails
        """
        if not generated_flashcards:
            logger.warning("No flashcards provided for BERTScore evaluation")
            return None

        try:
            questions = [card['question'] for card in generated_flashcards]
            answers = [card['answer'] for card in generated_flashcards]
            
            # Create reference texts for comparison
            source_texts = [source_text] * len(questions)
            
            # Score questions against source
            P_q, R_q, F1_q = bert_scorer(questions, source_texts, lang='en', verbose=False)
            
            # Score answers against source
            P_a, R_a, F1_a = bert_scorer(answers, source_texts, lang='en', verbose=False)
            
            # Calculate average scores
            avg_precision = (P_q.mean() + P_a.mean()) / 2
            avg_recall = (R_q.mean() + R_a.mean()) / 2
            avg_f1 = (F1_q.mean() + F1_a.mean()) / 2
            
            return {
                'precision': avg_precision.item(),
                'recall': avg_recall.item(),
                'f1': avg_f1.item(),
                'question_f1': F1_q.mean().item(),
                'answer_f1': F1_a.mean().item()
            }
            
        except Exception as e:
            logger.error(f"BERTScore evaluation failed: {e}")
            return None

    def evaluate_keyword_coverage(self, source_text: str, generated_flashcards: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Calculate keyword coverage percentage and identify covered/missed keywords.
        
        Args:
            source_text: Original text used for flashcard generation
            generated_flashcards: List of dicts with 'question' and 'answer' keys
            
        Returns:
            Dict with coverage score, total keywords, covered keywords, and keyword lists
        """
        if not generated_flashcards:
            logger.warning("No flashcards provided for keyword coverage evaluation")
            return {'coverage_score': 0.0, 'total_keywords': 0, 'covered_keywords': 0, 'keywords': []}

        try:
            # Extract keywords using improved regex-based approach
            words = re.findall(r'\b[A-Za-z]+\b', source_text)
            
            source_keywords = set()
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
                'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
                'did', 'she', 'use', 'way', 'many', 'then', 'them', 'these', 'some', 
                'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 
                'over', 'such', 'take', 'than', 'they', 'well', 'were', 'will', 'with',
                'have', 'this', 'that', 'from', 'been', 'each', 'which', 'their', 'said',
                'would', 'there', 'what', 'could', 'other', 'after', 'first', 'never',
                'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall',
                'still', 'those', 'while', 'should', 'never', 'through', 'before', 'here',
                'between', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such'
            }
            
            for word in words:
                word_lower = word.lower()
                # Include capitalized words (proper nouns) or longer content words
                if (word[0].isupper() or len(word) > 4) and len(word) > 2:
                    if word_lower not in stop_words:
                        source_keywords.add(word_lower)
            
            if not source_keywords:
                logger.warning("No keywords found in source text for coverage analysis")
                return {'coverage_score': 0.0, 'total_keywords': 0, 'covered_keywords': 0, 'keywords': []}

            # Combine all flashcard text
            flashcards_text = " ".join([f"{card['question']} {card['answer']}" for card in generated_flashcards]).lower()
            
            # Count covered keywords
            covered_keywords = []
            missed_keywords = []
            
            for keyword in source_keywords:
                if keyword in flashcards_text:
                    covered_keywords.append(keyword)
                else:
                    missed_keywords.append(keyword)
            
            coverage_score = len(covered_keywords) / len(source_keywords)
            
            return {
                'coverage_score': coverage_score,
                'total_keywords': len(source_keywords),
                'covered_keywords': len(covered_keywords),
                'keywords': sorted(list(source_keywords)),
                'covered_keyword_list': sorted(covered_keywords),
                'missed_keyword_list': sorted(missed_keywords)
            }
            
        except Exception as e:
            logger.error(f"Keyword coverage evaluation failed: {e}")
            return {'coverage_score': 0.0, 'total_keywords': 0, 'covered_keywords': 0, 'keywords': []}

    def comprehensive_evaluate(self, source_text: str, generated_flashcards: List[Dict[str, str]], 
                             deck_id: Optional[int] = None, user_id: Optional[int] = None) -> Dict[str, any]:
        """
        Perform comprehensive evaluation using both metrics and store results.
        
        Args:
            source_text: Original text used for flashcard generation
            generated_flashcards: List of dicts with 'question' and 'answer' keys
            deck_id: Optional deck ID for database storage
            user_id: Optional user ID for database storage
            
        Returns:
            Dict with comprehensive evaluation results
        """
        logger.info(f"Starting comprehensive evaluation for {len(generated_flashcards)} flashcards")
        
        # Perform BERTScore evaluation
        bert_results = self.evaluate_bert_score(source_text, generated_flashcards)
        
        # Perform keyword coverage evaluation
        keyword_results = self.evaluate_keyword_coverage(source_text, generated_flashcards)
        
        # Create comprehensive results
        evaluation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'num_flashcards': len(generated_flashcards),
            'source_text_length': len(source_text),
            'bert_score': bert_results,
            'keyword_coverage': keyword_results,
            'deck_id': deck_id,
            'user_id': user_id
        }
        
        # Store results in database if IDs provided
        if deck_id is not None:
            self._store_evaluation_results(evaluation_results)
        
        logger.info("Comprehensive evaluation completed successfully")
        return evaluation_results

    def _store_evaluation_results(self, results: Dict[str, any]) -> None:
        """Store evaluation results in database for later analysis."""
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # Store in evaluation_results table
            cur.execute("""
                INSERT INTO evaluation_results (
                    deck_id, user_id, timestamp, num_flashcards, source_text_length,
                    bert_precision, bert_recall, bert_f1, question_f1, answer_f1,
                    keyword_coverage_score, total_keywords, covered_keywords,
                    evaluation_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results['deck_id'],
                results['user_id'], 
                results['timestamp'],
                results['num_flashcards'],
                results['source_text_length'],
                results['bert_score']['precision'] if results['bert_score'] else None,
                results['bert_score']['recall'] if results['bert_score'] else None,
                results['bert_score']['f1'] if results['bert_score'] else None,
                results['bert_score']['question_f1'] if results['bert_score'] else None,
                results['bert_score']['answer_f1'] if results['bert_score'] else None,
                results['keyword_coverage']['coverage_score'],
                results['keyword_coverage']['total_keywords'],
                results['keyword_coverage']['covered_keywords'],
                json.dumps(results)  # Store full results as JSON
            ))
            
            conn.commit()
            conn.close()
            logger.info("Evaluation results stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store evaluation results: {e}")

    def get_evaluation_history(self, user_id: Optional[int] = None, deck_id: Optional[int] = None, 
                             limit: int = 50) -> List[Dict[str, any]]:
        """
        Retrieve evaluation history from database.
        
        Args:
            user_id: Filter by user ID (optional)
            deck_id: Filter by deck ID (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of evaluation result dictionaries
        """
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            where_clause = "WHERE 1=1"
            params = []
            
            if user_id is not None:
                where_clause += " AND user_id = ?"
                params.append(user_id)
                
            if deck_id is not None:
                where_clause += " AND deck_id = ?"
                params.append(deck_id)
            
            params.append(limit)
            
            cur.execute(f"""
                SELECT * FROM evaluation_results 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ?
            """, params)
            
            rows = cur.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cur.description]
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to retrieve evaluation history: {e}")
            return []

    def get_evaluation_statistics(self, user_id: Optional[int] = None) -> Dict[str, any]:
        """
        Get aggregated evaluation statistics.
        
        Args:
            user_id: Filter by user ID (optional)
            
        Returns:
            Dict with statistical summary of evaluations
        """
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            where_clause = ""
            params = []
            
            if user_id is not None:
                where_clause = "WHERE user_id = ?"
                params.append(user_id)
            
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(bert_f1) as avg_bert_f1,
                    AVG(keyword_coverage_score) as avg_keyword_coverage,
                    AVG(num_flashcards) as avg_flashcards_per_deck,
                    MIN(bert_f1) as min_bert_f1,
                    MAX(bert_f1) as max_bert_f1,
                    MIN(keyword_coverage_score) as min_keyword_coverage,
                    MAX(keyword_coverage_score) as max_keyword_coverage
                FROM evaluation_results 
                {where_clause}
                AND bert_f1 IS NOT NULL
            """, params)
            
            stats = cur.fetchone()
            conn.close()
            
            if stats and stats[0] > 0:  # If we have evaluations
                return {
                    'total_evaluations': stats[0],
                    'avg_bert_f1': stats[1],
                    'avg_keyword_coverage': stats[2],
                    'avg_flashcards_per_deck': stats[3],
                    'bert_f1_range': {'min': stats[4], 'max': stats[5]},
                    'keyword_coverage_range': {'min': stats[6], 'max': stats[7]}
                }
            else:
                return {'total_evaluations': 0}
                
        except Exception as e:
            logger.error(f"Failed to get evaluation statistics: {e}")
            return {'total_evaluations': 0}


# Global evaluator instance
_evaluator_instance = None

def get_evaluator() -> FlashcardEvaluator:
    """Get singleton evaluator instance."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = FlashcardEvaluator()
    return _evaluator_instance


def evaluate_flashcard_generation(source_text: str, generated_flashcards: List[Dict[str, str]], 
                                deck_id: Optional[int] = None, user_id: Optional[int] = None) -> Dict[str, any]:
    """
    Convenience function for comprehensive flashcard evaluation.
    
    Args:
        source_text: Original text used for flashcard generation
        generated_flashcards: List of dicts with 'question' and 'answer' keys
        deck_id: Optional deck ID for database storage
        user_id: Optional user ID for database storage
        
    Returns:
        Dict with comprehensive evaluation results
    """
    evaluator = get_evaluator()
    return evaluator.comprehensive_evaluate(source_text, generated_flashcards, deck_id, user_id)
