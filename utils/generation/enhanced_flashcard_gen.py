"""
Enhanced flashcard generation with integrated evaluation capabilities.

This module wraps the original flashcard generation with comprehensive evaluation
using BERTScore and Keyword Coverage metrics for research and quality assessment.
"""

import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from utils.generation.flashcard_gen import generate_flashcards as _original_generate_flashcards
from utils.evaluation.evaluator import evaluate_flashcard_generation
from utils.datamodels.db import get_connection

logger = logging.getLogger(__name__)


def store_source_text(deck_id: int, user_id: int, content: str, filename: Optional[str] = None) -> int:
    """
    Store source text in database for evaluation purposes.
    
    Args:
        deck_id: ID of the deck
        user_id: ID of the user
        content: Source text content
        filename: Optional filename
        
    Returns:
        source_id: ID of stored source text
    """
    try:
        # Create content hash for deduplication
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Check if this content already exists for this deck
        cur.execute("""
            SELECT source_id FROM source_texts 
            WHERE deck_id = ? AND content_hash = ?
        """, (deck_id, content_hash))
        
        existing = cur.fetchone()
        if existing:
            conn.close()
            return existing[0]
        
        # Store new source text
        cur.execute("""
            INSERT INTO source_texts (deck_id, user_id, filename, content, content_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (deck_id, user_id, filename, content, content_hash))
        
        source_id = cur.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Stored source text with ID {source_id} for deck {deck_id}")
        return source_id
        
    except Exception as e:
        logger.error(f"Failed to store source text: {e}")
        return None


def generate_flashcards_with_evaluation(text: str, deck_id: Optional[int] = None, 
                                       user_id: Optional[int] = None, 
                                       filename: Optional[str] = None,
                                       enable_evaluation: bool = True) -> Tuple[List[Dict[str, str]], Optional[Dict]]:
    """
    Generate flashcards with integrated evaluation capabilities.
    
    Args:
        text: Source text for flashcard generation
        deck_id: Optional deck ID for database storage
        user_id: Optional user ID for database storage  
        filename: Optional source filename
        enable_evaluation: Whether to perform evaluation (default: True)
        
    Returns:
        Tuple of (flashcards, evaluation_results)
        - flashcards: List of generated flashcard dicts
        - evaluation_results: Dict with evaluation metrics or None if disabled
    """
    logger.info(f"Generating flashcards with evaluation for text of length {len(text)}")
    
    # Generate flashcards using original function
    flashcards = _original_generate_flashcards(text)
    
    if not flashcards:
        logger.warning("No flashcards generated, skipping evaluation")
        return flashcards, None
    
    evaluation_results = None
    
    if enable_evaluation:
        try:
            # Store source text if deck_id provided
            if deck_id is not None and user_id is not None:
                store_source_text(deck_id, user_id, text, filename)
            
            # Perform comprehensive evaluation
            evaluation_results = evaluate_flashcard_generation(
                source_text=text,
                generated_flashcards=flashcards,
                deck_id=deck_id,
                user_id=user_id
            )
            
            logger.info(f"Evaluation completed - BERTScore F1: {evaluation_results['bert_score']['f1']:.3f}, "
                       f"Keyword Coverage: {evaluation_results['keyword_coverage']['coverage_score']:.3f}")
                       
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation_results = None
    
    return flashcards, evaluation_results


def generate_flashcards(text: str) -> List[Dict[str, str]]:
    """
    Backward-compatible wrapper for original flashcard generation.
    This maintains compatibility with existing code while enabling evaluation when needed.
    """
    flashcards, _ = generate_flashcards_with_evaluation(text, enable_evaluation=False)
    return flashcards


def get_deck_evaluation_summary(deck_id: int) -> Optional[Dict]:
    """
    Get evaluation summary for a specific deck.
    
    Args:
        deck_id: ID of the deck
        
    Returns:
        Dict with evaluation summary or None if no evaluations found
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Get latest evaluation for this deck
        cur.execute("""
            SELECT 
                bert_f1, bert_precision, bert_recall,
                keyword_coverage_score, total_keywords, covered_keywords,
                num_flashcards, timestamp
            FROM evaluation_results 
            WHERE deck_id = ? AND bert_f1 IS NOT NULL
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (deck_id,))
        
        result = cur.fetchone()
        conn.close()
        
        if result:
            return {
                'bert_f1': result[0],
                'bert_precision': result[1], 
                'bert_recall': result[2],
                'keyword_coverage': result[3],
                'total_keywords': result[4],
                'covered_keywords': result[5],
                'num_flashcards': result[6],
                'timestamp': result[7]
            }
        return None
        
    except Exception as e:
        logger.error(f"Failed to get deck evaluation summary: {e}")
        return None


def get_user_evaluation_overview(user_id: int) -> Dict:
    """
    Get evaluation overview for a user across all their decks.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dict with user evaluation overview
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Get aggregated evaluation metrics for user
        cur.execute("""
            SELECT 
                COUNT(*) as total_evaluations,
                AVG(bert_f1) as avg_bert_f1,
                AVG(keyword_coverage_score) as avg_keyword_coverage,
                AVG(num_flashcards) as avg_flashcards,
                MAX(timestamp) as latest_evaluation
            FROM evaluation_results 
            WHERE user_id = ? AND bert_f1 IS NOT NULL
        """, (user_id,))
        
        result = cur.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            return {
                'total_evaluations': result[0],
                'avg_bert_f1': result[1],
                'avg_keyword_coverage': result[2],
                'avg_flashcards': result[3],
                'latest_evaluation': result[4],
                'has_evaluations': True
            }
        else:
            return {'has_evaluations': False, 'total_evaluations': 0}
            
    except Exception as e:
        logger.error(f"Failed to get user evaluation overview: {e}")
        return {'has_evaluations': False, 'total_evaluations': 0}
