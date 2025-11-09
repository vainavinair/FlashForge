"""
Batch Evaluation System for Research and Paper Metrics

This module provides batch evaluation capabilities for systematic evaluation
of flashcard generation quality across multiple documents and datasets.
Perfect for generating research metrics and paper results.
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", message="Could get FontBBox from font descriptor")

from utils.generation.enhanced_flashcard_gen import generate_flashcards_with_evaluation
from utils.extraction.text_handler import load_txt, load_docx
from utils.extraction.pdf_handler import load_pdf, extract_text_or_ocr
from utils.extraction.text_cleaner import clean_text_for_llm
from utils.evaluation.evaluator import get_evaluator
from utils.evaluation.dashboard import get_dashboard

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """
    Comprehensive batch evaluation system for research purposes.
    
    Supports:
    - Batch processing of multiple documents
    - Statistical analysis across datasets
    - Export to various formats for research
    - Parallel processing for efficiency
    """
    
    def __init__(self, max_workers: int = 4):
        self.evaluator = get_evaluator()
        self.dashboard = get_dashboard()
        self.max_workers = max_workers
        
    def evaluate_document_batch(self, document_paths: List[str], 
                               output_dir: str = "evaluation_results",
                               enable_parallel: bool = True) -> Dict[str, Any]:
        """
        Evaluate a batch of documents and generate comprehensive metrics.
        
        Args:
            document_paths: List of paths to documents (PDF, TXT, DOCX)
            output_dir: Directory to save results
            enable_parallel: Whether to use parallel processing
            
        Returns:
            Dict with batch evaluation results and statistics
        """
        logger.info(f"Starting batch evaluation of {len(document_paths)} documents")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        failed_documents = []
        start_time = time.time()
        
        if enable_parallel and len(document_paths) > 1:
            results, failed_documents = self._evaluate_parallel(document_paths)
        else:
            results, failed_documents = self._evaluate_sequential(document_paths)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Generate comprehensive statistics
        stats = self._calculate_batch_statistics(results)
        
        # Create batch report
        batch_report = {
            'batch_metadata': {
                'total_documents': len(document_paths),
                'successful_evaluations': len(results),
                'failed_evaluations': len(failed_documents),
                'processing_time_seconds': processing_time,
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'parallel_processing': enable_parallel
            },
            'statistical_summary': stats,
            'individual_results': results,
            'failed_documents': failed_documents
        }
        
        # Save results
        self._save_batch_results(batch_report, output_dir)
        
        logger.info(f"Batch evaluation completed in {processing_time:.2f}s. "
                   f"{len(results)} successful, {len(failed_documents)} failed")
        
        return batch_report
    
    def _evaluate_parallel(self, document_paths: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Evaluate documents in parallel."""
        results = []
        failed_documents = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._evaluate_single_document, path): path 
                for path in document_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed_documents.append({
                            'document_path': path,
                            'error': 'No result returned'
                        })
                except Exception as e:
                    logger.error(f"Failed to evaluate {path}: {e}")
                    failed_documents.append({
                        'document_path': path,
                        'error': str(e)
                    })
        
        return results, failed_documents
    
    def _evaluate_sequential(self, document_paths: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Evaluate documents sequentially."""
        results = []
        failed_documents = []
        
        for i, path in enumerate(document_paths):
            logger.info(f"Processing document {i+1}/{len(document_paths)}: {path}")
            try:
                result = self._evaluate_single_document(path)
                if result:
                    results.append(result)
                else:
                    failed_documents.append({
                        'document_path': path,
                        'error': 'No result returned'
                    })
            except Exception as e:
                logger.error(f"Failed to evaluate {path}: {e}")
                failed_documents.append({
                    'document_path': path,
                    'error': str(e)
                })
        
        return results, failed_documents
    
    def _evaluate_single_document(self, document_path: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single document."""
        try:
            # Extract text from document
            text = self._extract_text_from_document(document_path)
            if not text or len(text.strip()) < 100:  # Minimum text length
                logger.warning(f"Insufficient text extracted from {document_path}")
                return None
            
            # Clean text
            cleaned_text = clean_text_for_llm(text)
            
            # Generate flashcards with evaluation
            flashcards, evaluation_results = generate_flashcards_with_evaluation(
                cleaned_text, 
                enable_evaluation=True
            )
            
            if not flashcards or not evaluation_results:
                return None
            
            # Prepare result
            result = {
                'document_path': document_path,
                'document_name': os.path.basename(document_path),
                'source_text_length': len(text),
                'cleaned_text_length': len(cleaned_text),
                'num_flashcards': len(flashcards),
                'evaluation_metrics': evaluation_results,
                'flashcards': flashcards,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating document {document_path}: {e}")
            raise
    
    def _extract_text_from_document(self, document_path: str) -> str:
        """Extract text from various document formats."""
        ext = os.path.splitext(document_path)[1].lower()
        
        if ext == '.pdf':
            pdf = load_pdf(document_path)
            if pdf:
                text = extract_text_or_ocr(pdf)
                pdf.close()
                return text
        elif ext == '.txt':
            return load_txt(document_path)
        elif ext == '.docx':
            return load_docx(document_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")
        
        return ""
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics across the batch."""
        if not results:
            return {'message': 'No results to analyze'}
        
        # Extract metrics
        bert_f1_scores = []
        bert_precision_scores = []
        bert_recall_scores = []
        coverage_scores = []
        flashcard_counts = []
        text_lengths = []
        keyword_counts = []
        
        for result in results:
            eval_metrics = result['evaluation_metrics']
            if eval_metrics and eval_metrics.get('bert_score'):
                bert_f1_scores.append(eval_metrics['bert_score']['f1'])
                bert_precision_scores.append(eval_metrics['bert_score']['precision'])
                bert_recall_scores.append(eval_metrics['bert_score']['recall'])
            
            if eval_metrics and eval_metrics.get('keyword_coverage'):
                coverage_scores.append(eval_metrics['keyword_coverage']['coverage_score'])
                keyword_counts.append(eval_metrics['keyword_coverage']['total_keywords'])
            
            flashcard_counts.append(result['num_flashcards'])
            text_lengths.append(result['source_text_length'])
        
        # Calculate statistics
        stats = {
            'sample_size': len(results),
            'bert_score_metrics': {
                'f1': self._calculate_descriptive_stats(bert_f1_scores),
                'precision': self._calculate_descriptive_stats(bert_precision_scores),
                'recall': self._calculate_descriptive_stats(bert_recall_scores)
            },
            'keyword_coverage_metrics': {
                'coverage_score': self._calculate_descriptive_stats(coverage_scores),
                'keywords_per_document': self._calculate_descriptive_stats(keyword_counts)
            },
            'document_characteristics': {
                'flashcards_per_document': self._calculate_descriptive_stats(flashcard_counts),
                'text_length_per_document': self._calculate_descriptive_stats(text_lengths)
            },
            'quality_distribution': self._calculate_quality_distribution(bert_f1_scores, coverage_scores)
        }
        
        return stats
    
    def _calculate_descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a list of values."""
        if not values:
            return {'count': 0}
        
        values = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _calculate_quality_distribution(self, bert_scores: List[float], 
                                      coverage_scores: List[float]) -> Dict[str, Any]:
        """Calculate quality distribution across different thresholds."""
        if not bert_scores or not coverage_scores:
            return {}
        
        # Combined quality score
        combined_scores = [(b + c) / 2 for b, c in zip(bert_scores, coverage_scores)]
        
        return {
            'bert_f1_distribution': {
                'excellent_ge_0.8': sum(1 for s in bert_scores if s >= 0.8),
                'good_0.6_to_0.8': sum(1 for s in bert_scores if 0.6 <= s < 0.8),
                'fair_0.4_to_0.6': sum(1 for s in bert_scores if 0.4 <= s < 0.6),
                'poor_lt_0.4': sum(1 for s in bert_scores if s < 0.4)
            },
            'coverage_distribution': {
                'excellent_ge_0.7': sum(1 for s in coverage_scores if s >= 0.7),
                'good_0.5_to_0.7': sum(1 for s in coverage_scores if 0.5 <= s < 0.7),
                'fair_0.3_to_0.5': sum(1 for s in coverage_scores if 0.3 <= s < 0.5),
                'poor_lt_0.3': sum(1 for s in coverage_scores if s < 0.3)
            },
            'combined_quality': {
                'high_ge_0.75': sum(1 for s in combined_scores if s >= 0.75),
                'medium_0.55_to_0.75': sum(1 for s in combined_scores if 0.55 <= s < 0.75),
                'low_0.35_to_0.55': sum(1 for s in combined_scores if 0.35 <= s < 0.55),
                'very_low_lt_0.35': sum(1 for s in combined_scores if s < 0.35)
            }
        }
    
    def _save_batch_results(self, batch_report: Dict[str, Any], output_dir: str):
        """Save batch results in multiple formats."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save complete JSON report
        json_path = os.path.join(output_dir, f"batch_evaluation_report_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary for easy analysis
        csv_path = os.path.join(output_dir, f"batch_evaluation_summary_{timestamp}.csv")
        self._save_csv_summary(batch_report['individual_results'], csv_path)
        
        # Save statistical summary
        stats_path = os.path.join(output_dir, f"batch_statistics_{timestamp}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report['statistical_summary'], f, indent=2)
        
        logger.info(f"Batch results saved to {output_dir}")
    
    def _save_csv_summary(self, results: List[Dict[str, Any]], csv_path: str):
        """Save results summary as CSV for analysis."""
        if not results:
            return
        
        rows = []
        for result in results:
            eval_metrics = result['evaluation_metrics']
            row = {
                'document_name': result['document_name'],
                'source_text_length': result['source_text_length'],
                'num_flashcards': result['num_flashcards'],
                'bert_f1': eval_metrics['bert_score']['f1'] if eval_metrics and eval_metrics.get('bert_score') else None,
                'bert_precision': eval_metrics['bert_score']['precision'] if eval_metrics and eval_metrics.get('bert_score') else None,
                'bert_recall': eval_metrics['bert_score']['recall'] if eval_metrics and eval_metrics.get('bert_score') else None,
                'keyword_coverage': eval_metrics['keyword_coverage']['coverage_score'] if eval_metrics and eval_metrics.get('keyword_coverage') else None,
                'total_keywords': eval_metrics['keyword_coverage']['total_keywords'] if eval_metrics and eval_metrics.get('keyword_coverage') else None,
                'covered_keywords': eval_metrics['keyword_coverage']['covered_keywords'] if eval_metrics and eval_metrics.get('keyword_coverage') else None,
                'processing_timestamp': result['processing_timestamp']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    
    def evaluate_existing_decks(self, user_id: Optional[int] = None, 
                               deck_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Evaluate existing decks in the database for research purposes.
        
        Args:
            user_id: Optional user ID to filter by
            deck_ids: Optional list of specific deck IDs to evaluate
            
        Returns:
            Dict with evaluation results for existing decks
        """
        logger.info("Evaluating existing decks in database")
        
        try:
            from utils.datamodels.db import get_connection
            
            conn = get_connection()
            cur = conn.cursor()
            
            # Build query to get source texts and associated flashcards
            conditions = []
            params = []
            
            if user_id is not None:
                conditions.append("st.user_id = ?")
                params.append(user_id)
            
            if deck_ids:
                placeholders = ','.join('?' * len(deck_ids))
                conditions.append(f"st.deck_id IN ({placeholders})")
                params.extend(deck_ids)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Get source texts with their decks
            cur.execute(f"""
                SELECT st.source_id, st.deck_id, st.user_id, st.content, st.filename,
                       d.name as deck_name
                FROM source_texts st
                JOIN decks d ON st.deck_id = d.deck_id
                {where_clause}
                ORDER BY st.upload_timestamp DESC
            """, params)
            
            source_texts = cur.fetchall()
            
            if not source_texts:
                conn.close()
                return {'message': 'No source texts found for evaluation'}
            
            results = []
            
            for source_text in source_texts:
                source_id, deck_id, user_id, content, filename, deck_name = source_text
                
                # Get flashcards for this deck
                cur.execute("""
                    SELECT question, answer FROM cards 
                    WHERE deck_id = ?
                    ORDER BY created_at ASC
                """, (deck_id,))
                
                flashcard_rows = cur.fetchall()
                flashcards = [{'question': row[0], 'answer': row[1]} for row in flashcard_rows]
                
                if flashcards:
                    # Evaluate these flashcards against the source
                    evaluation_results = self.evaluator.comprehensive_evaluate(
                        content, flashcards, deck_id, user_id
                    )
                    
                    results.append({
                        'deck_id': deck_id,
                        'deck_name': deck_name,
                        'user_id': user_id,
                        'source_filename': filename,
                        'num_flashcards': len(flashcards),
                        'evaluation_results': evaluation_results
                    })
            
            conn.close()
            
            # Generate statistics
            stats = self._calculate_existing_deck_statistics(results)
            
            return {
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'total_decks_evaluated': len(results),
                'statistical_summary': stats,
                'deck_results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate existing decks: {e}")
            return {'error': str(e)}
    
    def _calculate_existing_deck_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for existing deck evaluations."""
        if not results:
            return {'message': 'No results to analyze'}
        
        bert_scores = []
        coverage_scores = []
        
        for result in results:
            eval_results = result['evaluation_results']
            if eval_results and eval_results.get('bert_score'):
                bert_scores.append(eval_results['bert_score']['f1'])
            if eval_results and eval_results.get('keyword_coverage'):
                coverage_scores.append(eval_results['keyword_coverage']['coverage_score'])
        
        return {
            'sample_size': len(results),
            'bert_f1_statistics': self._calculate_descriptive_stats(bert_scores),
            'coverage_statistics': self._calculate_descriptive_stats(coverage_scores),
            'quality_distribution': self._calculate_quality_distribution(bert_scores, coverage_scores)
        }


# Global batch evaluator instance
_batch_evaluator_instance = None

def get_batch_evaluator() -> BatchEvaluator:
    """Get singleton batch evaluator instance."""
    global _batch_evaluator_instance
    if _batch_evaluator_instance is None:
        _batch_evaluator_instance = BatchEvaluator()
    return _batch_evaluator_instance
