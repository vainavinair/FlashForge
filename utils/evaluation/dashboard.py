"""
Evaluation Dashboard and Reporting for FlashForge

This module provides dashboard functionality and reporting capabilities
for comprehensive evaluation metrics analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

from utils.datamodels.db import get_connection
from utils.evaluation.evaluator import get_evaluator

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """Dashboard for evaluation metrics and analytics."""
    
    def __init__(self):
        self.evaluator = get_evaluator()
    
    def get_system_wide_metrics(self) -> Dict[str, Any]:
        """Get system-wide evaluation metrics across all users and decks."""
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # Overall statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT deck_id) as unique_decks,
                    AVG(bert_f1) as avg_bert_f1,
                    STDDEV(bert_f1) as std_bert_f1,
                    AVG(keyword_coverage_score) as avg_keyword_coverage,
                    STDDEV(keyword_coverage_score) as std_keyword_coverage,
                    AVG(num_flashcards) as avg_flashcards_per_generation,
                    MIN(timestamp) as first_evaluation,
                    MAX(timestamp) as latest_evaluation
                FROM evaluation_results 
                WHERE bert_f1 IS NOT NULL
            """)
            
            stats = cur.fetchone()
            
            if not stats or stats[0] == 0:
                conn.close()
                return {'total_evaluations': 0, 'message': 'No evaluations found'}
            
            # Quality distribution
            cur.execute("""
                SELECT 
                    COUNT(CASE WHEN bert_f1 >= 0.8 THEN 1 END) as excellent_quality,
                    COUNT(CASE WHEN bert_f1 >= 0.6 AND bert_f1 < 0.8 THEN 1 END) as good_quality,
                    COUNT(CASE WHEN bert_f1 >= 0.4 AND bert_f1 < 0.6 THEN 1 END) as fair_quality,
                    COUNT(CASE WHEN bert_f1 < 0.4 THEN 1 END) as poor_quality
                FROM evaluation_results 
                WHERE bert_f1 IS NOT NULL
            """)
            
            quality_dist = cur.fetchone()
            
            # Recent trends (last 30 days)
            cur.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as evaluations,
                    AVG(bert_f1) as avg_bert_f1,
                    AVG(keyword_coverage_score) as avg_coverage
                FROM evaluation_results 
                WHERE bert_f1 IS NOT NULL 
                AND timestamp >= datetime('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            
            trends = cur.fetchall()
            conn.close()
            
            return {
                'total_evaluations': stats[0],
                'unique_users': stats[1],
                'unique_decks': stats[2],
                'avg_bert_f1': stats[3],
                'std_bert_f1': stats[4],
                'avg_keyword_coverage': stats[5],
                'std_keyword_coverage': stats[6],
                'avg_flashcards_per_generation': stats[7],
                'evaluation_period': {
                    'start': stats[8],
                    'end': stats[9]
                },
                'quality_distribution': {
                    'excellent': quality_dist[0],  # >= 0.8
                    'good': quality_dist[1],       # 0.6-0.8
                    'fair': quality_dist[2],       # 0.4-0.6
                    'poor': quality_dist[3]        # < 0.4
                },
                'recent_trends': [
                    {
                        'date': row[0],
                        'evaluations': row[1],
                        'avg_bert_f1': row[2],
                        'avg_coverage': row[3]
                    } for row in trends
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get system-wide metrics: {e}")
            return {'error': str(e)}
    
    def get_user_detailed_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get detailed evaluation metrics for a specific user."""
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # User-specific statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    COUNT(DISTINCT deck_id) as user_decks,
                    AVG(bert_f1) as avg_bert_f1,
                    AVG(keyword_coverage_score) as avg_keyword_coverage,
                    AVG(num_flashcards) as avg_flashcards,
                    MIN(bert_f1) as min_bert_f1,
                    MAX(bert_f1) as max_bert_f1,
                    MIN(keyword_coverage_score) as min_coverage,
                    MAX(keyword_coverage_score) as max_coverage
                FROM evaluation_results 
                WHERE user_id = ? AND bert_f1 IS NOT NULL
            """, (user_id,))
            
            user_stats = cur.fetchone()
            
            if not user_stats or user_stats[0] == 0:
                conn.close()
                return {'user_id': user_id, 'total_evaluations': 0}
            
            # Per-deck breakdown
            cur.execute("""
                SELECT 
                    er.deck_id,
                    d.name as deck_name,
                    COUNT(*) as evaluations,
                    AVG(bert_f1) as avg_bert_f1,
                    AVG(keyword_coverage_score) as avg_coverage,
                    MAX(timestamp) as latest_evaluation
                FROM evaluation_results er
                JOIN decks d ON er.deck_id = d.deck_id
                WHERE er.user_id = ? AND bert_f1 IS NOT NULL
                GROUP BY er.deck_id, d.name
                ORDER BY latest_evaluation DESC
            """, (user_id,))
            
            deck_breakdown = cur.fetchall()
            
            # Improvement over time
            cur.execute("""
                SELECT 
                    timestamp,
                    bert_f1,
                    keyword_coverage_score,
                    num_flashcards
                FROM evaluation_results 
                WHERE user_id = ? AND bert_f1 IS NOT NULL
                ORDER BY timestamp ASC
            """, (user_id,))
            
            time_series = cur.fetchall()
            conn.close()
            
            return {
                'user_id': user_id,
                'total_evaluations': user_stats[0],
                'user_decks': user_stats[1],
                'avg_bert_f1': user_stats[2],
                'avg_keyword_coverage': user_stats[3],
                'avg_flashcards': user_stats[4],
                'performance_range': {
                    'bert_f1': {'min': user_stats[5], 'max': user_stats[6]},
                    'keyword_coverage': {'min': user_stats[7], 'max': user_stats[8]}
                },
                'deck_breakdown': [
                    {
                        'deck_id': row[0],
                        'deck_name': row[1],
                        'evaluations': row[2],
                        'avg_bert_f1': row[3],
                        'avg_coverage': row[4],
                        'latest_evaluation': row[5]
                    } for row in deck_breakdown
                ],
                'time_series': [
                    {
                        'timestamp': row[0],
                        'bert_f1': row[1],
                        'keyword_coverage': row[2],
                        'num_flashcards': row[3]
                    } for row in time_series
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get user detailed metrics: {e}")
            return {'error': str(e)}
    
    def generate_evaluation_report(self, user_id: Optional[int] = None, 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            user_id: Optional user ID to filter by
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            conn = get_connection()
            cur = conn.cursor()
            
            # Build query conditions
            conditions = ["bert_f1 IS NOT NULL"]
            params = []
            
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
                
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Main statistics
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(bert_f1) as avg_bert_f1,
                    AVG(bert_precision) as avg_bert_precision,
                    AVG(bert_recall) as avg_bert_recall,
                    AVG(keyword_coverage_score) as avg_keyword_coverage,
                    AVG(num_flashcards) as avg_num_flashcards,
                    AVG(total_keywords) as avg_total_keywords,
                    AVG(covered_keywords) as avg_covered_keywords
                FROM evaluation_results 
                {where_clause}
            """, params)
            
            main_stats = cur.fetchone()
            
            # Quality benchmarks
            cur.execute(f"""
                SELECT 
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bert_f1) as bert_f1_q25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY bert_f1) as bert_f1_median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bert_f1) as bert_f1_q75,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY keyword_coverage_score) as coverage_q25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY keyword_coverage_score) as coverage_median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY keyword_coverage_score) as coverage_q75
                FROM evaluation_results 
                {where_clause}
            """, params)
            
            percentiles = cur.fetchone()
            conn.close()
            
            if not main_stats or main_stats[0] == 0:
                return {
                    'report_generated': datetime.utcnow().isoformat(),
                    'filters': {'user_id': user_id, 'start_date': start_date, 'end_date': end_date},
                    'total_evaluations': 0,
                    'message': 'No evaluations found for the specified criteria'
                }
            
            return {
                'report_generated': datetime.utcnow().isoformat(),
                'filters': {
                    'user_id': user_id,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'summary': {
                    'total_evaluations': main_stats[0],
                    'avg_bert_f1': main_stats[1],
                    'avg_bert_precision': main_stats[2],
                    'avg_bert_recall': main_stats[3],
                    'avg_keyword_coverage': main_stats[4],
                    'avg_flashcards_per_generation': main_stats[5],
                    'avg_keywords_per_source': main_stats[6],
                    'avg_keywords_covered': main_stats[7]
                },
                'quality_benchmarks': {
                    'bert_f1': {
                        'q25': percentiles[0],
                        'median': percentiles[1],
                        'q75': percentiles[2]
                    },
                    'keyword_coverage': {
                        'q25': percentiles[3],
                        'median': percentiles[4], 
                        'q75': percentiles[5]
                    }
                },
                'interpretation': self._generate_interpretation(main_stats, percentiles)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            return {'error': str(e)}
    
    def _generate_interpretation(self, main_stats, percentiles) -> Dict[str, str]:
        """Generate human-readable interpretation of evaluation results."""
        interpretations = {}
        
        avg_bert_f1 = main_stats[1]
        avg_coverage = main_stats[4]
        
        # BERTScore interpretation
        if avg_bert_f1 >= 0.8:
            interpretations['bert_quality'] = "Excellent semantic similarity between generated flashcards and source material"
        elif avg_bert_f1 >= 0.6:
            interpretations['bert_quality'] = "Good semantic alignment with source material"
        elif avg_bert_f1 >= 0.4:
            interpretations['bert_quality'] = "Fair semantic similarity, some improvement possible"
        else:
            interpretations['bert_quality'] = "Poor semantic alignment, significant improvement needed"
        
        # Coverage interpretation
        if avg_coverage >= 0.7:
            interpretations['coverage_quality'] = "Excellent keyword coverage, captures most important concepts"
        elif avg_coverage >= 0.5:
            interpretations['coverage_quality'] = "Good keyword coverage, captures key concepts well"
        elif avg_coverage >= 0.3:
            interpretations['coverage_quality'] = "Fair keyword coverage, missing some important concepts"
        else:
            interpretations['coverage_quality'] = "Poor keyword coverage, many concepts not captured"
        
        # Overall assessment
        overall_score = (avg_bert_f1 + avg_coverage) / 2
        if overall_score >= 0.75:
            interpretations['overall'] = "High-quality flashcard generation with strong semantic alignment and concept coverage"
        elif overall_score >= 0.55:
            interpretations['overall'] = "Good flashcard generation quality with room for minor improvements"
        elif overall_score >= 0.35:
            interpretations['overall'] = "Moderate flashcard quality, consider optimizing generation parameters"
        else:
            interpretations['overall'] = "Low flashcard quality, significant improvements recommended"
        
        return interpretations


# Global dashboard instance
_dashboard_instance = None

def get_dashboard() -> EvaluationDashboard:
    """Get singleton dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = EvaluationDashboard()
    return _dashboard_instance
