#!/usr/bin/env python3
"""
Research Evaluation Script for FlashForge

This script demonstrates how to use the comprehensive evaluation system
for generating research metrics and paper results. It provides examples
of batch evaluation, statistical analysis, and report generation.

Usage:
    python research_evaluation.py --help
    python research_evaluation.py batch --input-dir /path/to/documents
    python research_evaluation.py existing --user-id 1
    python research_evaluation.py demo
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation.batch_evaluator import get_batch_evaluator
from utils.evaluation.dashboard import get_dashboard
from utils.evaluation.evaluator import get_evaluator
from utils.datamodels.db import init_db


def batch_evaluate_documents(input_dir: str, output_dir: str = "evaluation_results", 
                           parallel: bool = True) -> None:
    """
    Batch evaluate all documents in a directory.
    
    Args:
        input_dir: Directory containing documents to evaluate
        output_dir: Output directory for results
        parallel: Whether to use parallel processing
    """
    print(f"🔍 Starting batch evaluation of documents in: {input_dir}")
    
    # Find all supported documents
    supported_extensions = ['.pdf', '.txt', '.docx']
    document_paths = []
    
    for ext in supported_extensions:
        document_paths.extend(Path(input_dir).glob(f"**/*{ext}"))
    
    document_paths = [str(p) for p in document_paths]
    
    if not document_paths:
        print(f"❌ No supported documents found in {input_dir}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"📄 Found {len(document_paths)} documents to evaluate")
    
    # Initialize database
    init_db()
    
    # Run batch evaluation
    batch_evaluator = get_batch_evaluator()
    results = batch_evaluator.evaluate_document_batch(
        document_paths=document_paths,
        output_dir=output_dir,
        enable_parallel=parallel
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BATCH EVALUATION RESULTS")
    print("="*60)
    
    metadata = results['batch_metadata']
    print(f"Total Documents: {metadata['total_documents']}")
    print(f"Successful Evaluations: {metadata['successful_evaluations']}")
    print(f"Failed Evaluations: {metadata['failed_evaluations']}")
    print(f"Processing Time: {metadata['processing_time_seconds']:.2f} seconds")
    print(f"Parallel Processing: {metadata['parallel_processing']}")
    
    if 'statistical_summary' in results:
        stats = results['statistical_summary']
        if 'bert_score_metrics' in stats:
            bert_stats = stats['bert_score_metrics']['f1']
            print(f"\n🎯 BERTScore F1 Statistics:")
            print(f"   Mean: {bert_stats['mean']:.3f} ± {bert_stats['std']:.3f}")
            print(f"   Range: [{bert_stats['min']:.3f}, {bert_stats['max']:.3f}]")
            print(f"   Median: {bert_stats['median']:.3f}")
        
        if 'keyword_coverage_metrics' in stats:
            coverage_stats = stats['keyword_coverage_metrics']['coverage_score']
            print(f"\n🔍 Keyword Coverage Statistics:")
            print(f"   Mean: {coverage_stats['mean']:.3f} ± {coverage_stats['std']:.3f}")
            print(f"   Range: [{coverage_stats['min']:.3f}, {coverage_stats['max']:.3f}]")
            print(f"   Median: {coverage_stats['median']:.3f}")
        
        if 'quality_distribution' in stats:
            quality = stats['quality_distribution']
            print(f"\n📈 Quality Distribution:")
            if 'bert_f1_distribution' in quality:
                bert_dist = quality['bert_f1_distribution']
                print(f"   BERTScore F1 - Excellent (≥0.8): {bert_dist['excellent_ge_0.8']}")
                print(f"                 Good (0.6-0.8): {bert_dist['good_0.6_to_0.8']}")
                print(f"                 Fair (0.4-0.6): {bert_dist['fair_0.4_to_0.6']}")
                print(f"                 Poor (<0.4): {bert_dist['poor_lt_0.4']}")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("   - batch_evaluation_report_*.json (complete results)")
    print("   - batch_evaluation_summary_*.csv (for analysis)")
    print("   - batch_statistics_*.json (statistical summary)")


def evaluate_existing_decks(user_id: Optional[int] = None, 
                          deck_ids: Optional[List[int]] = None) -> None:
    """
    Evaluate existing decks in the database.
    
    Args:
        user_id: Optional user ID to filter by
        deck_ids: Optional specific deck IDs to evaluate
    """
    print("🔍 Evaluating existing decks in database...")
    
    # Initialize database
    init_db()
    
    batch_evaluator = get_batch_evaluator()
    results = batch_evaluator.evaluate_existing_decks(
        user_id=user_id,
        deck_ids=deck_ids
    )
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    if 'message' in results:
        print(f"ℹ️  {results['message']}")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EXISTING DECK EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total Decks Evaluated: {results['total_decks_evaluated']}")
    
    if 'statistical_summary' in results:
        stats = results['statistical_summary']
        if 'bert_f1_statistics' in stats:
            bert_stats = stats['bert_f1_statistics']
            print(f"\n🎯 BERTScore F1 Statistics:")
            print(f"   Mean: {bert_stats['mean']:.3f} ± {bert_stats['std']:.3f}")
            print(f"   Range: [{bert_stats['min']:.3f}, {bert_stats['max']:.3f}]")
        
        if 'coverage_statistics' in stats:
            coverage_stats = stats['coverage_statistics']
            print(f"\n🔍 Keyword Coverage Statistics:")
            print(f"   Mean: {coverage_stats['mean']:.3f} ± {coverage_stats['std']:.3f}")
            print(f"   Range: [{coverage_stats['min']:.3f}, {coverage_stats['max']:.3f}]")
    
    # Show individual deck results
    print(f"\n📚 Individual Deck Results:")
    for result in results['deck_results'][:10]:  # Show first 10
        eval_results = result['evaluation_results']
        bert_f1 = eval_results['bert_score']['f1'] if eval_results.get('bert_score') else 'N/A'
        coverage = eval_results['keyword_coverage']['coverage_score'] if eval_results.get('keyword_coverage') else 'N/A'
        print(f"   {result['deck_name']}: BERTScore={bert_f1:.3f}, Coverage={coverage:.3f}")


def run_demo_evaluation() -> None:
    """Run a demonstration of the evaluation system with sample data."""
    print("🚀 Running FlashForge Evaluation System Demo")
    print("="*50)
    
    # Initialize database
    init_db()
    
    # Sample texts for demonstration
    sample_texts = [
        {
            "title": "Biology - Mitochondria",
            "content": """
            The powerhouse of the cell, the mitochondrion, is responsible for generating most of the cell's supply of adenosine triphosphate (ATP), 
            used as a source of chemical energy. This process is called cellular respiration. Mitochondria are composed of two membranes: an outer membrane 
            and an inner membrane, which is folded into structures called cristae. These cristae increase the surface area for ATP production. 
            The citric acid cycle, also known as the Krebs cycle, occurs in the mitochondrial matrix, while the electron transport chain is located on the inner membrane. 
            Mitochondria are unique in that they have their own small circular DNA, known as mitochondrial DNA or mtDNA.
            """
        },
        {
            "title": "Physics - Quantum Mechanics",
            "content": """
            Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. 
            The wave function, denoted by the Greek letter psi (ψ), is a mathematical description of the quantum state of an isolated quantum system. 
            Heisenberg's uncertainty principle states that the position and momentum of a particle cannot both be precisely determined at the same time. 
            Schrödinger's equation describes how the quantum state of a quantum system changes with time. The principle of superposition allows quantum systems 
            to exist in multiple states simultaneously until measured, at which point the wave function collapses to a single state.
            """
        },
        {
            "title": "Computer Science - Algorithms",
            "content": """
            An algorithm is a finite sequence of well-defined instructions for solving a computational problem. Big O notation describes the limiting behavior 
            of a function when the argument tends towards a particular value or infinity. Time complexity measures the amount of time an algorithm takes to complete 
            as a function of the length of the input. Space complexity measures the amount of memory space an algorithm uses during its execution. 
            Common sorting algorithms include bubble sort, merge sort, and quicksort. Dynamic programming is an optimization technique that solves complex problems 
            by breaking them down into simpler subproblems and storing the results of subproblems to avoid computing the same results again.
            """
        }
    ]
    
    evaluator = get_evaluator()
    
    print("📚 Evaluating sample texts...\n")
    
    all_results = []
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"📖 Sample {i}: {sample['title']}")
        print("-" * 40)
        
        # Generate flashcards with evaluation
        from utils.generation.enhanced_flashcard_gen import generate_flashcards_with_evaluation
        flashcards, evaluation_results = generate_flashcards_with_evaluation(
            sample['content'], 
            enable_evaluation=True
        )
        
        if flashcards and evaluation_results:
            # Print results
            print(f"   Generated Flashcards: {len(flashcards)}")
            
            if evaluation_results.get('bert_score'):
                bert_score = evaluation_results['bert_score']
                print(f"   BERTScore F1: {bert_score['f1']:.3f}")
                print(f"   BERTScore Precision: {bert_score['precision']:.3f}")
                print(f"   BERTScore Recall: {bert_score['recall']:.3f}")
            
            if evaluation_results.get('keyword_coverage'):
                coverage = evaluation_results['keyword_coverage']
                print(f"   Keyword Coverage: {coverage['coverage_score']:.3f}")
                print(f"   Keywords Found: {coverage['covered_keywords']}/{coverage['total_keywords']}")
            
            all_results.append({
                'title': sample['title'],
                'evaluation': evaluation_results
            })
            
            # Show a few sample flashcards
            print(f"   Sample Flashcards:")
            for j, card in enumerate(flashcards[:3]):
                print(f"     Q{j+1}: {card['question']}")
                print(f"     A{j+1}: {card['answer']}")
        
        print()
    
    # Calculate overall statistics
    if all_results:
        bert_f1_scores = []
        coverage_scores = []
        
        for result in all_results:
            eval_data = result['evaluation']
            if eval_data.get('bert_score'):
                bert_f1_scores.append(eval_data['bert_score']['f1'])
            if eval_data.get('keyword_coverage'):
                coverage_scores.append(eval_data['keyword_coverage']['coverage_score'])
        
        print("📊 DEMO EVALUATION SUMMARY")
        print("=" * 40)
        print(f"Samples Evaluated: {len(all_results)}")
        
        if bert_f1_scores:
            import numpy as np
            print(f"Average BERTScore F1: {np.mean(bert_f1_scores):.3f} ± {np.std(bert_f1_scores):.3f}")
        
        if coverage_scores:
            import numpy as np
            print(f"Average Coverage: {np.mean(coverage_scores):.3f} ± {np.std(coverage_scores):.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nℹ️  This demo shows how the evaluation system works.")
    print("   For research purposes, use the batch evaluation features")
    print("   with larger datasets to generate comprehensive metrics.")


def main():
    """Main entry point for research evaluation script."""
    parser = argparse.ArgumentParser(
        description="FlashForge Research Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch evaluate documents in a directory
  python research_evaluation.py batch --input-dir ./test_documents
  
  # Evaluate with custom output directory and sequential processing
  python research_evaluation.py batch --input-dir ./docs --output-dir ./results --no-parallel
  
  # Evaluate existing decks for a specific user
  python research_evaluation.py existing --user-id 1
  
  # Evaluate specific decks
  python research_evaluation.py existing --deck-ids 1,2,3
  
  # Run demonstration with sample data
  python research_evaluation.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch evaluation command
    batch_parser = subparsers.add_parser('batch', help='Batch evaluate documents')
    batch_parser.add_argument('--input-dir', required=True, help='Directory containing documents')
    batch_parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    batch_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    # Existing decks evaluation command
    existing_parser = subparsers.add_parser('existing', help='Evaluate existing decks')
    existing_parser.add_argument('--user-id', type=int, help='Filter by user ID')
    existing_parser.add_argument('--deck-ids', help='Comma-separated deck IDs to evaluate')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration evaluation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'batch':
            if not os.path.exists(args.input_dir):
                print(f"❌ Input directory not found: {args.input_dir}")
                return
            
            batch_evaluate_documents(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                parallel=not args.no_parallel
            )
        
        elif args.command == 'existing':
            user_id = args.user_id
            deck_ids = None
            if args.deck_ids:
                try:
                    deck_ids = [int(x.strip()) for x in args.deck_ids.split(',')]
                except ValueError:
                    print("❌ Invalid deck IDs format. Use comma-separated integers.")
                    return
            
            evaluate_existing_decks(user_id=user_id, deck_ids=deck_ids)
        
        elif args.command == 'demo':
            run_demo_evaluation()
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
