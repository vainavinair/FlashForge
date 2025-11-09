# FlashForge Evaluation System Guide

## Overview

The FlashForge evaluation system provides comprehensive metrics for assessing flashcard generation quality using two key metrics from academic research:

1. **BERTScore**: Measures semantic similarity between generated flashcards and source text
2. **Keyword Coverage**: Measures how well generated flashcards cover important concepts from source material

This system is designed for both real-time evaluation during flashcard generation and batch evaluation for research purposes.

## Key Features

- ✅ **Integrated Evaluation**: Automatic evaluation during flashcard generation
- ✅ **Real-time Metrics**: Immediate quality feedback to users
- ✅ **Batch Processing**: Evaluate multiple documents for research
- ✅ **Database Storage**: Persistent storage of evaluation results
- ✅ **Statistical Analysis**: Comprehensive statistical reporting
- ✅ **Research Export**: CSV and JSON export for academic papers
- ✅ **Dashboard Analytics**: Visual insights and trends

## Architecture

```
utils/evaluation/
├── __init__.py
├── evaluator.py           # Core evaluation metrics (BERTScore, Keyword Coverage)
├── dashboard.py           # Analytics and reporting dashboard
└── batch_evaluator.py     # Batch processing for research

utils/generation/
└── enhanced_flashcard_gen.py  # Enhanced generation with evaluation

research_evaluation.py     # Research script for paper metrics
```

## Installation

The main FlashForge application works without any additional dependencies. 

**For evaluation features only**, install additional dependencies:

```bash
pip install bert-score nltk
```

The evaluation system will automatically download required NLTK data and BERT models on first use.

**Note:** The main flashcard generation and review functionality works without these dependencies. These are only required for quality evaluation and research features.

## Usage

### 1. Manual Evaluation

```python
from utils.evaluation.evaluator import evaluate_flashcard_generation

# Evaluate specific flashcards
evaluation_results = evaluate_flashcard_generation(
    source_text="Your source text here...",
    generated_flashcards=[
        {"question": "What is ATP?", "answer": "Adenosine triphosphate"},
        {"question": "Where is ATP produced?", "answer": "Mitochondria"}
    ]
)

print(f"BERTScore F1: {evaluation_results['bert_score']['f1']:.3f}")
print(f"Keyword Coverage: {evaluation_results['keyword_coverage']['coverage_score']:.3f}")
```

### 2. Batch Evaluation for Research

#### Using the Research Script

```bash
# Evaluate all documents in a directory
python research_evaluation.py batch --input-dir ./research_documents

# Custom output directory
python research_evaluation.py batch --input-dir ./docs --output-dir ./results

# Sequential processing (for debugging)
python research_evaluation.py batch --input-dir ./docs --no-parallel

# Evaluate existing decks in database
python research_evaluation.py existing --user-id 1

# Run demonstration
python research_evaluation.py demo
```

#### Using the API Directly

```python
from utils.evaluation.batch_evaluator import get_batch_evaluator

# Batch evaluate documents
batch_evaluator = get_batch_evaluator()
results = batch_evaluator.evaluate_document_batch(
    document_paths=["/path/to/doc1.pdf", "/path/to/doc2.txt"],
    output_dir="evaluation_results",
    enable_parallel=True
)

# Evaluate existing decks
existing_results = batch_evaluator.evaluate_existing_decks(user_id=1)
```

### 3. Dashboard Analytics

```python
from utils.evaluation.dashboard import get_dashboard

dashboard = get_dashboard()

# System-wide metrics
system_metrics = dashboard.get_system_wide_metrics()
print(f"Total evaluations: {system_metrics['total_evaluations']}")
print(f"Average BERTScore: {system_metrics['avg_bert_f1']:.3f}")

# User-specific metrics
user_metrics = dashboard.get_user_detailed_metrics(user_id=1)
print(f"User evaluations: {user_metrics['total_evaluations']}")

# Generate comprehensive report
report = dashboard.generate_evaluation_report(
    user_id=1,
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

## Evaluation Metrics

### BERTScore

BERTScore uses BERT embeddings to measure semantic similarity between generated flashcards and source text.

**Interpretation:**
- **0.8-1.0**: Excellent semantic alignment
- **0.6-0.8**: Good semantic similarity  
- **0.4-0.6**: Fair alignment, some improvement needed
- **0.0-0.4**: Poor alignment, significant improvement needed

**Components:**
- `precision`: How well flashcard content matches source
- `recall`: How much source content is captured
- `f1`: Harmonic mean of precision and recall

### Keyword Coverage

Measures percentage of important keywords from source text that appear in generated flashcards.

**Interpretation:**
- **0.7-1.0**: Excellent concept coverage
- **0.5-0.7**: Good concept coverage
- **0.3-0.5**: Fair coverage, missing some concepts
- **0.0-0.3**: Poor coverage, many concepts missed

**Components:**
- `coverage_score`: Percentage of keywords covered
- `total_keywords`: Number of important keywords identified
- `covered_keywords`: Number of keywords found in flashcards
- `covered_keyword_list`: List of covered keywords
- `missed_keyword_list`: List of missed keywords

## Database Schema

The evaluation system adds these tables:

```sql
-- Evaluation results storage
CREATE TABLE evaluation_results (
    evaluation_id INTEGER PRIMARY KEY,
    deck_id INTEGER,
    user_id INTEGER,
    timestamp DATETIME,
    
    -- BERTScore metrics
    bert_precision REAL,
    bert_recall REAL,
    bert_f1 REAL,
    question_f1 REAL,
    answer_f1 REAL,
    
    -- Keyword coverage metrics
    keyword_coverage_score REAL,
    total_keywords INTEGER,
    covered_keywords INTEGER,
    
    -- Metadata
    source_text_length INTEGER,
    num_flashcards INTEGER,
    evaluation_data TEXT  -- Full JSON results
);

-- Source text storage for evaluation
CREATE TABLE source_texts (
    source_id INTEGER PRIMARY KEY,
    deck_id INTEGER,
    user_id INTEGER,
    filename TEXT,
    content TEXT,
    content_hash TEXT,
    upload_timestamp DATETIME
);
```

## Research Output Formats

### Batch Evaluation Output

The batch evaluator generates multiple output files:

1. **`batch_evaluation_report_TIMESTAMP.json`**: Complete results with all data
2. **`batch_evaluation_summary_TIMESTAMP.csv`**: Summary for statistical analysis
3. **`batch_statistics_TIMESTAMP.json`**: Statistical summary only

### CSV Format for Analysis

```csv
document_name,source_text_length,num_flashcards,bert_f1,bert_precision,bert_recall,keyword_coverage,total_keywords,covered_keywords,processing_timestamp
biology_ch1.pdf,2450,12,0.742,0.758,0.727,0.683,41,28,2024-01-15T10:30:00
physics_ch2.pdf,3120,15,0.689,0.701,0.678,0.591,52,31,2024-01-15T10:31:15
```

### Statistical Summary

```json
{
  "sample_size": 25,
  "bert_score_metrics": {
    "f1": {
      "mean": 0.712,
      "std": 0.089,
      "min": 0.543,
      "max": 0.856,
      "median": 0.718,
      "q25": 0.651,
      "q75": 0.774
    }
  },
  "keyword_coverage_metrics": {
    "coverage_score": {
      "mean": 0.634,
      "std": 0.112,
      "min": 0.421,
      "max": 0.823,
      "median": 0.642
    }
  },
  "quality_distribution": {
    "bert_f1_distribution": {
      "excellent_ge_0.8": 4,
      "good_0.6_to_0.8": 16,
      "fair_0.4_to_0.6": 5,
      "poor_lt_0.4": 0
    }
  }
}
```

## Research Applications

### Paper Metrics

Use the evaluation system to generate metrics for academic papers:

1. **Comparative Analysis**: Compare flashcard quality across different domains
2. **Algorithm Evaluation**: Test different generation parameters
3. **User Studies**: Analyze quality trends over time
4. **Benchmark Creation**: Establish quality baselines

### Example Research Workflow

```python
# 1. Batch evaluate a research dataset
results = batch_evaluator.evaluate_document_batch(
    document_paths=research_document_paths,
    output_dir="paper_results"
)

# 2. Generate statistical analysis
stats = results['statistical_summary']
mean_bert_f1 = stats['bert_score_metrics']['f1']['mean']
std_bert_f1 = stats['bert_score_metrics']['f1']['std']

# 3. Export for analysis
import pandas as pd
df = pd.read_csv("paper_results/batch_evaluation_summary_*.csv")

# 4. Statistical tests
from scipy import stats
t_stat, p_value = stats.ttest_1samp(df['bert_f1'], 0.7)  # Test against baseline
```

## Performance Considerations

### Batch Processing

- **Parallel Processing**: Enabled by default for multiple documents
- **Memory Management**: Large documents are processed in chunks
- **Error Handling**: Failed evaluations are logged and reported separately
- **Progress Tracking**: Real-time progress updates during batch processing

### Optimization Tips

1. **Use Parallel Processing**: Enable for multiple documents (default)
2. **Batch Size**: Process 50-100 documents at a time for optimal memory usage
3. **Text Length**: Very long documents (>10,000 words) may take longer
4. **BERT Model**: Uses `bert-base-uncased` by default for good speed/quality balance

## Troubleshooting

### Common Issues

1. **BERT Model Download**: First run downloads ~500MB BERT model
2. **NLTK Data**: Automatically downloads required NLTK datasets
3. **Memory Usage**: Large batches may require more RAM
4. **API Limits**: Gemini API rate limits may affect batch processing

### Error Messages

- **"No keywords found"**: Source text too short or lacks content words
- **"BERTScore evaluation failed"**: BERT model or NLTK data issues
- **"No flashcards generated"**: LLM generation failed

### Solutions

```bash
# Manually download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Check available GPU for BERT
python -c "import torch; print(torch.cuda.is_available())"

# Verify dependencies
pip install bert-score nltk torch
```

## API Reference

### Core Functions

```python
# Main evaluation function
evaluate_flashcard_generation(source_text, flashcards, deck_id=None, user_id=None)

# Enhanced generation with evaluation
generate_flashcards_with_evaluation(text, deck_id=None, user_id=None, filename=None)

# Batch evaluation
batch_evaluator.evaluate_document_batch(document_paths, output_dir, enable_parallel=True)

# Dashboard metrics
dashboard.get_system_wide_metrics()
dashboard.get_user_detailed_metrics(user_id)
dashboard.generate_evaluation_report(user_id=None, start_date=None, end_date=None)
```

### Return Formats

All evaluation functions return standardized dictionaries with metrics and metadata for consistent analysis and reporting.

## Contributing

When adding new evaluation metrics:

1. Add metric calculation to `evaluator.py`
2. Update database schema in `db.py`
3. Add dashboard visualization in `dashboard.py`
4. Include in batch processing in `batch_evaluator.py`
5. Update this documentation

## License

This evaluation system is part of the FlashForge project and follows the same license terms.
