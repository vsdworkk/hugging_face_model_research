# WFA Profile Analyzer

A tool for analyzing and evaluating profile quality predictions using Large Language Models (LLMs).

## Overview

This analyzer evaluates how well different models can classify profiles as "good" or "bad" quality. The primary use case is to automatically flag potentially problematic profiles for human review, reducing manual workload while maintaining quality standards.

## Key Files

- `analyzer.py` - Main analysis logic
- `evaluate.py` - Model evaluation utilities
- `prompt.py` - Prompt templates for model interactions
- `config.yaml` - Configuration settings
- `run_analysis.ipynb` - Notebook for running analyses

## How to Run

### Quick Start

1. **Set up authentication** (optional, for gated models):

   - Create an .env according to the example template and add 'your_huggingface_token_here' with your actual token.
   - If running gated models apply for access with hugging face - usually takes 24 hours. 

2. **Prepare your data**:
   - Upload your CSV file to Azure Storage using Azure Storage Explorer
   - Ensure CSV has profile text column (default: `about_me`)
   - Ensure CSV has human quality labels column (default: `Human_flag`) with 'good'/'bad' values
   - Update `config.yaml` paths section with your file path

3. **Configure models**:
   - Edit `config.yaml` to enable/disable models
   - Adjust quantization settings if needed for memory constraints

4. **Run the analysis**:
     
     Run notebook run_analysis.ipynb

5. **Results will be saved to**:
   - **Model comparison metrics**: `model_metrics_{timestamp}.csv` - Performance comparison table
   - **Full analysis results**: `profile_analysis_results_{timestamp}.csv` - All predictions with reasoning
   - Timestamps are automatically added (e.g., `2024-01-15_143022`)
   - Results are automatically saved to Azure Storage and can be downloaded using Azure Storage Explorer
   - Output paths can be customized in `config.yaml` under the `paths` section


### Complete Workflow
The system follows this pipeline:
1. **Load models** → Configured language models from Hugging Face
2. **Generate prompts** → Profile text + system prompt for quality assessment  
3. **Batch inference** → Process profiles through models to get quality predictions
4. **Parse outputs** → Extract structured JSON responses (quality, reasoning, tags)
5. **Evaluate performance** → Compare model predictions against human labels
6. **Generate metrics** → Precision, recall, F1-score for model comparison
7. **Save results** → Export metrics and full analysis to timestamped CSV files

## Understanding the Evaluation Results

### Binary Classification Setup

The system uses binary classification where:
- **Positive class (1)**: Bad quality profiles
- **Negative class (0)**: Good quality profiles

### Model Comparison Table

When you run run_analysis.ipynb , you'll get a comparison table with these columns:

| Column | Description | Why It Matters |
|--------|-------------|----------------|
| `model` | Model name | Identifies which model produced these results |
| `accuracy` | Overall accuracy | General performance indicator |
| `precision_bad` | Precision for detecting bad profiles | 
| `recall_bad` | Recall for detecting bad profiles |
| `f1_bad` | F1-score for detecting bad profiles | Balanced measure of precision and recall |
| `support` | Number of valid predictions | Sample size for reliability assessment |

### Key Metrics Explained

#### Precision for Bad Profiles (`precision_bad`)
**What it means**: Of all profiles the model flagged as "bad", what percentage were actually bad?

**Why it's important**: High precision means fewer **False Positives** - fewer good profiles incorrectly sent to human reviewers.

- `precision_bad = 0.90` → 90% of flagged profiles are actually bad, 10% are false alarms
- `precision_bad = 0.70` → 70% of flagged profiles are actually bad, 30% are false alarms

**Impact**: Low precision = wasted human review time on good profiles.

#### Recall for Bad Profiles (`recall_bad`)
**What it means**: Of all actually bad profiles, what percentage did the model successfully identify?

**Why it's important**: High recall means fewer **False Negatives** - fewer bad profiles slip through undetected.

- `recall_bad = 0.85` → Model catches 85% of bad profiles, misses 15%
- `recall_bad = 0.60` → Model catches 60% of bad profiles, misses 40%

**Impact**: Low recall = bad profiles get through without human review.

### Choosing the Right Model

**For minimising human workload**: Prioritise high `precision_bad`
- Reduces false positives
- Less time wasted reviewing good profiles

**For catching all bad profiles**: Prioritise high `recall_bad`
- Reduces false negatives  
- Ensures fewer bad profiles slip through

**For balanced performance**: Look at `f1_bad`
- Harmonic mean of precision and recall
- Good overall indicator when you need both

### Example Output

```
MODEL COMPARISON
======================================================================
    model  accuracy  precision_bad  recall_bad  f1_bad  support
     gpt-4     0.850          0.90       0.80   0.85      100
  claude-3     0.820          0.75       0.95   0.84      100
    gemini     0.800          0.95       0.70   0.81      100
```


