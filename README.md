# WFA Profile Analyzer

A tool for analyzing and evaluating profile quality predictions using Large Language Models (LLMs).

## Overview

This tool leverages Large Language Models (LLMs) to automate the quality assurance of job seeker profiles. It analyzes the "about me" section to identify and flag content that fails to meet an evaluation criteria.

## Required Resources to Run Workflow

- **Compute Infrastructure**: Databricks Workspace with GPU Cluster (H100)
- **Azure Storage Explorer**: Azure Storage Explorer with access to the project container/folder
- **Model Access**: Hugging Face account with valid User Access Token (for gated models)

## Project Structure

### Core Files
- `config.yaml` - Configuration settings for models, batch sizes, data paths, and generation parameters
- `requirements.txt` - Python dependencies including transformers, torch, pandas, and evaluation libraries
- `run_analysis.ipynb` - Interactive notebook for executing the complete analysis pipeline

### Source Code (`src/`)
- `analyzer.py` - Core analysis engine that loads models, generates predictions, and orchestrates the evaluation workflow
- `evaluate.py` - Evaluation metrics calculation including precision, recall, F1-scores, and classification reports
- `prompt.py` - System prompt template and prompt generation functions for different model types (instruct vs harmony)
- `utils.py` - Utility functions for loading data, parsing JSON responses, and saving results to CSV
- `__init__.py` - Python package initialization file


## How to Run

### Quick Start

1. **Set up authentication**:

   - Create a Hugging Face account at [huggingface.co](https://huggingface.co)
   - Apply for access to gated models (approval usually takes 24 hours)
   - Generate a Hugging Face token following [this guide](https://huggingface.co/docs/hub/en/security-tokens)
   - Create a `.env` file and add your token.
   

2. **Prepare your data**:
  
  - **Step 1 – Verify structure**
    - Keep the dataset as an Excel file (`.xlsx`); do **not** export to CSV because commas and special characters in profile text will corrupt the format.
    - Confirm the sheet contains these exact column names:
      - `about_me` (profile text)
      - `Human_flag` (quality labels with `good` / `bad`)
      - `personal_information`, `sensitive_information`, `inappropriate_information`, `poor_grammar` (tag columns with binary 0/1 values)
  - **Step 2 – Upload & configure**
    - Upload the `.xlsx` file directly to `{folder name}` in Azure Storage Explorer.
    - Update the `paths.input_data` entry in `config.yaml` with the new blob path.

3. **Configure models**:
  
   - Open `config.yaml` and:
     - Toggle each model’s `enabled` flag based on which ones you want to run.
     - Adjust quantization, device map, and torch dtype to match your hardware profile.
     - Review generation parameters (`batch_size`, `max_new_tokens`, etc.) to ensure they fit your dataset size and GPU memory.

4. **Run the analysis**:
  
   - Run `run_analysis.ipynb` 
   - Once execution finishes, outputs are automatically written to Azure Storage and can be downloaded via Azure Storage Explorer.
   - Generated artifacts:
    - `model_metrics_{timestamp}.csv` – performance comparison table
    - `profile_analysis_results_{timestamp}.csv` – full prediction rows with reasoning and tags


### Complete Workflow
The system follows this pipeline:
1. **Load models** → Configured language models from Hugging Face
2. **Generate prompts** → Profile text + system prompt for quality assessment  
3. **Batch inference** → Process profiles through models to get quality predictions
4. **Parse outputs** → Extract structured JSON responses (quality, reasoning, tags)
5. **Evaluate performance** → Compare model predictions against human labels
6. **Generate metrics** → Precision, recall, F1-score for model comparison
7. **Save results** → Export metrics and full analysis to timestamped CSV files in Azure Storage

## Understanding the Evaluation Results

### Binary Classification Setup

The system uses binary classification where:

- **Positive class (1)**: Bad quality profiles
- **Negative class (0)**: Good quality profiles

### Model Comparison Table

When you run `run_analysis.ipynb`, you'll get a detailed performance report for each model with these columns:

| Column | Description |
|--------|-------------|
| `Tag` | The category being evaluated (`overall` quality or specific tags like `personal_information`) |
| `Precision` | Precision for the positive class (Bad Quality / Tag Present) |
| `Recall` | Recall for the positive class (Bad Quality / Tag Present) |
| `F1-Score` | Harmonic mean of Precision and Recall |
| `Support` | Number of samples in the evaluation set |
| `Accuracy` | Overall classification accuracy |
| `TP`/`FP`/`FN` | Raw counts: True Positives, False Positives, False Negatives |

### Key Metrics Explained

#### Precision
**What it means**: When the model flags a profile as "bad" (or having a specific tag), how often is it correct?
- High Precision = Few **False Positives** (fewer good profiles incorrectly flagged).

#### Recall
**What it means**: Of all actually "bad" profiles (or tags present), what percentage did the model catch?
- High Recall = Few **False Negatives** (fewer bad profiles slipped through).

#### F1-Score
**What it means**: A balanced metric combining both Precision and Recall (harmonic mean).

#### Accuracy
**What it means**: The overall percentage of correct predictions (both "good" and "bad").

#### Support
**What it means**: The sample size used for the metric.
- For **Specific Tags**: The number of actual profiles containing that tag (positive cases).
- For **Overall**: The total number of profiles processed (total dataset size).

### Example Output

```
PER-TAG PERFORMANCE METRICS FOR MODEL: llama_1b
================================================================================
Tag                        Precision  Recall  F1-Score  Support   TP   FP   FN  Accuracy
personal_information           0.92    0.85      0.88      150   25    2    4      0.95
sensitive_information          0.88    0.90      0.89      150   15    2    2      0.97
inappropriate_information      1.00    0.75      0.86      150    3    0    1      0.99
poor_grammar                   0.75    0.80      0.77      150   40   13   10      0.85
overall                        0.85    0.82      0.83      150   55   10   12      0.85
────────────────────────────────────────────────────────────────────────────────
```


