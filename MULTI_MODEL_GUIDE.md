# Multi-Model Support Guide

## Overview

The WFA Profile Analyzer now supports running up to **3 different models** simultaneously on your data. This allows you to:
- Compare model outputs side-by-side
- Build ensemble predictions
- Evaluate which model works best for your use case
- Run experiments with different model sizes

## How It Works

### Option B Implementation: Enable/Disable Flags

Each model in the configuration has an **`enabled`** flag. Only models with `enabled: true` will be processed.

**Key Features:**
- ✅ Support for up to 3 models
- ✅ Easy toggle on/off without deleting configuration
- ✅ Sequential processing to manage GPU memory
- ✅ Automatic cleanup between models
- ✅ Model-specific output columns
- ✅ Validation ensures at least 1 model is enabled

---

## Configuration

### config.yaml Structure

```yaml
models:
  - name: "llama_3b"
    enabled: true  # ← This model WILL run
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
    torch_dtype: "auto"
    device_map: "auto"
  
  - name: "llama_8b"
    enabled: false  # ← This model will be SKIPPED
    model_id: "meta-llama/Llama-3-8B-Instruct"
    torch_dtype: "auto"
    device_map: "auto"
  
  - name: "mistral_7b"
    enabled: false  # ← This model will be SKIPPED
    model_id: "mistralai/Mistral-7B-Instruct-v0.2"
    torch_dtype: "auto"
    device_map: "auto"

# Shared token for all models
hf_token: "your_token_here"

# Shared generation settings
generation:
  batch_size: 10
  max_new_tokens: 2000
  # ... other settings
```

### To Run 2 Models

Just set `enabled: true` for the 2 models you want:

```yaml
models:
  - name: "llama_3b"
    enabled: true  # ← Run this one
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
  
  - name: "llama_8b"
    enabled: true  # ← Run this one
    model_id: "meta-llama/Llama-3-8B-Instruct"
  
  - name: "mistral_7b"
    enabled: false  # ← Skip this one
    model_id: "mistralai/Mistral-7B-Instruct-v0.2"
```

### To Run All 3 Models

Set all to `enabled: true`:

```yaml
models:
  - name: "llama_3b"
    enabled: true
  - name: "llama_8b"
    enabled: true
  - name: "mistral_7b"
    enabled: true
```

---

## Usage

### In Databricks Notebook (Ochestrator.py)

The updated notebook automatically detects and processes all enabled models:

```python
# Load configuration
config = load_config(config_path)

# Create multi-model processor
multi_processor = MultiModelBatchProcessor(config)

# Process your data through all enabled models
result_df = multi_processor.process_dataframe(
    df,
    input_column='about_me',
    parse_outputs=True
)
```

**The processor will:**
1. Load first enabled model
2. Process all data
3. Cleanup and free GPU memory
4. Load second enabled model
5. Process all data
6. Cleanup and free GPU memory
7. Continue for all enabled models...

### Command Line (run_scoring.py)

```bash
# Automatically uses all enabled models
python scripts/run_scoring.py \
    --input data.csv \
    --output results.csv \
    --config config/config.yaml \
    --multi-model

# Single model mode (uses first enabled model only)
python scripts/run_scoring.py \
    --input data.csv \
    --output results.csv \
    --config config/config.yaml
```

---

## Output Structure

### Column Naming Convention

Each model creates its own set of output columns:

#### Raw Model Outputs
- `about_me_processed_llama_3b` - Raw JSON from Llama 3B
- `about_me_processed_llama_8b` - Raw JSON from Llama 8B
- `about_me_processed_mistral_7b` - Raw JSON from Mistral 7B

#### Parsed Columns (if `parse_outputs=True`)
- `ai_llama_3b_flag` - Flag prediction from Llama 3B
- `ai_llama_3b_reasoning` - Reasoning from Llama 3B
- `ai_llama_3b_tags` - Tags from Llama 3B
- `ai_llama_3b_recommendation` - Recommendation from Llama 3B

- `ai_llama_8b_flag` - Flag prediction from Llama 8B
- `ai_llama_8b_reasoning` - Reasoning from Llama 8B
- ... (and so on)

### Example DataFrame

```
| about_me | about_me_processed_llama_3b | ai_llama_3b_flag | about_me_processed_llama_8b | ai_llama_8b_flag |
|----------|---------------------------|------------------|---------------------------|------------------|
| "Hi"     | {"flag": "bad", ...}      | bad              | {"flag": "bad", ...}      | bad              |
| "I am... | {"flag": "good", ...}     | good             | {"flag": "good", ...}     | good             |
```

---

## Comparing Models

### In the Notebook

The updated Ochestrator.py includes a comparison section:

```python
# Get flag columns from each model
flag_columns = [col for col in result_df.columns if col.endswith('_flag')]

# Compare predictions
for i, row in result_df.iterrows():
    flags = [row[col] for col in flag_columns]
    print(f"Profile {i}: {flags}")
```

### Agreement Analysis

```python
# Count how often models agree
agreement_count = 0
for i, row in result_df.iterrows():
    flags = [row[col] for col in flag_columns if pd.notna(row[col])]
    if len(set(flags)) == 1:  # All flags are the same
        agreement_count += 1

agreement_rate = agreement_count / len(result_df)
print(f"Models agree on {agreement_rate:.1%} of profiles")
```

### Ensemble Voting

```python
# Majority vote ensemble
def ensemble_vote(row, flag_columns):
    flags = [row[col] for col in flag_columns if pd.notna(row[col])]
    if not flags:
        return None
    # Return most common prediction
    return max(set(flags), key=flags.count)

result_df['ensemble_flag'] = result_df.apply(
    lambda row: ensemble_vote(row, flag_columns), 
    axis=1
)
```

---

## Performance Considerations

### Memory Management

Models are processed **sequentially** to avoid GPU memory issues:
- ✅ Model 1 loads → processes → unloads → GPU cleared
- ✅ Model 2 loads → processes → unloads → GPU cleared
- ✅ Model 3 loads → processes → unloads → GPU cleared

### Runtime

If you enable 3 models, expect:
- **~3x processing time** compared to single model
- Each model processes the full dataset independently

### Optimization Tips

1. **Start with a subset**: Use `limit` parameter for testing
   ```python
   result_df = multi_processor.process_dataframe(df.head(100))
   ```

2. **Enable only needed models**: Disable models you're not comparing
   ```yaml
   enabled: false  # Skip this model
   ```

3. **Adjust batch size**: Larger batches = faster, but more GPU memory
   ```yaml
   generation:
     batch_size: 16  # Increase if you have memory
   ```

---

## Validation Rules

The system validates your configuration:

### ✅ Valid Configurations
- 1 enabled model
- 2 enabled models
- 3 enabled models

### ❌ Invalid Configurations
- 0 enabled models → Error: "At least one model must be enabled"
- 4+ models → Error: "Maximum of 3 models allowed"
- Duplicate model names → Error: "Model names must be unique"
- Missing model_id → Error: "Model must have a valid model_id"

---

## Troubleshooting

### Issue: "At least one model must be enabled"
**Solution:** Set at least one model to `enabled: true`

### Issue: Out of Memory (OOM)
**Solution:** 
1. Reduce batch size in config
2. Use smaller models (3B instead of 8B)
3. Process fewer models at once

### Issue: Model names not showing in columns
**Solution:** Check that `name` field is set correctly in config

### Issue: Wrong model is running
**Solution:** Check `enabled: true/false` flags in config.yaml

---

## Migration from Single Model

If you had the old single-model config:

### Old Format (DEPRECATED)
```yaml
model:
  model_id: "meta-llama/Llama-3.2-3B-Instruct"
```

### New Format
```yaml
models:
  - name: "llama_3b"
    enabled: true
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
```

**Note:** The old format is no longer supported. Update your config.yaml to the new format.

---

## API Reference

### MultiModelBatchProcessor

```python
from src.processing.batch_processor import MultiModelBatchProcessor

# Initialize
processor = MultiModelBatchProcessor(config)

# Process DataFrame
result_df = processor.process_dataframe(
    df,
    input_column='about_me',  # Optional, defaults to config
    parse_outputs=True,        # Parse JSON to columns
)

# Process CSV
result_df = processor.process_csv(
    input_path=Path('data.csv'),
    output_path=Path('results.csv'),
    limit=100,                 # Optional row limit
    parse_outputs=True,
)

# Get summary
print(processor.get_model_summary())
```

### AppConfig Methods

```python
# Get only enabled models
enabled = config.get_enabled_models()

# Validate configuration
config.validate()  # Raises ValueError if invalid
```

---

## Examples

### Example 1: Compare 2 Models
```yaml
models:
  - name: "small"
    enabled: true
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
  - name: "large"
    enabled: true
    model_id: "meta-llama/Llama-3-8B-Instruct"
  - name: "mistral"
    enabled: false
```

### Example 2: Run Single Model
```yaml
models:
  - name: "primary"
    enabled: true
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
  - name: "backup"
    enabled: false
    model_id: "meta-llama/Llama-3-8B-Instruct"
  - name: "experimental"
    enabled: false
```

### Example 3: Full Model Comparison
```yaml
models:
  - name: "llama_3b"
    enabled: true
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
  - name: "llama_8b"
    enabled: true
    model_id: "meta-llama/Llama-3-8B-Instruct"
  - name: "mistral_7b"
    enabled: true
    model_id: "mistralai/Mistral-7B-Instruct-v0.2"
```

---

## Questions?

- Check the main README.md for general usage
- Review config/config.yaml for full configuration options
- See Ochestrator.py for complete Databricks notebook example

