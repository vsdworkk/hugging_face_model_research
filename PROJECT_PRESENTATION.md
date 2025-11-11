# WFA Profile Analyzer: Project Presentation

---

## Slide 1: Title Slide

# WFA Profile Analyzer
## Automated Profile Quality Assessment Using LLMs

**Project Overview**
- Multi-model evaluation system for job seeker profile quality
- Binary classification: "good" vs "bad" profiles
- Designed to reduce manual review workload

---

## Slide 2: Problem Statement

### The Challenge

**Manual Profile Review is Time-Consuming**
- Human reviewers must evaluate thousands of profiles
- Need to identify problematic profiles efficiently
- Quality standards: flag profiles with personal info, inappropriate content, or poor grammar

**Business Impact**
- High false positive rate = wasted reviewer time
- High false negative rate = bad profiles slip through
- Need automated solution that balances precision and recall

---

## Slide 3: Solution Overview

### Our Approach

**Multi-Model Evaluation System**
- Test multiple LLMs simultaneously on the same dataset
- Compare performance metrics (precision, recall, F1-score)
- Choose the best model for production deployment

**Key Capabilities**
- Batch processing for efficiency
- Structured JSON output parsing
- Comprehensive evaluation metrics
- Configurable model selection

---

## Slide 4: Architecture Overview

### System Architecture

```
┌─────────────────┐
│  config.yaml    │  ← Model configurations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   analyzer.py   │  ← Core orchestration
│  - Load models  │
│  - Batch proc.  │
│  - Parse JSON   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│prompt.py│ │evaluate.py│
│         │ │          │
│Prompts  │ │Metrics   │
└─────────┘ └──────────┘
```

**Modular Design**
- Separation of concerns
- Easy to add new models
- Reusable components

---

## Slide 5: Core Components

### Component Breakdown

**1. `analyzer.py` - Orchestration Engine**
- Model pipeline management
- Batch processing logic
- Memory management
- JSON parsing with fallback strategies

**2. `prompt.py` - Prompt Engineering**
- System prompt with clear criteria
- Model-specific formatting (instruct vs non-instruct)
- Chat template application

**3. `evaluate.py` - Metrics & Comparison**
- Binary classification metrics
- Model comparison tables
- Precision/Recall/F1 calculations

**4. `config.yaml` - Configuration**
- Model enable/disable flags
- Quantization settings
- Batch size and token limits

---

## Slide 6: Data Flow

### Processing Pipeline

```
1. Load Data
   └─> CSV with profile text + human labels

2. Load Models (from config.yaml)
   └─> Hugging Face models (Llama, Gemma, etc.)

3. Generate Prompts
   └─> System prompt + profile text
   └─> Apply chat templates for instruct models

4. Batch Inference
   └─> Process in configurable batch sizes
   └─> Generate quality assessments

5. Parse Outputs
   └─> Extract JSON from model responses
   └─> Handle malformed outputs gracefully

6. Evaluate Performance
   └─> Compare predictions vs human labels
   └─> Calculate precision, recall, F1

7. Export Results
   └─> Timestamped CSV files
   └─> Model comparison metrics
```

---

## Slide 7: Challenge 1: Memory Management

### Problem: Large Models, Limited Resources

**The Issue**
- LLMs can be several GB in size
- Multiple models = memory multiplication
- GPU memory constraints
- Need to process multiple models sequentially

**Our Solution**

**1. Quantization Support**
```python
# Configurable quantization in config.yaml
quantization: "4bit"  # Reduces memory by ~75%
quantization: "8bit"  # Reduces memory by ~50%
quantization: "none"  # Full precision
```

**2. Explicit Memory Cleanup**
```python
# After each model processing
del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**3. Sequential Processing**
- Load one model at a time
- Process all profiles
- Clean up before loading next model
- Prevents memory accumulation

**Result**: Can run multiple large models on limited hardware

---

## Slide 8: Challenge 2: JSON Parsing Reliability

### Problem: Inconsistent Model Outputs

**The Issue**
- Models don't always output clean JSON
- Sometimes include markdown formatting
- Sometimes include explanatory text
- Sometimes malformed JSON

**Our Solution: Multi-Strategy Parsing**

```python
def parse_json_output(text: str):
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # Strategy 2: Regex extraction
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return None  # Graceful failure
```

**Why This Works**
- Handles multiple output formats
- Doesn't fail on edge cases
- Returns None for truly unparseable outputs
- Evaluation handles missing predictions gracefully

**Result**: Robust parsing that handles model variability

---

## Slide 9: Challenge 3: Optimizing Unreliable Model Outputs

### Problem: Models Don't Always Follow Instructions

**The Issue**
- Smaller models (1B-3B parameters) are less reliable than larger models
- Inconsistent output quality across different profiles
- Models sometimes ignore JSON format requirements
- Models may produce low-quality or irrelevant reasoning
- Non-deterministic outputs make evaluation difficult

**Our Solution: Multi-Layered Optimization Strategy**

**1. Deterministic Sampling**
```python
batch_outputs = pipe(
    batch,
    max_new_tokens=max_new_tokens,
    do_sample=False,  # Deterministic, reproducible outputs
    return_full_text=False
)
```
- Ensures reproducible results for evaluation
- Reduces randomness that can hurt smaller models
- Makes debugging and comparison easier

**2. Structured Prompt Engineering**
- Clear task definition with explicit boundaries
- Detailed scoring criteria with examples
- Multiple reminders about output format
- Example output included in prompt (few-shot learning)
- XML-style tags for structure (`<task>`, `<output_format>`)

**3. Explicit Output Format Specification**
```python
SYSTEM_PROMPT = """
<output_instructions>
You MUST respond with ONLY a valid JSON object. 
Do not include any text before or after the JSON.
Do not include markdown formatting, code blocks, or any other formatting.
Output ONLY the raw JSON object starting with { and ending with }
</output_instructions>

<example_output>
{
  "quality": "bad",
  "reasoning": "...",
  "tags": [...],
  "improvement_points": [...]
}
</example_output>
"""
```
- Multiple explicit instructions about format
- Example output shows exactly what we want
- Reduces format violations

**4. Graceful Degradation**
- Multi-strategy JSON parsing (covered in Challenge 2)
- Handles missing or invalid predictions
- System continues processing even with failures
- Evaluation filters out invalid predictions

**5. Token Limit Management**
```python
max_new_tokens: 2000  # Configurable in config.yaml
```
- Prevents models from generating excessively long outputs
- Reduces chance of format drift
- Balances completeness vs consistency

**Why This Works**
- Deterministic sampling reduces variability
- Structured prompts guide smaller models better
- Example outputs provide few-shot learning
- Multiple format reminders increase compliance
- Graceful handling prevents system failures

**Result**: Reliable outputs even from smaller, less capable models

---

## Slide 10: Challenge 4: Model Format Compatibility

### Problem: Different Models, Different Formats

**The Issue**
- Instruct models (Llama-3.2-Instruct) use chat templates
- Non-instruct models use simple concatenation
- Need to support both formats

**Our Solution: Configuration-Driven Formatting**

```python
def generate_prompt(text, model_config, tokenizer):
    if model_config.get('is_instruct', False):
        # Use chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"About me:\n{text}"}
        ]
        return tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    else:
        # Simple concatenation
        return f"{SYSTEM_PROMPT}\n\nAbout me:\n{text}\n\nAnalysis:"
```

**Why This Matters**
- Single codebase supports multiple model types
- Proper formatting improves model performance
- Easy to add new models via config

**Result**: Flexible system that works with diverse model architectures

---

## Slide 11: Challenge 5: Batch Processing Efficiency

### Problem: Processing Thousands of Profiles

**The Issue**
- Sequential processing is slow
- Need to balance speed vs memory
- Tokenizer padding requirements

**Our Solution: Optimized Batching**

```python
def process_in_batches(pipe, prompts, batch_size, max_new_tokens):
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_outputs = pipe(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic
            return_full_text=False  # Only new tokens
        )
        # Extract generated text
        outputs.extend(extract_outputs(batch_outputs))
    return outputs
```

**Key Optimizations**
- Configurable batch size (default: 10)
- Left padding for batch consistency
- Deterministic sampling for reproducibility
- Return only new tokens (not full prompt)

**Result**: Efficient processing that scales to large datasets

---

## Slide 12: Challenge 6: Evaluation Metrics Design

### Problem: Choosing the Right Metrics

**The Issue**
- Need to compare multiple models
- Different business priorities (precision vs recall)
- Binary classification with imbalanced classes

**Our Solution: Comprehensive Metrics**

```python
def evaluate_all_models(df, model_names, true_col):
    # For each model:
    # - Accuracy (overall performance)
    # - Precision_bad (minimize false positives)
    # - Recall_bad (minimize false negatives)
    # - F1_bad (balanced measure)
    # - Support (sample size)
```

**Why These Metrics**
- **Precision_bad**: Critical for minimizing wasted reviewer time
- **Recall_bad**: Critical for catching all bad profiles
- **F1_bad**: Balanced view when both matter
- **Support**: Ensures statistical reliability

**Result**: Clear comparison framework for model selection

---

## Slide 13: Architecture Decisions: Why We Did It This Way

### Decision 1: YAML Configuration

**Why YAML instead of hardcoded values?**
- Easy to enable/disable models without code changes
- Non-technical users can modify settings
- Version control friendly
- Supports multiple environments (dev/prod)

### Decision 2: Modular Component Design

**Why separate analyzer/prompt/evaluate?**
- Single Responsibility Principle
- Easy to test components independently
- Easy to swap implementations
- Clear separation of concerns

### Decision 3: In-Place DataFrame Modification

**Why modify DataFrame in-place?**
- Memory efficient (no copying)
- All results in one structure
- Easy to export to CSV
- Maintains original data alongside predictions

---

## Slide 14: Architecture Decisions (Continued)

### Decision 4: Hugging Face Pipeline

**Why use transformers Pipeline?**
- Standardized interface across models
- Built-in quantization support
- Handles device mapping automatically
- Consistent API regardless of model architecture

### Decision 5: Timestamped Output Files

**Why timestamp filenames?**
- Prevents overwriting previous runs
- Easy to track experiment history
- Supports A/B testing different configurations
- Audit trail for production decisions

### Decision 6: Graceful Error Handling

**Why return None for parse failures?**
- System continues processing other profiles
- Evaluation handles missing predictions
- No crashes on edge cases
- Better user experience

---

## Slide 15: Key Features

### What Makes This System Robust

**1. Flexibility**
- Easy to add new models via config
- Supports different quantization levels
- Configurable batch sizes and token limits

**2. Reliability**
- Multi-strategy JSON parsing
- Graceful error handling
- Memory cleanup between models

**3. Usability**
- Simple YAML configuration
- Clear output formats
- Comprehensive metrics

**4. Scalability**
- Batch processing for efficiency
- Sequential model loading for memory
- Handles large datasets

---

## Slide 16: Example Output

### Model Comparison Results

```
MODEL COMPARISON
======================================================================
    model  accuracy  precision_bad  recall_bad  f1_bad  support
  llama_1b     0.850          0.90       0.80   0.85      100
  gemma_1b     0.820          0.75       0.95   0.84      100
```

**Interpretation**
- **llama_1b**: Higher precision (fewer false positives)
- **gemma_1b**: Higher recall (catches more bad profiles)
- Choose based on business priority

### Detailed Results CSV
- Original profile text
- Human labels
- Model predictions with reasoning
- Tags and improvement points

---

## Slide 17: Technical Stack

### Technologies Used

**Core Libraries**
- `transformers` - Hugging Face model loading
- `torch` - PyTorch deep learning framework
- `pandas` - Data manipulation
- `scikit-learn` - Evaluation metrics
- `bitsandbytes` - Model quantization
- `pyyaml` - Configuration parsing

**Infrastructure**
- Azure Storage - Data storage
- Hugging Face Hub - Model repository
- Jupyter Notebooks - Interactive execution

**Why This Stack?**
- Industry standard tools
- Active community support
- Well-documented APIs
- Production-ready components

---

## Slide 18: Lessons Learned

### Key Takeaways

**1. Memory Management is Critical**
- Always clean up between models
- Quantization is essential for resource constraints
- Monitor memory footprint during development

**2. Robust Parsing is Essential**
- Models are unpredictable
- Multiple fallback strategies needed
- Graceful degradation better than crashes

**3. Configuration > Hardcoding**
- YAML config enables experimentation
- Easy to test different model combinations
- Non-technical users can modify settings

**4. Metrics Matter**
- Choose metrics aligned with business goals
- Precision vs Recall tradeoff is real
- Support sample size ensures reliability

**5. Optimizing Smaller Models Requires Strategy**
- Deterministic sampling improves consistency
- Structured prompts with examples are essential
- Multiple format reminders increase compliance
- Graceful degradation prevents system failures

---

## Slide 19: Future Enhancements

### Potential Improvements

**1. Model Ensembles**
- Combine predictions from multiple models
- Voting or weighted averaging
- Potentially better performance

**2. Caching**
- Cache model outputs for repeated profiles
- Reduce redundant computation
- Faster iteration cycles

**3. Real-time Processing**
- API endpoint for single profile analysis
- Web interface for reviewers
- Integration with existing systems

**4. Advanced Prompting**
- Few-shot examples
- Chain-of-thought reasoning
- Prompt optimization via experimentation

---

## Slide 20: Business Impact

### Value Delivered

**Efficiency Gains**
- Automated flagging reduces manual review time
- High precision minimizes false positives
- Scalable to thousands of profiles

**Quality Assurance**
- Consistent evaluation criteria
- Comprehensive reasoning provided
- Improvement points for profile enhancement

**Decision Support**
- Data-driven model selection
- Clear performance metrics
- Reproducible evaluation process

**Cost Savings**
- Reduced reviewer workload
- Faster processing times
- Lower infrastructure costs with quantization

---

## Slide 21: Conclusion

### Summary

**What We Built**
- Multi-model evaluation system for profile quality assessment
- Robust, scalable, and configurable architecture
- Comprehensive evaluation framework

**How We Overcame Challenges**
- Memory management through quantization and cleanup
- Robust JSON parsing with multiple strategies
- Optimized unreliable model outputs with deterministic sampling and structured prompts
- Flexible prompt formatting for different model types
- Efficient batch processing
- Well-designed evaluation metrics

**Key Success Factors**
- Modular, maintainable code
- Configuration-driven design
- Graceful error handling
- Clear separation of concerns

**Next Steps**
- Deploy best-performing model to production
- Monitor performance on real-world data
- Iterate based on feedback

---

## Slide 22: Q&A

### Questions?

**Contact & Resources**
- Project Repository: [GitHub URL]
- Documentation: README.md
- Configuration: config.yaml
- Example Notebook: run_analysis.ipynb

**Thank You!**

