# GPT-OSS Mercury Age Query - Databricks Guide

This notebook demonstrates how to use OpenAI's gpt-oss-20b model with Hugging Face Transformers to query Mercury's age and return structured JSON output.

## Prerequisites

### Databricks Cluster Requirements

1. **GPU**: H100 or RTX 50xx series (Hopper architecture or later)
   - Required for MXFP4 quantization support
   - ~16GB VRAM will be used with MXFP4
   
2. **Runtime**: Databricks Runtime 13.0 ML or later
   - Includes Python 3.10+ and CUDA support

3. **Cluster Configuration**:
   - Worker Type: GPU instance (e.g., `g5.xlarge` or H100 instance)
   - Single node cluster is sufficient for this demo
   - Driver RAM: At least 32GB recommended

### Important Notes on GPU Compatibility

⚠️ **MXFP4 Quantization Requirements** (from documentation):
- MXFP4 is supported on **Hopper or later architectures only**
- This includes: H100, GB200, RTX 50xx series
- If using older GPUs (A100, V100, T4): The model will fall back to bfloat16, requiring ~48GB VRAM

## Setup Instructions

### Step 0: Configure Hugging Face Authentication

Create a `.env.local` file in the project root with your Hugging Face token:

```
HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here
# Optional alias (the notebook will mirror automatically)
HF_TOKEN=your_hugging_face_token_here
```

Notes:
- Do not commit `.env.local` to source control.
- Get a token from `https://huggingface.co/settings/tokens` (read access is sufficient).
- The notebook will load `.env.local`, set env vars, and attempt a `huggingface_hub.login()` automatically.

### Step 1: Upload the Notebook

1. Log into your Databricks workspace
2. Navigate to **Workspace** → **Users** → Your Username
3. Click **Import**
4. Select `gpt-oss-mercury-age.ipynb`
5. Click **Import**

### Step 2: Create a GPU Cluster

1. Go to **Compute** → **Create Cluster**
2. Configure:
   - **Cluster name**: `gpt-oss-demo`
   - **Cluster mode**: Single Node
   - **Databricks runtime**: 13.0 ML or later
   - **Worker type**: Select GPU instance (H100 recommended)
   - **Enable autoscaling**: Off
3. Click **Create Cluster**
4. Wait for cluster to start (~5-10 minutes)

### Step 3: Attach Notebook to Cluster

1. Open the imported notebook
2. Click the cluster dropdown at the top
3. Select your GPU cluster
4. Click **Attach**

### Step 4: Run the Notebook

1. **First Run**: The initial execution will download ~16GB of model weights
   - This can take 10-20 minutes depending on network speed
   - The model is cached for subsequent runs
2. **Subsequent Runs**: Much faster (~2-3 minutes total)
3. Run cells sequentially from top to bottom

## Notebook Structure

The notebook is organized into clear steps:

### 1. Install Dependencies (Cell 2)
Installs required packages with specific versions:
- `transformers` - Hugging Face model loading
- `accelerate` - Device management
- `torch` - PyTorch backend
- `triton==3.4` - **Critical** for MXFP4 kernel compatibility on H100

**Note**: Some documentation mentions a "kernels" package, but this doesn't appear to be a real PyPI package. MXFP4 support is built into triton 3.4 and transformers.

### 2. Verify GPU (Cell 4)
Checks that H100 GPU is available and displays memory information.

### 3. Load Model (Cell 6)
Loads `openai/gpt-oss-20b` with:
- `torch_dtype="auto"` - Automatically uses MXFP4 on H100
- `device_map="auto"` - Places model on GPU automatically

### 4. Construct Prompt (Cell 8)
Builds prompt using **Harmony response format**:
- **System message**: Defines model identity, reasoning level, channels
- **Developer message**: Includes instructions and JSON schema
- **User message**: The question about Mercury's age

### 5. Generate Response (Cell 10)
Calls `model.generate()` with:
- `max_new_tokens=500` - Limits response length
- `eos_token_id=[200002, 200012]` - Stop tokens for `<|return|>` and `<|call|>`

### 6. Parse Harmony Format (Cells 12-14)
Extracts content from the **final channel** while removing:
- Analysis channel (chain-of-thought reasoning)
- Commentary channel (tool calls)
- Special tokens (`<|channel|>`, `<|message|>`, `<|return|>`)

### 7. Display JSON Output (Cell 16)
Parses and displays the structured JSON with Mercury's age.

## Expected Output

The final output should look like:

```json
{
  "years": 4503000000,
  "months": 54036000000,
  "weeks": 234906450000,
  "centuries": 45030000
}
```

## Understanding the Harmony Response Format

The gpt-oss models use a multi-channel output format:

### Channels:
- **analysis**: Chain-of-thought reasoning (internal, not safe for users)
- **commentary**: Tool calls and preambles
- **final**: User-facing response (what we extract)

### Special Tokens:
- `<|start|>` (200006): Beginning of message
- `<|end|>` (200007): End of message
- `<|message|>` (200008): Transition to content
- `<|channel|>` (200005): Channel specification
- `<|return|>` (200002): Stop token indicating completion
- `<|call|>` (200012): Stop token for tool calls

### Example Raw Output:
```
<|channel|>analysis<|message|>User asks about Mercury's age. Need to provide in multiple units.<|end|>
<|start|>assistant<|channel|>final<|message|>{"years": 4503000000, ...}<|return|>
```

Our parsing code extracts only the `final` channel content.

## Troubleshooting

### Issue: "No GPU detected"
**Solution**: 
- Verify cluster has GPU instance type
- Restart cluster if GPU isn't recognized
- Check Databricks runtime includes ML

### Issue: "MXFP4 not supported"
**Solution**:
- Verify GPU is H100 or RTX 50xx
- If using older GPU (A100, V100), model will use bfloat16 (~48GB VRAM needed)
- Consider switching to smaller batch size or different GPU

### Issue: Model download is slow
**Solution**:
- First run always downloads ~16GB
- Consider using shared storage for model cache
- Set environment variable: `TRANSFORMERS_CACHE=/dbfs/ml/cache`

### Issue: "Could not find final channel content"
**Solution**:
- Model may not have followed expected format
- Check FULL OUTPUT section to see raw response
- Adjust prompt construction if needed
- May need to increase `max_new_tokens`

### Issue: JSON parsing fails
**Solution**:
- Model output might not be valid JSON
- Check EXTRACTED FINAL CHANNEL CONTENT section
- May need to adjust developer message instructions
- Try different temperature value (lower = more deterministic)

## Performance Optimization

### Memory Management
- Monitor GPU memory: `torch.cuda.memory_allocated()`
- MXFP4 should keep usage at ~16GB
- Clear cache between runs if needed: `torch.cuda.empty_cache()`

### Speed Optimization
- Use `temperature=0.0` for deterministic, faster inference
- Reduce `max_new_tokens` to 200-300 if responses are too long
- Consider using pipeline API for simpler use cases (though less control over parsing)

### Cost Optimization
- Stop cluster when not in use
- Use spot instances for non-production workloads
- Cache model weights in shared DBFS location

## Advanced Usage

### Customizing the Question
Modify Cell 8 to ask different questions:

```python
user_message = "What is the diameter of Mars in kilometers and miles?"

# Update JSON schema accordingly
json_schema = {
    "type": "object",
    "properties": {
        "kilometers": {"type": "number"},
        "miles": {"type": "number"}
    }
}
```

### Adjusting Reasoning Level
In the system message (Cell 8), change reasoning level:
- `Reasoning: low` - Minimal chain-of-thought
- `Reasoning: medium` - Balanced (default)
- `Reasoning: high` - Extensive reasoning

### Using Different Models
Replace `openai/gpt-oss-20b` with:
- `openai/gpt-oss-120b` - Larger model (requires ≥60GB VRAM)

## Key Takeaways

1. ✅ **MXFP4** keeps memory at ~16GB on H100
2. ✅ **Harmony format** separates reasoning from final output
3. ✅ **Structured output** requires JSON schema in developer message
4. ✅ **Stop tokens** control generation endpoints
5. ✅ **Channel parsing** extracts only user-facing content

## References

- [OpenAI GPT-OSS Documentation](openai-how-to-run-the-gpt-oss-models.md)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GPT-OSS-20B Model Card](https://huggingface.co/openai/gpt-oss-20b)
- [Harmony Response Format Specification](https://cookbook.openai.com/articles/gpt-oss/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full documentation in `openai-how-to-run-the-gpt-oss-models.md`
3. Verify GPU compatibility and cluster configuration

