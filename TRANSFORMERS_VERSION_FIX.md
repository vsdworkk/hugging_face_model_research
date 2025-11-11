# Transformers 4.55.1 Compatibility Fix

## Issue Summary

Models failed to load when upgrading from `transformers` version 4.50.1 to 4.55.1, despite working correctly with the same models using the example code from Hugging Face documentation.

## Root Cause

The issue was in how the `torch_dtype` parameter was being passed to the `pipeline()` function in `src/analyzer.py`.

### Old Code (Incompatible with transformers >= 4.55.1)

```python
return pipeline(
    "text-generation",
    model=model_config['model_id'],
    device_map=model_config.get('device_map', 'auto'),
    torch_dtype=torch_dtype,  # ❌ Direct parameter - deprecated
    model_kwargs=model_kwargs,
    **hub_kwargs
)
```

### New Code (Compatible with transformers >= 4.55.1)

```python
# Add torch_dtype to model_kwargs instead
if torch_dtype != 'auto':
    if torch_dtype == 'float16':
        model_kwargs['torch_dtype'] = torch.float16
    elif torch_dtype == 'bfloat16':
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif torch_dtype == 'float32':
        model_kwargs['torch_dtype'] = torch.float32
    else:
        model_kwargs['torch_dtype'] = torch_dtype

return pipeline(
    "text-generation",
    model=model_config['model_id'],
    device_map=model_config.get('device_map', 'auto'),
    model_kwargs=model_kwargs,  # ✓ torch_dtype inside model_kwargs
    **hub_kwargs
)
```

## What Changed in Transformers 4.55.1

Two major changes in transformers 4.55.1 affect model loading:

### 1. torch_dtype Parameter Location
The `pipeline()` function's API was updated to require model-specific parameters like `torch_dtype` to be passed inside the `model_kwargs` dictionary rather than as direct parameters to the `pipeline()` function.

### 2. Default Safetensors Loading
Transformers 4.55.1 defaults to loading models using the `safetensors` format for enhanced security and performance. However, this can cause loading to hang for some models that don't have `safetensors` files or have compatibility issues.

This aligns with the pattern shown in the Hugging Face documentation:

```python
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,  # Inside model_kwargs
        "use_safetensors": False        # Force PyTorch .bin files
    },
    device_map="auto",
)
```

## Files Modified

1. **`src/analyzer.py`**
   - Updated `load_model_pipeline()` function (lines 119-162)
   - Moved `torch_dtype` parameter from direct pipeline parameter to inside `model_kwargs`
   - Added proper torch dtype conversion (string to torch type)
   - Added `use_safetensors=False` to force PyTorch `.bin` file loading and prevent hanging

2. **`requirements.txt`**
   - Updated comment to indicate testing with version 4.55.1

## Testing

To test the fix, run:

```bash
python test_model_loading.py
```

This will attempt to load all enabled models in `config.yaml` and verify they load correctly.

## Backward Compatibility

This fix maintains backward compatibility with older versions of `transformers` (>= 4.35.0) while also supporting the latest versions (>= 4.55.1).

## Additional Notes

- The fix handles both quantized and non-quantized models
- String dtype values ('auto', 'float16', 'bfloat16', 'float32') are properly converted to torch types
- When quantization is enabled, dtype is automatically set to 'auto' as before
- No changes needed in configuration files or usage patterns

