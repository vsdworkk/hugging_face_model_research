# WFA Profile Analyzer

## Quantization Support (8-bit / 4-bit)

You can enable model quantization per model in `config/config.yaml` using the `quantization` field:

```yaml
models:
  - name: "gemma-3-1b-it"
    enabled: true
    model_id: "google/gemma-3-1b-it"
    is_instruct: true
    device_map: "auto"
    torch_dtype: "auto"
    quantization: "4bit"   # or "8bit" or "none"
```

Notes:
- Quantization requires a CUDA-enabled environment and the `bitsandbytes` package.
- Install with: `pip install bitsandbytes`
- 4-bit settings are fixed internally to: `nf4`, double-quant enabled, compute dtype `bfloat16`.
- When quantization is enabled, the top-level `torch_dtype` is ignored (kept `auto`).
- macOS (MPS) is not supported for 4/8-bit `bitsandbytes` quantization.


