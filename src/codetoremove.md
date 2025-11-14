
1. Remove the below debugging code we don't need it anymore.

def debug_print_pipeline_args(args: Dict[str, Any]) -> None:
    """Print the final pipeline call exactly as it will be executed (token masked)."""
    tok_mask = "***" if args.get("token") else None
    td_str = args.get("torch_dtype")
    td_str = "auto" if td_str == "auto" else str(td_str)
    print(
        f"  pipeline(\"text-generation\",\n"
        f"    model=\"{args['model']}\",\n"
        f"    device_map=\"{args.get('device_map', 'auto')}\",\n"
        f"    torch_dtype=\"{td_str}\",\n"
        f"    token={tok_mask}\n"
        f"  )"
    )

2. High Priority: Refactor process_in_batches for Simplicity. Let's not process harmony models in batches, but instead let's do it individually, removes the need to manually implement batching and makes the implementation a bit simpler.


3. Let's remove this debug statement:

                if debug_enabled and (not parsed_output):
                    # Attempt a safer boundary search
                    alt_boundary = search_boundary_by_suffix(row, prompt_ids)
                    alt_completion_tokens = row[alt_boundary:]
                    alt_parsed = parse_harmony_response(alt_completion_tokens)

                    # Prepare debug preview strings
                    preview = pipe.tokenizer.decode(completion_tokens, skip_special_tokens=False)
                    alt_preview = pipe.tokenizer.decode(alt_completion_tokens, skip_special_tokens=False)
                    preview_short = preview[:240].replace("\n", "\\n")
                    alt_preview_short = alt_preview[:240].replace("\n", "\\n")

                    # Check if EOS/stop tokens appear at the end of either completion
                    row_last = row[-8:]
                    stop_hits = [tok for tok in row_last if tok in stop_token_ids]

                    print(
                        f"[Harmony DEBUG] batch={i//batch_size} item={j} len(row)={len(row)} max_len={max_len} "
                        f"prompt_len={prompt_len} pad_len={pad_len} left_pad={has_left_pad_prefix} "
                        f"boundary={boundary} alt_boundary={alt_boundary} gen_len={len(completion_tokens)} "
                        f"alt_gen_len={len(alt_completion_tokens)} stop_tail_hits={len(stop_hits)}"
                    )
                    print(f"[Harmony DEBUG] preview: {preview_short}")
                    print(f"[Harmony DEBUG] alt_preview: {alt_preview_short}")

                    # Prefer alt_parsed if it yields a result
                    if alt_parsed:
                        parsed_output = alt_parsed

                outputs.append(parsed_output or "")