#!/usr/bin/env python3
# Demo runner for BEAVER compression + generation.
# Reads a JSONL dataset, compresses context (keeping exact keep-trace), and writes a JSON report.

import os
import json
import re
import argparse
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Wrapper import HSPBlackBoxWrapper, HSPWrapperConfig


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_get_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, str):
        return x
    return str(x)


def _kmp_build_lps(pattern: List[int]) -> List[int]:
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps


def _kmp_find_last(text: List[int], pattern: List[int]) -> int:
    """Return last start index of pattern in text, or -1."""
    if not pattern:
        return 0
    rt = list(reversed(text))
    rp = list(reversed(pattern))
    lps = _kmp_build_lps(rp)
    j = 0
    ridx = -1
    for i in range(len(rt)):
        while j > 0 and rt[i] != rp[j]:
            j = lps[j - 1]
        if rt[i] == rp[j]:
            j += 1
            if j == len(rp):
                ridx = i - len(rp) + 1
                break
    if ridx < 0:
        return -1
    return len(text) - (ridx + len(pattern))


def split_compressed_context(
    tokenizer,
    compressed_input_ids: torch.Tensor,     # [L]
    compressed_attention_mask: torch.Tensor, # [L]
    instruction: str,
) -> str:
    """Recover the compressed context by stripping the (uncompressed) query suffix when possible."""
    comp_attn = compressed_attention_mask.bool()
    ids_valid = compressed_input_ids[comp_attn]
    ids_list = ids_valid.detach().cpu().tolist()

    q_ids = tokenizer.encode(instruction, add_special_tokens=False)
    q_len = len(q_ids)
    if q_len > 0 and q_len <= len(ids_list):
        suffix = ids_list[-q_len:]
        if suffix == q_ids:
            ctx_ids = ids_valid[:-q_len]
            return tokenizer.decode(ctx_ids, skip_special_tokens=True)
    if q_len > 0:
        q_start = _kmp_find_last(ids_list, q_ids)
        if q_start >= 0:
            ctx_ids = ids_valid[:q_start]
            return tokenizer.decode(ctx_ids, skip_special_tokens=True)
    return tokenizer.decode(ids_valid, skip_special_tokens=True)


def token_indices_to_char_spans(offset_mapping: List[List[int]], kept_indices: List[int]) -> List[List[int]]:
    """Convert kept token indices to merged character spans on the original context."""
    spans: List[List[int]] = []
    if not offset_mapping or not kept_indices:
        return spans

    for ti in kept_indices:
        if ti < 0 or ti >= len(offset_mapping):
            continue
        s, e = offset_mapping[ti]
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        spans.append([s, e])

    if not spans:
        return []

    spans.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_pos, int) and max_pos > 0 and input_ids.shape[1] > max_pos:
        input_ids = input_ids[:, -max_pos:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_pos:]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
    )[0]

    new_tokens = out[input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--model_path", type=str, required=True, help='HF casual LLM. e.g. Qwen,Llama')
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])

    ap.add_argument("--start", type=int, default=0, help=' The starting point for testing samples')
    ap.add_argument("--limit", type=int, default=1, help=' The endpoint of the test sample')

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--page_size", type=int, default=64)
    ap.add_argument("--anchor_pages", type=int, default=1)
    ap.add_argument("--flow_window", type=int, default=1)
    ap.add_argument("--flash_top_k", type=int, default=1)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch_dtype = None
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32

    print(f"[Load] model={args.model_path} device={device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Exact kept_char_spans export requires a fast tokenizer (use_fast=True).")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).eval().to(device)

    cfg = HSPWrapperConfig(
        page_size=int(args.page_size),
        anchor_pages=int(args.anchor_pages),
        flow_window=int(args.flow_window),
        flash_top_k=int(args.flash_top_k),
    )
    wrapper = HSPBlackBoxWrapper(model, tokenizer, cfg, idf_weights=None, device=device)

    results: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    seen = 0
    kept = 0
    for idx, obj in enumerate(read_jsonl(args.in_jsonl)):
        if idx < args.start:
            continue
        if args.limit >= 0 and kept >= args.limit:
            break

        context = safe_get_str(obj.get("input", ""), "")
        instruction = safe_get_str(obj.get("instruction", ""), "")
        gold = safe_get_str(obj.get("output", ""), "")

        ctx_enc = tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ctx_token_ids = ctx_enc["input_ids"]
        offset_mapping = ctx_enc["offset_mapping"]
        offset_mapping = [[int(s), int(e)] for (s, e) in offset_mapping]
        ctx_len = len(ctx_token_ids)

        input_ids, attention_mask, explicit_qp = wrapper._build_inputs_from_texts([context], [instruction])
        compressed, stats = wrapper.compress_inputs_for_prefill(input_ids, attention_mask, explicit_qp)

        kept_idx_list = stats.get("kept_context_token_indices", [[]])
        kept_idx = kept_idx_list[0] if kept_idx_list else []

        qp_list = stats.get("aligned_query_pos", [ctx_len])
        qp = int(qp_list[0]) if qp_list else ctx_len
        if qp != ctx_len:
            raise RuntimeError(f"aligned_query_pos ({qp}) != context token length ({ctx_len}). Cannot export exact kept_char_spans.")

        kept_char_spans = token_indices_to_char_spans(offset_mapping, [int(x) for x in kept_idx])

        comp_ctx = split_compressed_context(
            tokenizer=tokenizer,
            compressed_input_ids=compressed["input_ids"][0],
            compressed_attention_mask=compressed["attention_mask"][0],
            instruction=instruction,
        )

        prompt = comp_ctx + "\n\n" + instruction
        answer = generate_answer(model, tokenizer, prompt, device, args.max_new_tokens)

        results.append({
            "idx": idx,
            "compression_ratio": float(stats["compression_ratio"]),
            "original_len": int(stats["original_len"]),
            "compressed_len": int(stats["compressed_len"]),
            "compressed_context": comp_ctx,
            "kept_char_spans": kept_char_spans,
            "model_answer": answer,
            "gold_output": gold,
        })

        kept += 1
        if kept % 10 == 0:
            print(f"[Progress] {kept} samples processed...")

        seen += 1

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(results)} records to: {args.out_json}")


if __name__ == "__main__":
    main()