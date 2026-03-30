#!/usr/bin/env python3
"""BEAVER Compression API Server — exposes BEAVER as a REST endpoint for OpenClaw plugin."""

import argparse
import threading
from typing import Optional, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from Wrapper import HSPBlackBoxWrapper, HSPWrapperConfig
from Demo import split_compressed_context, token_indices_to_char_spans

app = FastAPI(title="BEAVER Compression API", version="1.0.0")

_model = None
_tokenizer = None
_device: torch.device = torch.device("cpu")
_lock = threading.Lock()


def suggest_params(context: str) -> Tuple[int, int, int, int, float, float]:
    """Auto-suggest hyperparameters based on document length."""
    if not context or not context.strip():
        return 64, 4, 4, 22, 0.7, 0.3
    word_count = len(context.split())
    char_count = len(context)
    est_tokens = max(int(word_count * 1.3), int(char_count / 3))

    if est_tokens < 512:
        return 32, 2, 2, 2, 0.7, 0.3
    elif est_tokens < 2048:
        return 32, 4, 4, 4, 0.7, 0.3
    elif est_tokens < 8192:
        return 64, 2, 2, 8, 0.7, 0.3
    elif est_tokens < 32768:
        return 64, 4, 4, 8, 0.7, 0.3
    else:
        return 128, 4, 4, 32, 0.7, 0.3


class CompressRequest(BaseModel):
    context: str = Field(..., description="Long text to compress")
    query: str = Field(..., description="Query/question guiding compression")
    page_size: Optional[int] = Field(None, ge=16, le=256, description="Auto-detected if omitted")
    anchor_pages: Optional[int] = Field(None, ge=0, le=16, description="Auto-detected if omitted")
    flow_window: Optional[int] = Field(None, ge=0, le=16, description="Auto-detected if omitted")
    flash_top_k: Optional[int] = Field(None, ge=1, le=64, description="Auto-detected if omitted")
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Auto-detected if omitted")
    lexical_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Auto-detected if omitted")


class CompressResponse(BaseModel):
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    speedup: float


@app.post("/compress", response_model=CompressResponse)
@torch.no_grad()
def compress(req: CompressRequest):
    # Auto-detect params if not provided
    auto_ps, auto_ap, auto_fw, auto_tk, auto_sw, auto_lw = suggest_params(req.context)

    cfg = HSPWrapperConfig(
        page_size=req.page_size if req.page_size is not None else auto_ps,
        anchor_pages=req.anchor_pages if req.anchor_pages is not None else auto_ap,
        flow_window=req.flow_window if req.flow_window is not None else auto_fw,
        flash_top_k=req.flash_top_k if req.flash_top_k is not None else auto_tk,
        semantic_score_weight=req.semantic_weight if req.semantic_weight is not None else auto_sw,
        lexical_score_weight=req.lexical_weight if req.lexical_weight is not None else auto_lw,
    )

    with _lock:
        wrapper = HSPBlackBoxWrapper(_model, _tokenizer, cfg, device=_device)
        ctx_enc = _tokenizer(req.context, add_special_tokens=False, return_offsets_mapping=True)
        input_ids, attention_mask, explicit_qp = wrapper._build_inputs_from_texts(
            [req.context], [req.query]
        )
        compressed, stats = wrapper.compress_inputs_for_prefill(input_ids, attention_mask, explicit_qp)

    comp_ctx = split_compressed_context(
        tokenizer=_tokenizer,
        compressed_input_ids=compressed["input_ids"][0],
        compressed_attention_mask=compressed["attention_mask"][0],
        instruction=req.query,
    )

    orig_tok = stats["original_len"]
    comp_tok = stats["compressed_len"]
    ratio = stats["compression_ratio"]
    speedup = round(1.0 / max(ratio, 1e-6), 1)

    return CompressResponse(
        compressed_text=comp_ctx,
        original_tokens=orig_tok,
        compressed_tokens=comp_tok,
        compression_ratio=round(ratio, 4),
        speedup=speedup,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def load_model(model_path: str, mode: str = "cpu", dtype: str = "bf16"):
    global _model, _tokenizer, _device

    if mode == "gpu":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, None)
    else:
        _device = torch.device("cpu")
        torch_dtype = torch.float32

    print(f"[BEAVER API] Loading model: {model_path} | device={_device} | dtype={dtype}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True,
    ).eval().to(_device)
    print("[BEAVER API] Model loaded.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BEAVER Compression API Server")
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--mode", type=str, default="cpu", choices=["gpu", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    load_model(args.model_path, args.mode, args.dtype)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
