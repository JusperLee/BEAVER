#!/usr/bin/env python3
"""BEAVER Gradio Demo — Interactive long-text compression with visualization and LLM generation."""

import argparse
import html as html_mod
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Wrapper import HSPBlackBoxWrapper, HSPWrapperConfig
from Demo import split_compressed_context, token_indices_to_char_spans
from Build_Html import normalize_kept_char_spans, render_highlight_html, compute_kept_intervals

# ---------------------------------------------------------------------------
# Global state (set once at startup)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device: torch.device = torch.device("cpu")
_mode: str = "cpu"
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# CSS for highlighted output (embedded in HTML)
# ---------------------------------------------------------------------------
HIGHLIGHT_CSS = """
<style>
.beaver-vis { font-family: "SF Mono", "Menlo", "Consolas", monospace; font-size: 13px;
  line-height: 1.7; white-space: pre-wrap; word-break: break-word;
  max-height: 500px; overflow-y: auto; padding: 16px;
  border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
.beaver-vis .kept { background: rgba(72, 199, 142, 0.35); border-radius: 2px; }
.beaver-vis .dropped { color: #b0b0b0; }
</style>
"""

STATS_BAR_CSS = """
<style>
.beaver-stats-bar { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
.beaver-stat-pill { display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 500; }
.beaver-stat-pill.original { background: #fee2e2; color: #991b1b; }
.beaver-stat-pill.compressed { background: #d1fae5; color: #065f46; }
.beaver-stat-pill.ratio { background: #dbeafe; color: #1e40af; }
.beaver-stat-pill.speedup { background: #fef3c7; color: #92400e; }
.beaver-stat-pill .num { font-size: 18px; font-weight: 700; }
</style>
"""

# ---------------------------------------------------------------------------
# Auto-suggest hyperparameters based on document length
# ---------------------------------------------------------------------------

def suggest_params(context: str) -> Tuple[int, int, int, int, float, float]:
    """Suggest hyperparameters based on document token count. Returns (page_size, anchor, flow, topk, sem_w, lex_w)."""
    if not context or not context.strip():
        return 64, 4, 4, 22, 0.7, 0.3
    # Rough token estimate: ~1.3 tokens per word for English, ~1.5 per char for Chinese
    word_count = len(context.split())
    char_count = len(context)
    # Heuristic: use whichever gives a higher estimate
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

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(file_path: str) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        return "[Error] pymupdf not installed. Run: pip install pymupdf"
    try:
        doc = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        return f"[Error] Failed to extract PDF text: {e}"


# ---------------------------------------------------------------------------
# Core compression
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_compression(
    context: str,
    query: str,
    page_size: int,
    anchor_pages: int,
    flow_window: int,
    flash_top_k: int,
    semantic_weight: float,
    lexical_weight: float,
) -> Tuple[str, str, str]:
    """Run BEAVER compression. Returns (stats_json, highlight_html, compressed_text)."""
    if not context or not context.strip():
        return '{"error": "Please provide context text."}', "", ""
    if not query or not query.strip():
        return '{"error": "Please provide a query/question."}', "", ""

    cfg = HSPWrapperConfig(
        page_size=int(page_size),
        anchor_pages=int(anchor_pages),
        flow_window=int(flow_window),
        flash_top_k=int(flash_top_k),
        semantic_score_weight=float(semantic_weight),
        lexical_score_weight=float(lexical_weight),
    )

    with _lock:
        wrapper = HSPBlackBoxWrapper(_model, _tokenizer, cfg, device=_device)

        # Get offset mapping for char-span tracking
        ctx_enc = _tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        offset_mapping = [[int(s), int(e)] for s, e in ctx_enc["offset_mapping"]]
        ctx_token_len = len(ctx_enc["input_ids"])

        # Build inputs and compress
        input_ids, attention_mask, explicit_qp = wrapper._build_inputs_from_texts([context], [query])
        compressed, stats = wrapper.compress_inputs_for_prefill(input_ids, attention_mask, explicit_qp)

    # Extract kept char spans
    kept_idx_list = stats.get("kept_context_token_indices", [[]])
    kept_idx = kept_idx_list[0] if kept_idx_list else []
    kept_char_spans = token_indices_to_char_spans(offset_mapping, [int(x) for x in kept_idx])

    # Get compressed context text
    comp_ctx = split_compressed_context(
        tokenizer=_tokenizer,
        compressed_input_ids=compressed["input_ids"][0],
        compressed_attention_mask=compressed["attention_mask"][0],
        instruction=query,
    )

    # Normalize spans for visualization
    kept_intervals = normalize_kept_char_spans(kept_char_spans, len(context))
    if not kept_intervals:
        # Fallback: diff-based matching
        kept_intervals = compute_kept_intervals(context, comp_ctx)

    # Build highlighted HTML
    inner_html = render_highlight_html(context, kept_intervals)
    highlight_html = HIGHLIGHT_CSS + f'<div class="beaver-vis">{inner_html}</div>'

    # Build stats bar (visual pills) + JSON detail
    orig_tok = stats["original_len"]
    comp_tok = stats["compressed_len"]
    ratio = stats["compression_ratio"]
    kept_chars = sum(e - s for s, e in (kept_intervals or []))
    speedup = round(1.0 / max(ratio, 1e-6), 1)

    stats_html = STATS_BAR_CSS + f"""
<div class="beaver-stats-bar">
  <span class="beaver-stat-pill original">Original &nbsp;<span class="num">{orig_tok}</span>&nbsp;tokens</span>
  <span class="beaver-stat-pill compressed">Compressed &nbsp;<span class="num">{comp_tok}</span>&nbsp;tokens</span>
  <span class="beaver-stat-pill ratio">Ratio &nbsp;<span class="num">{ratio:.2%}</span></span>
  <span class="beaver-stat-pill speedup">~<span class="num">{speedup}x</span>&nbsp;speedup</span>
</div>
<details style="margin-top:8px;font-size:13px;color:#555;">
  <summary>Detailed stats</summary>
  <pre style="margin:4px 0;font-size:12px;">{json.dumps({
      "original_tokens": orig_tok, "compressed_tokens": comp_tok,
      "compression_ratio": round(ratio, 4), "speedup": speedup,
      "kept_spans": len(kept_intervals) if kept_intervals else 0,
      "kept_chars": kept_chars, "total_chars": len(context),
      "char_keep_rate": round(kept_chars / max(len(context), 1), 4),
  }, indent=2)}</pre>
</details>
"""

    return stats_html, highlight_html, comp_ctx


# ---------------------------------------------------------------------------
# External LLM API generation
# ---------------------------------------------------------------------------

def call_external_llm(
    compressed_context: str,
    query: str,
    provider: str,
    api_key: str,
    base_url: str,
    model_name: str,
) -> str:
    """Send compressed context + query to an external LLM API."""
    if not compressed_context or not compressed_context.strip():
        return "[Error] No compressed context. Please run compression first."
    if not query or not query.strip():
        return "[Error] No query provided."

    prompt_text = f"Context:\n{compressed_context}\n\nQuestion: {query}\n\nAnswer:"
    base_url_clean = base_url.strip() if base_url and base_url.strip() else None

    if provider == "Local Model":
        if _mode != "gpu":
            return "[Error] Local generation is only available in GPU mode."
        from Demo import generate_answer
        try:
            with _lock:
                answer = generate_answer(_model, _tokenizer, prompt_text, _device, 1024)
            return answer
        except Exception as e:
            return f"[Error] Local generation failed: {e}"

    if not api_key or not api_key.strip():
        return "[Error] Please provide an API key."

    if provider in ("OpenAI", "OpenAI-Compatible"):
        try:
            import openai
        except ImportError:
            return "[Error] openai not installed. Run: pip install openai"
        try:
            client_kwargs = {"api_key": api_key.strip()}
            if base_url_clean:
                client_kwargs["base_url"] = base_url_clean
            client = openai.OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
                    {"role": "user", "content": prompt_text},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error] OpenAI API call failed: {e}"

    elif provider == "Claude":
        try:
            import anthropic
        except ImportError:
            return "[Error] anthropic not installed. Run: pip install anthropic"
        try:
            client_kwargs = {"api_key": api_key.strip()}
            if base_url_clean:
                client_kwargs["base_url"] = base_url_clean
            client = anthropic.Anthropic(**client_kwargs)
            resp = client.messages.create(
                model=model_name or "claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            return f"[Error] Claude API call failed: {e}"

    return f"[Error] Unknown provider: {provider}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    global _model, _tokenizer, _device, _mode
    _mode = args.mode

    if args.mode == "gpu":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(args.dtype, None)
    else:
        _device = torch.device("cpu")
        torch_dtype = torch.float32

    model_path = args.model_path
    print(f"[BEAVER] Loading model: {model_path} | mode={args.mode} | device={_device} | dtype={args.dtype}")

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True,
    ).eval().to(_device)

    print(f"[BEAVER] Model loaded successfully.")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr

    provider_choices = ["OpenAI", "Claude", "OpenAI-Compatible"]
    if _mode == "gpu":
        provider_choices.append("Local Model")

    with gr.Blocks(
        title="BEAVER: Hierarchical Prompt Compression",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1100px !important; margin: 0 auto !important; }",
    ) as demo:

        # ---- Paper info header ----
        gr.HTML("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
          <h1 style="margin:0 0 4px 0; font-size:28px;">
            BEAVER: Training-Free Hierarchical Prompt Compression
            <br><span style="font-size:16px; font-weight:400; color:#666;">via Structure-Aware Page Selection</span>
          </h1>
          <p style="margin:8px 0 4px 0; font-size:14px; color:#444;">
            <strong>Zhengpei Hu</strong><sup>1*</sup>,
            <strong>Kai Li</strong><sup>2*</sup>,
            <strong>Dapeng Fu</strong><sup>3</sup>,
            <strong>Chang Zeng</strong><sup>4</sup>,
            <strong>Yue Li</strong><sup>1</sup>,
            <strong>Yuanhao Tang</strong><sup>1</sup>,
            <strong>Jianqiang Huang</strong><sup>1&dagger;</sup>
          </p>
          <p style="margin:2px 0 8px 0; font-size:12px; color:#888;">
            <sup>1</sup>Qinghai University &nbsp;
            <sup>2</sup>Tsinghua University &nbsp;
            <sup>3</sup>Ant Group SIL &nbsp;
            <sup>4</sup>National Institute of Informatics &nbsp;&nbsp;
            <span style="color:#aaa;">* Equal contribution &nbsp; &dagger; Corresponding author</span>
          </p>
          <p style="margin:0; font-size:14px;">
            <a href="https://arxiv.org/abs/2603.19635" target="_blank" style="margin:0 8px;">&#x1F4DC; Paper</a>
            <a href="https://cslikai.cn/BEAVER" target="_blank" style="margin:0 8px;">&#x1F3AC; Demo Page</a>
            <a href="https://github.com/JusperLee/BEAVER" target="_blank" style="margin:0 8px;">&#x2B50; GitHub</a>
          </p>
        </div>
        """)
        gr.Markdown(
            "> **BEAVER** is a training-free, structure-aware prompt compression framework that keeps discourse "
            "integrity while delivering extreme efficiency (~26x speedup) on long-context LLMs. "
            "Upload your long text or PDF below, adjust compression parameters, and see the results instantly."
        )

        # ---- Input section ----
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Text Input"):
                    context_box = gr.Textbox(
                        label="Context (long text)",
                        lines=12,
                        placeholder="Paste your long document here...",
                    )
                with gr.Tab("PDF Upload"):
                    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                    pdf_status = gr.Textbox(label="Extraction Status", interactive=False, lines=1)
            with gr.Column(scale=1):
                query_box = gr.Textbox(
                    label="Query / Question",
                    lines=3,
                    placeholder="What would you like to ask about this text?",
                )
                gr.Markdown(
                    "<small style='color:#888;'>The query guides which pages are most relevant. "
                    "BEAVER scores each page against your query using semantic + lexical matching.</small>"
                )

        # ---- Compression parameters ----
        with gr.Accordion("Compression Parameters", open=True):
            gr.Markdown(
                "**Tip:** Click **Auto Suggest** to get recommended parameters based on your document length. "
                "Shorter texts need fewer pages; longer texts benefit from higher `flash_top_k`."
            )
            auto_btn = gr.Button("Auto Suggest Parameters", variant="secondary", size="sm")
            with gr.Row():
                sl_page_size = gr.Slider(
                    16, 256, value=64, step=16,
                    label="Page Size",
                    info="Tokens per page. Smaller = finer granularity but slower. Default: 64",
                )
                sl_anchor = gr.Slider(
                    0, 16, value=4, step=1,
                    label="Anchor Pages",
                    info="Force-keep first N pages (title, intro). Preserves structural info. Default: 4",
                )
            with gr.Row():
                sl_flow = gr.Slider(
                    0, 16, value=4, step=1,
                    label="Flow Window",
                    info="Sliding window around selected pages for local coherence. Default: 4",
                )
                sl_topk = gr.Slider(
                    1, 64, value=22, step=1,
                    label="Flash Top-K",
                    info="Number of highest-scoring pages to keep via semantic+lexical matching. Default: 22",
                )
            with gr.Row():
                sl_sem = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05,
                    label="Semantic Weight",
                    info="Weight for embedding cosine similarity. Higher = more meaning-based. Default: 0.7",
                )
                sl_lex = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="Lexical Weight",
                    info="Weight for token overlap. Higher = more keyword-based. Default: 0.3",
                )

        compress_btn = gr.Button("Compress", variant="primary", size="lg")

        # ---- Results section ----
        gr.Markdown("### Results")
        stats_output = gr.HTML(label="Compression Statistics")
        highlight_output = gr.HTML(label="Visualization (green=kept, gray=dropped)")
        compressed_text = gr.Textbox(label="Compressed Text", interactive=False, lines=8)

        # ---- LLM generation section ----
        with gr.Accordion("LLM Generation (optional)", open=False):
            with gr.Row():
                provider_dd = gr.Dropdown(
                    choices=provider_choices, value="OpenAI", label="API Provider",
                    info="OpenAI-Compatible: any API with OpenAI-compatible interface (vLLM, Ollama, DeepSeek, etc.)",
                )
                api_key_box = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
            with gr.Row():
                base_url_box = gr.Textbox(
                    label="Base URL (optional)",
                    placeholder="e.g., https://api.deepseek.com/v1, http://localhost:11434/v1",
                    info="Leave empty for official API endpoints. Set for self-hosted or third-party providers.",
                )
                model_name_box = gr.Textbox(label="Model Name", value="gpt-4o-mini", placeholder="e.g., gpt-4o-mini")
            generate_btn = gr.Button("Generate Answer", variant="secondary")
            answer_output = gr.Textbox(label="Generated Answer", interactive=False, lines=6)

        # ---- Event handlers ----
        def on_auto_suggest(context):
            page_size, anchor, flow, topk, sem_w, lex_w = suggest_params(context)
            word_count = len(context.split()) if context else 0
            est_tokens = max(int(word_count * 1.3), int(len(context or "") / 3))
            gr.Info(f"Suggested params for ~{est_tokens} estimated tokens")
            return page_size, anchor, flow, topk, sem_w, lex_w

        auto_btn.click(
            fn=on_auto_suggest,
            inputs=[context_box],
            outputs=[sl_page_size, sl_anchor, sl_flow, sl_topk, sl_sem, sl_lex],
        )

        def on_pdf_upload(file):
            if file is None:
                return "", "No file uploaded."
            text = extract_pdf_text(file.name)
            if text.startswith("[Error]"):
                return "", text
            char_count = len(text)
            return text, f"Extracted {char_count} characters from PDF."

        pdf_upload.change(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[context_box, pdf_status],
        )

        compress_btn.click(
            fn=run_compression,
            inputs=[context_box, query_box, sl_page_size, sl_anchor, sl_flow, sl_topk, sl_sem, sl_lex],
            outputs=[stats_output, highlight_output, compressed_text],
            api_name="compress",
        )

        generate_btn.click(
            fn=call_external_llm,
            inputs=[compressed_text, query_box, provider_dd, api_key_box, base_url_box, model_name_box],
            outputs=[answer_output],
            api_name="generate",
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BEAVER Gradio Demo")
    ap.add_argument("--mode", type=str, default="cpu", choices=["gpu", "cpu"],
                    help="gpu: full model on CUDA; cpu: lightweight model on CPU")
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B",
                    help="HuggingFace model name or local path")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="Model dtype (GPU mode only, CPU always uses fp32)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = ap.parse_args()

    load_model(args)
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
