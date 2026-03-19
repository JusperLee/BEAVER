#!/usr/bin/env python3
# HTML report generator for visualizing exact kept spans on the original context.
# Redesigned with a Modern/Clean aesthetic.

import json
import html
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# --- Data Loading Utilities ---

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# --- Interval Logic ---

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def normalize_kept_char_spans(kept_char_spans, original_len: int):
    """Validate and merge kept character spans on the original context."""
    if kept_char_spans is None:
        return None
    if not isinstance(kept_char_spans, (list, tuple)):
        raise ValueError(f"kept_char_spans must be a list, got {type(kept_char_spans)}")

    spans = []
    for pair in kept_char_spans:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid span entry: {pair!r}")
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        if s < 0 or e < 0 or s > original_len or e > original_len:
            pass 
        # Clamp to bounds to prevent crashes
        s = max(0, min(s, original_len))
        e = max(0, min(e, original_len))
        if s < e:
            spans.append((s, e))

    if not spans:
        return []
    return merge_intervals(spans)

def compute_kept_intervals(original: str, compressed: str, min_match_chars: int = 8):
    """Approximate kept spans via difflib matching (fallback only)."""
    if not original or not compressed:
        return []

    sm = SequenceMatcher(a=original, b=compressed, autojunk=False)
    intervals = []
    for a0, b0, size in sm.get_matching_blocks():
        if size >= min_match_chars:
            intervals.append((a0, a0 + size))
    return merge_intervals(intervals)

# --- HTML Rendering Logic ---

def render_highlight_html(original: str, kept_intervals):
    """Render the original text as HTML with kept spans highlighted."""
    parts = []
    cur = 0
    for s, e in kept_intervals:
        # Text before the kept span (Dropped)
        if cur < s:
            dropped = html.escape(original[cur:s])
            if dropped:
                parts.append(f'<span class="dropped">{dropped}</span>')
        
        # The kept span (Kept)
        kept = html.escape(original[s:e])
        if kept:
            parts.append(f'<span class="kept">{kept}</span>')
        cur = e

    # Tail (Dropped)
    if cur < len(original):
        dropped = html.escape(original[cur:])
        if dropped:
            parts.append(f'<span class="dropped">{dropped}</span>')

    return "".join(parts)

def build_report_item(idx: int, qa_row: dict, res_row: dict, min_match_chars: int):
    original = str(qa_row.get("input", ""))
    inst = str(qa_row.get("instruction", ""))
    comp = str(res_row.get("compressed_context", ""))
    ratio = res_row.get("compression_ratio", None)

    kept_char_spans = res_row.get("kept_char_spans", None)
    kept_intervals = normalize_kept_char_spans(kept_char_spans, original_len=len(original))

    if kept_intervals is None:
        if not res_row.get("_allow_approx", False):
            raise RuntimeError(
                f"Sample #{idx}: Missing `kept_char_spans`. "
                "Pass --allow_approx to use diff-based fallback."
            )
        kept_intervals = compute_kept_intervals(original, comp, min_match_chars=min_match_chars)

    highlighted_original = render_highlight_html(original, kept_intervals)

    # Statistics
    kept_chars = sum(e - s for s, e in kept_intervals)
    total_chars = len(original)
    kept_pct = (kept_chars / total_chars * 100.0) if total_chars else 0.0
    kept_spans_n = len(kept_intervals)

    original_len = res_row.get("original_len", total_chars)
    compressed_len = res_row.get("compressed_len", 0)

    # HTML Components
    inst_html = html.escape(inst)
    
    # Logic for Badge Colors based on ratio
    ratio_color_class = "neutral"
    if ratio:
        if ratio < 0.3: ratio_color_class = "success"
        elif ratio > 0.8: ratio_color_class = "warning"

    stats_html = []
    if ratio is not None:
        stats_html.append(f'''
            <div class="stat-pill {ratio_color_class}">
                <span class="label">Compression</span>
                <span class="value">{ratio:.2f}x</span>
            </div>
        ''')
    
    if original_len is not None and compressed_len is not None:
        stats_html.append(f'''
            <div class="stat-pill neutral">
                <span class="label">Tokens</span>
                <span class="value">{original_len} &rarr; {compressed_len}</span>
            </div>
        ''')

    stats_html.append(f'''
        <div class="stat-pill neutral">
            <span class="label">Kept</span>
            <span class="value">{kept_pct:.1f}% ({kept_spans_n} spans)</span>
        </div>
    ''')

    stats_block = "".join(stats_html)

    item_html = f"""
    <article class="card" id="item-{idx}">
      <header class="card-header">
        <div class="header-left">
            <span class="sample-id">#{idx}</span>
            <div class="stats-row">
                {stats_block}
            </div>
        </div>
      </header>

      <div class="content-body">
        
        <div class="section-box">
            <div class="section-label">Instruction / Query</div>
            <div class="instruction-text">{inst_html}</div>
        </div>

        <div class="section-box context-box">
             <div class="section-header">
                <div class="section-label">Original Context</div>
                <div class="legend">
                    <span class="legend-dot kept"></span> Kept
                    <span class="legend-dot dropped"></span> Dropped
                </div>
            </div>
            <div class="original-text-wrapper">{highlighted_original}</div>
        </div>
      </div>
    </article>
    """
    return item_html

def build_full_html(items_html: str, title: str):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg-app: #f3f4f6;
      --bg-card: #ffffff;
      
      --text-main: #1f2937;
      --text-muted: #9ca3af;
      --text-light: #d1d5db;
      
      --border-subtle: #e5e7eb;
      --border-focus: #d1d5db;

      --primary: #4f46e5;
      --primary-bg: #eef2ff;
      
      --success-text: #065f46;
      --success-bg: #d1fae5;
      
      --dropped-text: #9ca3af;
      
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      
      --font-serif: "Times New Roman", Times, serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}

    * {{ box-sizing: border-box; }}
    
    body {{
      margin: 0;
      padding: 0;
      background-color: var(--bg-app);
      color: var(--text-main);
      font-family: var(--font-serif);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}

    .container {{
      max-width: 1000px;
      margin: 0 auto;
      padding: 40px 20px;
    }}

    .app-header {{
      margin-bottom: 40px;
      text-align: center;
    }}

    .app-title {{
      font-size: 24px;
      font-weight: 700;
      color: #111827;
      margin: 0 0 8px 0;
      letter-spacing: -0.025em;
    }}

    .app-subtitle {{
      color: var(--text-muted);
      font-size: 14px;
    }}

    .card {{
      background: var(--bg-card);
      border-radius: 12px;
      box-shadow: var(--shadow-sm);
      margin-bottom: 32px;
      border: 1px solid var(--border-subtle);
      overflow: hidden;
      transition: box-shadow 0.2s ease;
    }}
    
    .card:hover {{
        box-shadow: var(--shadow-md);
        border-color: var(--border-focus);
    }}

    .card-header {{
      padding: 16px 24px;
      border-bottom: 1px solid var(--border-subtle);
      background: #f9fafb; /* Very light gray */
    }}

    .header-left {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }}

    .sample-id {{
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--primary);
      background: var(--primary-bg);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 13px;
    }}

    .stats-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .stat-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      font-size: 12px;
      background: #fff;
      color: var(--text-main);
    }}
    
    .stat-pill.success {{ border-color: #a7f3d0; background: #ecfdf5; color: #064e3b; }}
    .stat-pill.warning {{ border-color: #fde68a; background: #fffbeb; color: #92400e; }}
    
    .stat-pill .label {{
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    
    .stat-pill .value {{
        font-weight: 600;
        font-family: var(--font-mono);
    }}

    .content-body {{
      padding: 24px;
    }}

    .section-box {{
      margin-bottom: 24px;
    }}
    .section-box:last-child {{ margin-bottom: 0; }}

    .section-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}

    .section-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}

    .instruction-text {{
      background: #f8fafc;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 12px 16px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: #374151;
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .legend {{
        font-size: 12px;
        color: var(--text-muted);
        display: flex;
        gap: 12px;
    }}
    .legend-dot {{
        display: inline-block;
        width: 8px; 
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }}
    .legend-dot.kept {{ background: #10b981; }}
    .legend-dot.dropped {{ background: #d1d5db; }}

    .original-text-wrapper {{
      font-family: var(--font-serif);
      font-size: 16px; /* Increased slightly for Times New Roman readability */
      line-height: 1.65;
      color: var(--dropped-text); /* Default to dropped color */
      white-space: pre-wrap;
      word-break: break-word;
      background: #ffffff;
    }}
    
    .kept {{
        background-color: var(--success-bg);
        color: var(--success-text);
        border-radius: 3px;
        padding: 1px 0; /* Slight padding for the highlighter look */
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
        font-weight: 500;
    }}
    
    .dropped {{
        color: var(--dropped-text);
        /* Optional: text-decoration: line-through; opacity: 0.7; */
    }}

  </style>
</head>
<body>
  <div class="container">
    <div class="app-header">
      <div class="app-title">{html.escape(title)}</div>
      <div class="app-subtitle">Visualizing compressed context • Kept spans highlighted</div>
    </div>
    
    {items_html}
    
    <div style="text-align:center; color: #9ca3af; font-size: 12px; margin-top: 50px;">
        Generated by Beaver Visualizer
    </div>
  </div>
</body>
</html>
"""

# --- Main Driver ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_jsonl", type=str, default="./QA.jsonl")
    ap.add_argument("--result_json", type=str, default="./QA_Result.json")
    ap.add_argument("--out_html", type=str, default="./QA_Report.html")
    ap.add_argument("--min_match_chars", type=int, default=8)
    ap.add_argument("--allow_approx", action="store_true", help="Allow difflib matching if exact spans missing")
    args = ap.parse_args()

    qa_rows = read_jsonl(Path(args.qa_jsonl))
    res_rows = read_json(Path(args.result_json))

    res_by_idx = {}
    for i, r in enumerate(res_rows):
        ridx = r.get("idx", i)
        r["_allow_approx"] = bool(args.allow_approx)
        res_by_idx[int(ridx)] = r

    items = []
    for i, qa in enumerate(qa_rows):
        r = res_by_idx.get(i)
        if r is None:
            continue
        items.append(build_report_item(i, qa, r, min_match_chars=args.min_match_chars))

    title = f"Compression Report - {Path(args.qa_jsonl).name}"
    html_text = build_full_html("\n".join(items), title)

    Path(args.out_html).write_text(html_text, encoding="utf-8")
    print(f"[OK] wrote: {args.out_html}")

if __name__ == "__main__":
    main()