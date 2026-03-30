# BEAVER OpenClaw Plugin

Wraps BEAVER long-text compression as an OpenClaw plugin. Type `/beaver <your question>` to automatically compress conversation context and get an answer.

## Architecture

```
User: /beaver What are the main contributions of this paper?
  → OpenClaw collects long text from conversation
  → Calls BEAVER API to compress
  → Compressed text + question → LLM answers
```

## Quick Start

### 1. Start the BEAVER API Server

```bash
pip install fastapi uvicorn

# CPU mode
python beaver_server.py --model_path Qwen/Qwen3-0.6B --port 8765

# GPU mode
python beaver_server.py --model_path Qwen/Qwen3-0.6B --mode gpu --dtype bf16 --port 8765
```

### 2. Install the Plugin

```bash
# Option A: CLI install
openclaw plugins install ./openclaw-plugin

# Option B: Manual copy
cp -r openclaw-plugin ~/.openclaw/plugins/beaver-compression

# Option C: Skill only (lightest)
cp -r openclaw-plugin/skills/beaver ~/.openclaw/skills/beaver
```

### 3. Usage

In an OpenClaw conversation:

```
# Paste your long text first, then:
/beaver What are the main contributions of this paper?
/beaver Summarize the key findings
/beaver What is the core logic of this code?
```

## Capabilities

| Method | Description |
|--------|-------------|
| `/beaver <question>` | Slash command — auto-collects conversation context, compresses, and asks |
| `beaver_compress` tool | Pure compression — returns compressed text + stats |
| `beaver_ask` tool | Compress + format prompt for LLM in one step |

## Configuration (Optional)

```json5
{
  "plugins": {
    "beaver-compression": {
      "serverUrl": "http://127.0.0.1:8765",
      "defaultPageSize": 64,
      "defaultTopK": 22
    }
  }
}
```

## API Reference

### POST /compress

Request:
```json
{
  "context": "Your long text...",
  "query": "Your question",
  "page_size": 64,
  "anchor_pages": 4,
  "flow_window": 4,
  "flash_top_k": 22,
  "semantic_weight": 0.7,
  "lexical_weight": 0.3
}
```

Response:
```json
{
  "compressed_text": "Compressed text...",
  "original_tokens": 8432,
  "compressed_tokens": 324,
  "compression_ratio": 0.0384,
  "speedup": 26.0
}
```
