---
name: beaver
description: Compress long text using BEAVER hierarchical prompt compression, then answer the question based on compressed context. Usage: /beaver <your question>
command-dispatch: tool
metadata:
  openclaw:
    requires:
      bins:
        - curl
---

# BEAVER Prompt Compression Skill

You are using the BEAVER prompt compression tool. When the user invokes `/beaver`, follow these steps:

## Step 1: Identify Context and Query

The user's message after `/beaver` is the **query**. The **context** is the long text from the current conversation (previous messages, uploaded documents, or pasted text).

If no long context is available in the conversation, ask the user to provide the text they want to compress.

## Step 2: Compress via BEAVER API

Use the `exec` tool to call the BEAVER compression API:

```bash
curl -s -X POST http://127.0.0.1:8765/compress \
  -H "Content-Type: application/json" \
  -d '{
    "context": "<THE_LONG_TEXT>",
    "query": "<THE_USER_QUERY>",
    "page_size": 64,
    "anchor_pages": 4,
    "flow_window": 4,
    "flash_top_k": 22,
    "semantic_weight": 0.7,
    "lexical_weight": 0.3
  }'
```

The API returns JSON with these fields:
- `compressed_text`: The compressed context
- `original_tokens`: Original token count
- `compressed_tokens`: Compressed token count
- `compression_ratio`: Compression ratio
- `speedup`: Speedup factor

## Step 3: Answer the Question

After getting the compressed text, use it as context to answer the user's query. Format your response like this:

1. First, briefly show compression stats (original tokens → compressed tokens, speedup)
2. Then answer the question based on the compressed context

## Error Handling

If the BEAVER API is not reachable, tell the user to start the server:

```
BEAVER 压缩服务未启动，请先运行：
cd /path/to/BEAVER && python beaver_server.py --model_path Qwen/Qwen3-0.6B --port 8765
```

## Important Notes

- Always escape special characters in the JSON payload (newlines, quotes, etc.)
- For very long texts (>100K chars), consider splitting into chunks
- The compression is query-aware: different questions will keep different parts of the text
