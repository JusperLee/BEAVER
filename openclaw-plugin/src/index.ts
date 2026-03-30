/**
 * OpenClaw Plugin: BEAVER Prompt Compression
 *
 * Provides:
 *   - /beaver <question> — slash command for quick compress-and-ask
 *   - beaver_compress tool — pure compression
 *   - beaver_ask tool — compress + format prompt
 *
 * Requires the BEAVER API server running (beaver_server.py).
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { Type } from "@sinclair/typebox";

interface CompressResult {
  compressed_text: string;
  original_tokens: number;
  compressed_tokens: number;
  compression_ratio: number;
  speedup: number;
}

async function callBeaverAPI(
  serverUrl: string,
  context: string,
  query: string,
  options?: {
    page_size?: number;
    anchor_pages?: number;
    flow_window?: number;
    flash_top_k?: number;
    semantic_weight?: number;
    lexical_weight?: number;
  }
): Promise<CompressResult> {
  const body = {
    context,
    query,
    page_size: options?.page_size ?? 64,
    anchor_pages: options?.anchor_pages ?? 4,
    flow_window: options?.flow_window ?? 4,
    flash_top_k: options?.flash_top_k ?? 22,
    semantic_weight: options?.semantic_weight ?? 0.7,
    lexical_weight: options?.lexical_weight ?? 0.3,
  };

  const resp = await fetch(`${serverUrl}/compress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`BEAVER API error (${resp.status}): ${errText}`);
  }

  return (await resp.json()) as CompressResult;
}

export default definePluginEntry({
  id: "beaver-compression",
  name: "BEAVER Prompt Compression",
  description:
    "Compress long text using BEAVER. Use /beaver <question> to compress context and ask.",

  register(api) {
    const getConfig = () => {
      const cfg = api.getConfig?.() ?? {};
      return {
        serverUrl: (cfg.serverUrl as string) || "http://127.0.0.1:8765",
        defaultPageSize: (cfg.defaultPageSize as number) || 64,
        defaultTopK: (cfg.defaultTopK as number) || 22,
      };
    };

    // ---- /beaver slash command ----
    api.registerCommand({
      name: "beaver",
      description: "Compress conversation context with BEAVER and answer your question",
      async handler(ctx) {
        const query = ctx.args?.trim();
        if (!query) {
          return ctx.reply(
            "Usage: `/beaver <your question>`\n" +
            "Example: `/beaver 这篇论文的主要贡献是什么？`\n\n" +
            "Make sure there is long text in the conversation first."
          );
        }

        // Gather context from conversation history
        const messages = ctx.conversation?.messages ?? [];
        const contextParts: string[] = [];
        for (const msg of messages) {
          if (msg.role === "user" && msg.content && typeof msg.content === "string") {
            // Skip the /beaver command itself
            if (!msg.content.startsWith("/beaver")) {
              contextParts.push(msg.content);
            }
          }
        }

        const context = contextParts.join("\n\n");
        if (!context.trim()) {
          return ctx.reply(
            "No context found in conversation. Please paste your long text first, then use `/beaver <question>`."
          );
        }

        const cfg = getConfig();
        try {
          await ctx.reply("Compressing with BEAVER...");

          const result = await callBeaverAPI(cfg.serverUrl, context, query, {
            page_size: cfg.defaultPageSize,
            flash_top_k: cfg.defaultTopK,
          });

          const statsLine =
            `📊 ${result.original_tokens} → ${result.compressed_tokens} tokens ` +
            `(${result.speedup}x compression)`;

          // Send compressed context as a new user message for the LLM to answer
          await ctx.sendMessage({
            role: "user",
            content:
              `${statsLine}\n\n` +
              `Context (compressed by BEAVER):\n${result.compressed_text}\n\n` +
              `Question: ${query}`,
          });
        } catch (err: any) {
          return ctx.reply(
            `BEAVER compression failed: ${err.message}\n\n` +
            `Make sure the server is running:\n` +
            `\`python beaver_server.py --model_path Qwen/Qwen3-0.6B --port 8765\``
          );
        }
      },
    });

    // ---- beaver_compress tool ----
    api.registerTool({
      name: "beaver_compress",
      description:
        "Compress long text using BEAVER hierarchical prompt compression. " +
        "Returns compressed text + stats. Achieves up to 26x compression.",
      parameters: Type.Object({
        context: Type.String({ description: "The long text to compress" }),
        query: Type.String({ description: "Query guiding which content to keep" }),
        page_size: Type.Optional(Type.Number({ description: "Tokens per page (16-256)" })),
        flash_top_k: Type.Optional(Type.Number({ description: "Top pages to keep (1-64)" })),
      }),
      async execute(_id, params) {
        const cfg = getConfig();
        try {
          const result = await callBeaverAPI(cfg.serverUrl, params.context, params.query, {
            page_size: params.page_size ?? cfg.defaultPageSize,
            flash_top_k: params.flash_top_k ?? cfg.defaultTopK,
          });

          const summary =
            `[BEAVER] ${result.original_tokens} → ${result.compressed_tokens} tokens ` +
            `(${result.speedup}x)\n\n${result.compressed_text}`;

          return { content: [{ type: "text", text: summary }] };
        } catch (err: any) {
          return {
            content: [{ type: "text", text: `[BEAVER Error] ${err.message}` }],
          };
        }
      },
    });

    // ---- beaver_ask tool ----
    api.registerTool({
      name: "beaver_ask",
      description:
        "Compress context with BEAVER and return a formatted prompt for the LLM to answer.",
      parameters: Type.Object({
        context: Type.String({ description: "The long text/document" }),
        query: Type.String({ description: "The question to answer" }),
        page_size: Type.Optional(Type.Number()),
        flash_top_k: Type.Optional(Type.Number()),
      }),
      async execute(_id, params) {
        const cfg = getConfig();
        try {
          const result = await callBeaverAPI(cfg.serverUrl, params.context, params.query, {
            page_size: params.page_size ?? cfg.defaultPageSize,
            flash_top_k: params.flash_top_k ?? cfg.defaultTopK,
          });

          const prompt =
            `Context (compressed from ${result.original_tokens} to ${result.compressed_tokens} tokens, ` +
            `${result.speedup}x):\n${result.compressed_text}\n\nQuestion: ${params.query}\n\nAnswer:`;

          return { content: [{ type: "text", text: prompt }] };
        } catch (err: any) {
          return {
            content: [{ type: "text", text: `[BEAVER Error] ${err.message}` }],
          };
        }
      },
    });
  },
});
