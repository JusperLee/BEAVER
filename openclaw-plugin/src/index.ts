/**
 * OpenClaw Plugin: BEAVER Prompt Compression
 *
 * Provides:
 *   - /beaver <question> — slash command for quick compress-and-ask
 *   - beaver_compress tool — pure compression
 *   - beaver_ask tool — compress + format prompt
 *
 * All compression parameters are auto-detected by the server based on text length.
 * Requires the BEAVER API server running (beaver_server.py).
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

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
): Promise<CompressResult> {
  const resp = await fetch(`${serverUrl}/compress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, query }),
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
    const getServerUrl = () => {
      const cfg = api.getConfig?.() ?? {};
      return (cfg.serverUrl as string) || "http://127.0.0.1:8765";
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
            "Example: `/beaver What are the main contributions?`\n\n" +
            "Make sure there is long text in the conversation first."
          );
        }

        const messages = ctx.conversation?.messages ?? [];
        const contextParts: string[] = [];
        for (const msg of messages) {
          if (msg.role === "user" && msg.content && typeof msg.content === "string") {
            if (!msg.content.startsWith("/beaver")) {
              contextParts.push(msg.content);
            }
          }
        }

        const context = contextParts.join("\n\n");
        if (!context.trim()) {
          return ctx.reply(
            "No context found. Please paste your long text first, then use `/beaver <question>`."
          );
        }

        try {
          await ctx.reply("Compressing with BEAVER...");
          const result = await callBeaverAPI(getServerUrl(), context, query);

          await ctx.sendMessage({
            role: "user",
            content:
              `[BEAVER] ${result.original_tokens} -> ${result.compressed_tokens} tokens ` +
              `(${result.speedup}x)\n\n` +
              `Context (compressed):\n${result.compressed_text}\n\n` +
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
        "Returns compressed text + stats. Parameters are auto-detected by the server.",
      parameters: {
        type: "object",
        required: ["context", "query"],
        properties: {
          context: { type: "string", description: "The long text to compress" },
          query: { type: "string", description: "Query guiding which content to keep" },
        },
      },
      async execute(_id: any, params: any) {
        try {
          const result = await callBeaverAPI(getServerUrl(), params.context, params.query);
          return {
            content: [{
              type: "text",
              text: `[BEAVER] ${result.original_tokens} -> ${result.compressed_tokens} tokens ` +
                `(${result.speedup}x)\n\n${result.compressed_text}`,
            }],
          };
        } catch (err: any) {
          return { content: [{ type: "text", text: `[BEAVER Error] ${err.message}` }] };
        }
      },
    });

    // ---- beaver_ask tool ----
    api.registerTool({
      name: "beaver_ask",
      description:
        "Compress context with BEAVER and return a formatted prompt for the LLM to answer.",
      parameters: {
        type: "object",
        required: ["context", "query"],
        properties: {
          context: { type: "string", description: "The long text/document" },
          query: { type: "string", description: "The question to answer" },
        },
      },
      async execute(_id: any, params: any) {
        try {
          const result = await callBeaverAPI(getServerUrl(), params.context, params.query);
          return {
            content: [{
              type: "text",
              text: `Context (compressed from ${result.original_tokens} to ${result.compressed_tokens} tokens, ` +
                `${result.speedup}x):\n${result.compressed_text}\n\nQuestion: ${params.query}\n\nAnswer:`,
            }],
          };
        } catch (err: any) {
          return { content: [{ type: "text", text: `[BEAVER Error] ${err.message}` }] };
        }
      },
    });
  },
});
