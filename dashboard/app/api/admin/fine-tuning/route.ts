import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import Anthropic from "@anthropic-ai/sdk";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const maxDuration = 60;

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

function getClient() {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY not configured");
  }
  return new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}

async function callClaude(
  system: string,
  userMessage: string,
  maxTokens = 4096
): Promise<string> {
  const client = getClient();
  const message = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: maxTokens,
    system,
    messages: [{ role: "user", content: userMessage }],
  });
  const block = message.content[0];
  if (block.type === "text") return block.text;
  throw new Error("Unexpected response format");
}

const GREENTUNE_SYSTEM = `You are GreenTune AI, an expert assistant specializing in GPU infrastructure operations, energy-efficient ML training, and AMD ROCm.

Your domain knowledge includes:
- GPU power management (TDP, power capping, thermal throttling)
- Energy metrics: Joules-per-token, kWh, CO2 emissions from training
- AMD MI300X: 192GB HBM3, 750W TDP, CDNA3 architecture, gfx942
- ROCm tools: rocm-smi, amd-smi, amdsmi Python bindings
- QLoRA/LoRA fine-tuning: rank selection, NF4 quantization, PEFT, TRL
- Cost attribution for multi-tenant GPU clusters
- AluminatiAI platform: agent monitoring, energy benchmarks, chargeback

Always include specific numbers (watts, joules, dollars, grams CO2) in your answers.
Be concise and practical. Reference exact commands and configurations.`;

async function handlePlaygroundPrompt(body: Record<string, unknown>) {
  const { prompt } = body;
  if (!prompt || typeof prompt !== "string") {
    return NextResponse.json({ error: "prompt is required" }, { status: 400 });
  }

  const response = await callClaude(GREENTUNE_SYSTEM, prompt, 2048);
  return NextResponse.json({ response });
}

async function handleAnalyzeRun(body: Record<string, unknown>) {
  const { summary, config } = body;
  if (!summary) {
    return NextResponse.json(
      { error: "summary is required" },
      { status: 400 }
    );
  }

  const prompt = `Analyze this GreenTune fine-tuning run and provide insights:

Run Configuration:
${JSON.stringify(config, null, 2)}

Energy Summary:
${JSON.stringify(summary, null, 2)}

Provide:
1. Overall efficiency assessment (is the power draw healthy for this GPU/workload?)
2. Joules-per-token analysis (how does this compare to typical QLoRA workloads?)
3. Cost breakdown (energy vs compute cost ratio)
4. Carbon footprint context (what does this CO2 amount compare to in everyday terms?)
5. Optimization suggestions (could batch size, rank, or precision changes improve efficiency?)

Be specific with numbers and actionable recommendations.`;

  const analysis = await callClaude(GREENTUNE_SYSTEM, prompt);
  return NextResponse.json({ analysis });
}

export async function POST(req: NextRequest) {
  try {
    const supabase = await createSupabaseCookieClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
      return NextResponse.json({ error: "Admin only" }, { status: 403 });
    }

    const rl = await rateLimit(`admin:${user.id}`, 30);
    if (!rl.success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429, headers: getRateLimitHeaders(rl) }
      );
    }

    const body = (await req.json()) as Record<string, unknown>;
    const action = body.action as string;

    switch (action) {
      case "playground_prompt":
        return handlePlaygroundPrompt(body);
      case "analyze_run":
        return handleAnalyzeRun(body);
      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (e) {
    console.error("[fine-tuning API]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Internal error" },
      { status: 500 }
    );
  }
}
