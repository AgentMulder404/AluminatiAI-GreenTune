import { NextRequest } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { GoogleGenAI, Type } from "@google/genai";

export const maxDuration = 60;

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

// ─── Energy constants ──────────────────────────────────────────

const MI300X_TDP_W = 750;
const GRID_CO2_KG_PER_KWH = 0.39;
const ENERGY_COST_PER_KWH = 0.12;

interface PolicyResult {
  policy: string;
  limit: number;
  unit: string;
  actual: number;
  passed: boolean;
  headroom_pct: number;
}

// ─── Tool functions (real actions the agents execute) ──────────

function listHistoricalRuns() {
  return {
    runs: [
      {
        name: "Baseline (bs=2, ga=4)",
        batch_size: 2, grad_accum: 4, epochs: 1, lora_rank: 16,
        total_joules: 87300, joules_per_token: 0.355, duration_sec: 138.5,
        co2_grams: 9.46, cost_usd: 0.0024, avg_power_w: 630,
      },
      {
        name: "Small Batch (bs=1, ga=8)",
        batch_size: 1, grad_accum: 8, epochs: 1, lora_rank: 16,
        total_joules: 113834, joules_per_token: 0.463, duration_sec: 178.9,
        co2_grams: 12.33, cost_usd: 0.0032, avg_power_w: 636,
      },
    ],
    count: 2,
  };
}

function projectEnergy(args: {
  batch_size: number;
  gradient_accumulation_steps: number;
  num_epochs: number;
  lora_rank: number;
  max_samples: number;
  model_name?: string;
}) {
  const bs = args.batch_size;
  const ga = args.gradient_accumulation_steps;
  const epochs = args.num_epochs;
  const rank = args.lora_rank;
  const samples = args.max_samples;

  const effectiveBatch = bs * ga;
  const stepsPerEpoch = Math.ceil(samples / effectiveBatch);
  const totalSteps = stepsPerEpoch * epochs;
  const timePerStep = 0.4 + bs * 0.15 + rank / 64;
  const totalSeconds = totalSteps * timePerStep;

  const avgPower = MI300X_TDP_W * 0.85;
  const totalJoules = avgPower * totalSeconds;
  const totalKwh = totalJoules / 3_600_000;
  const totalTokens = samples * 512 * epochs;
  const jpt = totalTokens > 0 ? totalJoules / totalTokens : 0;
  const co2 = totalKwh * GRID_CO2_KG_PER_KWH * 1000;
  const cost = totalKwh * ENERGY_COST_PER_KWH;

  return {
    config: {
      model: args.model_name ?? "Qwen/Qwen2.5-7B",
      batch_size: bs, grad_accum: ga, epochs, lora_rank: rank, max_samples: samples,
    },
    projection: {
      total_steps: totalSteps,
      duration_seconds: +totalSeconds.toFixed(1),
      duration_human: `${(totalSeconds / 60).toFixed(1)} min`,
      avg_power_watts: +avgPower.toFixed(1),
      total_joules: +totalJoules.toFixed(1),
      total_kwh: +totalKwh.toFixed(4),
      joules_per_token: +jpt.toFixed(4),
      total_tokens: totalTokens,
      co2_grams: +co2.toFixed(2),
      cost_usd: +cost.toFixed(4),
    },
  };
}

function checkPolicies(args: {
  total_joules: number;
  joules_per_token: number;
  co2_grams: number;
  cost_usd: number;
}) {
  const policies = [
    { name: "carbon_budget", limit: 50, unit: "g", value: args.co2_grams },
    { name: "energy_cap", limit: 1, unit: "kWh", value: args.total_joules / 3_600_000 },
    { name: "efficiency_floor", limit: 0.8, unit: "J/tok", value: args.joules_per_token },
    { name: "cost_guard", limit: 1.0, unit: "USD", value: args.cost_usd },
  ];
  let allPassed = true;
  const results: PolicyResult[] = policies.map((p) => {
    const passed = p.value <= p.limit;
    if (!passed) allPassed = false;
    return {
      policy: p.name, limit: p.limit, unit: p.unit,
      actual: +p.value.toFixed(4), passed,
      headroom_pct: +(((p.limit - p.value) / p.limit) * 100).toFixed(1),
    };
  });
  return { all_passed: allPassed, policies: results };
}

function compareConfigs(args: { configs: Record<string, unknown>[] }) {
  const sorted = [...args.configs].sort(
    (a, b) =>
      ((a as Record<string, Record<string, number>>).projection?.joules_per_token ?? Infinity) -
      ((b as Record<string, Record<string, number>>).projection?.joules_per_token ?? Infinity)
  );
  const best = sorted[0] as Record<string, Record<string, number>>;
  const worst = sorted[sorted.length - 1] as Record<string, Record<string, number>>;
  const bJpt = best?.projection?.joules_per_token ?? 0;
  const wJpt = worst?.projection?.joules_per_token ?? 1;
  return {
    ranked: sorted,
    best, worst,
    energy_savings_pct: +(((wJpt - bJpt) / wJpt) * 100).toFixed(1),
  };
}

function getHardwareInfo() {
  return {
    gpu: "AMD Instinct MI300X",
    architecture: "CDNA3 (gfx942)",
    vram: "192 GB HBM3",
    tdp_watts: 750,
    memory_bandwidth: "5.3 TB/s",
    compute: "1307.4 TFLOPS (FP16)",
    monitoring: "amdsmi at 0.5s intervals",
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TOOL_FUNCTIONS: Record<string, (args: any) => unknown> = {
  list_historical_runs: () => listHistoricalRuns(),
  project_energy: projectEnergy,
  check_policies: checkPolicies,
  compare_configs: compareConfigs,
  get_hardware_info: () => getHardwareInfo(),
};

const TOOL_DECLARATIONS = [
  {
    name: "list_historical_runs",
    description: "List completed training runs with energy metrics.",
    parameters: { type: Type.OBJECT, properties: {} },
  },
  {
    name: "project_energy",
    description: "Project energy for a proposed config.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        batch_size: { type: Type.NUMBER, description: "Batch size" },
        gradient_accumulation_steps: { type: Type.NUMBER, description: "Grad accum steps" },
        num_epochs: { type: Type.NUMBER, description: "Epochs" },
        lora_rank: { type: Type.NUMBER, description: "LoRA rank" },
        max_samples: { type: Type.NUMBER, description: "Max samples" },
        model_name: { type: Type.STRING, description: "Model name" },
      },
      required: ["batch_size", "gradient_accumulation_steps", "num_epochs", "lora_rank", "max_samples"],
    },
  },
  {
    name: "check_policies",
    description: "Check projected metrics against Lobster Trap policies.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        total_joules: { type: Type.NUMBER, description: "Total Joules" },
        joules_per_token: { type: Type.NUMBER, description: "J/token" },
        co2_grams: { type: Type.NUMBER, description: "CO2 grams" },
        cost_usd: { type: Type.NUMBER, description: "Cost USD" },
      },
      required: ["total_joules", "joules_per_token", "co2_grams", "cost_usd"],
    },
  },
  {
    name: "compare_configs",
    description: "Compare and rank configs by J/token efficiency.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        configs: { type: Type.ARRAY, items: { type: Type.OBJECT }, description: "Projection results" },
      },
      required: ["configs"],
    },
  },
  {
    name: "get_hardware_info",
    description: "Get AMD MI300X GPU specs.",
    parameters: { type: Type.OBJECT, properties: {} },
  },
];

// ─── Agent prompts ─────────────────────────────────────────────

const OPTIMIZER_PROMPT = `You are the Config Optimizer in the GreenTune Agent Swarm.
Propose energy-efficient QLoRA hyperparameter configs for AMD MI300X.
MI300X draws ~750W regardless of batch size — larger batches = fewer steps = less energy.
LoRA rank 16 is optimal for 7B models. Use your tools to project and compare configs.
Always propose 3 configs: conservative, balanced, aggressive. Include specific numbers.`;

const GUARDIAN_PROMPT = `You are the Policy Guardian in the GreenTune Agent Swarm.
Enforce Lobster Trap policies: carbon_budget 50g CO2, energy_cap 1 kWh, efficiency_floor 0.8 J/tok, cost_guard $1.00.
Use check_policies on each config. No exceptions — all policies must pass.
For violations, suggest specific parameter changes.`;

const ANALYST_PROMPT = `You are the Energy Analyst in the GreenTune Agent Swarm.
Analyze energy data and project consumption. Provide absolute numbers, relative comparisons to baseline (0.355 J/tok), and actionable suggestions.
Use tools to list history, project energy, and compare configs.`;

// ─── Gemini agent with function calling ────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyContent = any;

async function agentAct(
  client: InstanceType<typeof GoogleGenAI>,
  systemPrompt: string,
  toolNames: string[],
  history: AnyContent[],
  message: string,
  maxRounds: number = 6,
): Promise<{ text: string; toolCalls: { tool: string; args: unknown; result: unknown }[] }> {
  const filteredDecls = TOOL_DECLARATIONS.filter((d) => toolNames.includes(d.name));
  const tools = toolNames.length
    ? [{ functionDeclarations: filteredDecls }]
    : undefined;

  const contents: AnyContent[] = [
    ...history,
    { role: "user", parts: [{ text: message }] },
  ];

  const toolCalls: { tool: string; args: unknown; result: unknown }[] = [];

  for (let i = 0; i < maxRounds; i++) {
    const resp = await client.models.generateContent({
      model: "gemini-2.5-flash",
      contents: contents as AnyContent,
      config: { systemInstruction: systemPrompt, tools: tools as AnyContent },
    });

    const parts = resp.candidates?.[0]?.content?.parts ?? [];
    contents.push({ role: "model", parts });

    const fcParts = parts.filter(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (p: any) => p.functionCall
    );
    if (fcParts.length === 0) {
      return { text: resp.text ?? "", toolCalls };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fnResponses: any[] = [];
    for (const p of fcParts) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const fc = (p as any).functionCall as { name: string; args: Record<string, unknown> };
      const fn = TOOL_FUNCTIONS[fc.name];
      const result = fn ? fn(fc.args ?? {}) : { error: `Unknown tool: ${fc.name}` };
      toolCalls.push({ tool: fc.name, args: fc.args, result });
      fnResponses.push({ functionResponse: { name: fc.name, response: result } });
    }
    contents.push({ role: "user", parts: fnResponses });
  }

  return { text: "[Max rounds reached]", toolCalls };
}

// ─── SSE Swarm endpoint ────────────────────────────────────────

export async function POST(req: NextRequest) {
  const supabase = await createSupabaseCookieClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user || !ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return new Response(JSON.stringify({ error: "Admin only" }), { status: 403 });
  }

  const body = await req.json();
  const goal = (body.goal as string) || "Find the most energy-efficient QLoRA config for Qwen2.5-7B on 500 Hermes traces";
  const maxIterations = Math.min(Number(body.iterations) || 3, 5);

  const key = process.env.GOOGLE_API_KEY;
  if (!key) {
    return new Response(JSON.stringify({ error: "GOOGLE_API_KEY not configured" }), { status: 500 });
  }

  const client = new GoogleGenAI({ apiKey: key });
  const encoder = new TextEncoder();

  const sharedTools = ["list_historical_runs", "project_energy", "compare_configs", "get_hardware_info"];

  const stream = new ReadableStream({
    async start(controller) {
      function send(event: Record<string, unknown>) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      }

      try {
        send({ type: "swarm_start", agent: "Orchestrator", data: goal });

        // Phase 1 — Analyst reviews history
        send({ type: "phase", agent: "Energy Analyst", data: "Analyzing historical runs" });
        const analysis = await agentAct(client, ANALYST_PROMPT, sharedTools, [],
          "List all historical training runs and analyze energy efficiency. Identify best and worst configs.");
        for (const tc of analysis.toolCalls) {
          send({ type: "tool_call", agent: "Energy Analyst", tool: tc.tool, args: tc.args });
          send({ type: "tool_result", agent: "Energy Analyst", tool: tc.tool, result: tc.result });
        }
        send({ type: "agent_response", agent: "Energy Analyst", data: analysis.text });

        for (let iter = 1; iter <= maxIterations; iter++) {
          send({ type: "iteration", number: iter, total: maxIterations });

          // Phase 2 — Optimizer proposes
          send({ type: "phase", agent: "Config Optimizer", data: `Proposing configs (iteration ${iter})` });
          const proposals = await agentAct(client, OPTIMIZER_PROMPT, sharedTools, [],
            `Goal: ${goal}\nHistory:\n${analysis.text}\n\nPropose 3 QLoRA configs. Use project_energy on each, then compare_configs.`);
          for (const tc of proposals.toolCalls) {
            send({ type: "tool_call", agent: "Config Optimizer", tool: tc.tool, args: tc.args });
            send({ type: "tool_result", agent: "Config Optimizer", tool: tc.tool, result: tc.result });
          }
          send({ type: "agent_response", agent: "Config Optimizer", data: proposals.text });

          // Phase 3 — Guardian checks policies
          send({ type: "phase", agent: "Policy Guardian", data: "Checking Lobster Trap compliance" });
          const guard = await agentAct(client, GUARDIAN_PROMPT, ["check_policies"], [],
            `Check these configs:\n\n${proposals.text}\n\nUse check_policies for each.`);
          for (const tc of guard.toolCalls) {
            send({ type: "tool_call", agent: "Policy Guardian", tool: tc.tool, args: tc.args });
            send({ type: "tool_result", agent: "Policy Guardian", tool: tc.tool, result: tc.result });
          }
          send({ type: "agent_response", agent: "Policy Guardian", data: guard.text });

          // Phase 4 — Analyst evaluates
          send({ type: "phase", agent: "Energy Analyst", data: "Ranking by efficiency" });
          const evaluation = await agentAct(client, ANALYST_PROMPT, sharedTools, [],
            `Proposals:\n${proposals.text}\nPolicy:\n${guard.text}\n\nRank passing configs by J/token. Compare to baseline 0.355 J/tok.`);
          for (const tc of evaluation.toolCalls) {
            send({ type: "tool_call", agent: "Energy Analyst", tool: tc.tool, args: tc.args });
            send({ type: "tool_result", agent: "Energy Analyst", tool: tc.tool, result: tc.result });
          }
          send({ type: "agent_response", agent: "Energy Analyst", data: evaluation.text });
        }

        // Final synthesis
        send({ type: "phase", agent: "Energy Analyst", data: "Producing final recommendation" });
        const final = await agentAct(client, ANALYST_PROMPT, sharedTools, [],
          "Provide your FINAL recommendation as a JSON block with: model, batch_size, grad_accum, epochs, lora_rank, projected_jpt, projected_co2_g, projected_cost_usd, savings_vs_baseline_pct, reasoning.");
        send({ type: "agent_response", agent: "Energy Analyst", data: final.text });

        let recommendation = null;
        const match = final.text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
        if (match) {
          try { recommendation = JSON.parse(match[1]); } catch { /* ignore */ }
        }

        send({ type: "recommendation", config: recommendation, text: final.text });
        send({ type: "swarm_complete", iterations: maxIterations, success: !!recommendation });
      } catch (err) {
        send({ type: "error", data: err instanceof Error ? err.message : "Unknown error" });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
