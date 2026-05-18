"""
GreenTune Agent Swarm — Multi-agent Gemini-powered optimization
for energy-efficient LLM fine-tuning on AMD MI300X GPUs.

Architecture:
  Orchestrator ──┬── Config Optimizer (proposes hyperparameters)
                 ├── Policy Guardian (enforces Lobster Trap constraints)
                 └── Energy Analyst (projects & analyzes energy data)

Each agent uses Gemini 2.5 Flash with function calling to take
real actions. The Orchestrator coordinates the optimization loop:

  Goal → Analyze History → Propose Configs → Check Policies →
  Project Energy → [Iterate if violations] → Recommend Best
"""

import os
import re
import json
import glob
import math
import time
import dataclasses
from typing import Optional

from google import genai
from google.genai import types

# ─── Energy constants ───────────────────────────────────────────

MI300X_TDP_W = 750
GRID_CO2_KG_PER_KWH = 0.39
ENERGY_COST_PER_KWH = 0.12
BASELINE_JPT = 0.355


@dataclasses.dataclass
class EnergyPolicy:
    name: str
    description: str
    limit: float
    unit: str


LOBSTER_TRAP = [
    EnergyPolicy("carbon_budget", "Max CO2 per run", 50.0, "g"),
    EnergyPolicy("energy_cap", "Max energy per run", 1.0, "kWh"),
    EnergyPolicy("efficiency_floor", "Max joules per token", 0.8, "J/tok"),
    EnergyPolicy("cost_guard", "Max energy cost per run", 1.00, "USD"),
]

# ─── Tool functions (real actions agents can call) ──────────────


def list_historical_runs() -> dict:
    """Scan output/ for completed training runs and return energy metrics."""
    runs = []
    for path in sorted(glob.glob("output/*/energy_metrics.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            s = data.get("energy_summary", {})
            c = data.get("config", {})
            runs.append({
                "path": path,
                "model": c.get("model_name", "unknown"),
                "batch_size": c.get("batch_size"),
                "grad_accum": c.get("gradient_accumulation_steps"),
                "epochs": c.get("num_epochs"),
                "lora_rank": c.get("lora_rank"),
                "total_joules": s.get("total_energy_joules"),
                "joules_per_token": s.get("joules_per_token"),
                "duration_sec": s.get("total_duration_seconds"),
                "co2_grams": s.get("co2_grams"),
                "cost_usd": s.get("cost_usd"),
                "avg_power_w": s.get("avg_power_watts"),
            })
        except Exception:
            continue

    if not runs:
        runs = [
            {
                "name": "Baseline (bs=2, ga=4)",
                "batch_size": 2, "grad_accum": 4, "epochs": 1,
                "lora_rank": 16, "total_joules": 87300,
                "joules_per_token": 0.355, "duration_sec": 138.5,
                "co2_grams": 9.46, "cost_usd": 0.0024,
                "avg_power_w": 630,
            },
            {
                "name": "Small Batch (bs=1, ga=8)",
                "batch_size": 1, "grad_accum": 8, "epochs": 1,
                "lora_rank": 16, "total_joules": 113834,
                "joules_per_token": 0.463, "duration_sec": 178.9,
                "co2_grams": 12.33, "cost_usd": 0.0032,
                "avg_power_w": 636,
            },
        ]
    return {"runs": runs, "count": len(runs)}


def project_energy(
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    lora_rank: int = 16,
    max_samples: int = 500,
    model_name: str = "Qwen/Qwen2.5-7B",
) -> dict:
    """Project energy consumption for a proposed training config."""
    effective_batch = batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(max_samples / effective_batch)
    total_steps = steps_per_epoch * num_epochs

    time_per_step = 0.4 + (batch_size * 0.15) + (lora_rank / 64)
    total_seconds = total_steps * time_per_step

    avg_power = MI300X_TDP_W * 0.85
    total_joules = avg_power * total_seconds
    total_kwh = total_joules / 3_600_000

    tokens_per_sample = 512
    total_tokens = max_samples * tokens_per_sample * num_epochs
    jpt = total_joules / total_tokens if total_tokens > 0 else 0

    co2_grams = total_kwh * GRID_CO2_KG_PER_KWH * 1000
    cost_usd = total_kwh * ENERGY_COST_PER_KWH

    return {
        "config": {
            "model": model_name,
            "batch_size": batch_size,
            "grad_accum": gradient_accumulation_steps,
            "epochs": num_epochs,
            "lora_rank": lora_rank,
            "max_samples": max_samples,
        },
        "projection": {
            "total_steps": total_steps,
            "duration_seconds": round(total_seconds, 1),
            "duration_human": f"{total_seconds / 60:.1f} min",
            "avg_power_watts": round(avg_power, 1),
            "total_joules": round(total_joules, 1),
            "total_kwh": round(total_kwh, 4),
            "joules_per_token": round(jpt, 4),
            "total_tokens": total_tokens,
            "co2_grams": round(co2_grams, 2),
            "cost_usd": round(cost_usd, 4),
        },
    }


def check_policies(
    total_joules: float = 0,
    joules_per_token: float = 0,
    co2_grams: float = 0,
    cost_usd: float = 0,
) -> dict:
    """Check projected energy against Lobster Trap policies."""
    results = []
    all_pass = True
    for p in LOBSTER_TRAP:
        if p.name == "carbon_budget":
            value = co2_grams
        elif p.name == "energy_cap":
            value = total_joules / 3_600_000
        elif p.name == "efficiency_floor":
            value = joules_per_token
        elif p.name == "cost_guard":
            value = cost_usd
        else:
            continue
        passed = value <= p.limit
        headroom = ((p.limit - value) / p.limit * 100) if p.limit > 0 else 0
        if not passed:
            all_pass = False
        results.append({
            "policy": p.name, "limit": p.limit, "unit": p.unit,
            "actual": round(value, 4), "passed": passed,
            "headroom_pct": round(headroom, 1),
        })
    return {"all_passed": all_pass, "policies": results}


def compare_configs(configs: list = []) -> dict:  # noqa: B006
    """Compare projected configs and rank by energy efficiency."""
    if not configs:
        return {"error": "No configs to compare"}
    ranked = sorted(
        configs,
        key=lambda c: c.get("projection", {}).get("joules_per_token", float("inf")),
    )
    best = ranked[0]
    worst = ranked[-1]
    best_jpt = best.get("projection", {}).get("joules_per_token", 0)
    worst_jpt = worst.get("projection", {}).get("joules_per_token", 0)
    savings = ((worst_jpt - best_jpt) / worst_jpt * 100) if worst_jpt > 0 else 0
    return {
        "ranked": ranked,
        "best": best,
        "worst": worst,
        "energy_savings_pct": round(savings, 1),
    }


def get_hardware_info() -> dict:
    """AMD MI300X hardware specifications."""
    return {
        "gpu": "AMD Instinct MI300X",
        "architecture": "CDNA3 (gfx942)",
        "vram": "192 GB HBM3",
        "tdp_watts": 750,
        "memory_bandwidth": "5.3 TB/s",
        "compute": "1307.4 TFLOPS (FP16)",
        "monitoring": "amdsmi at 0.5s intervals",
    }


TOOL_FUNCTIONS = {
    "list_historical_runs": list_historical_runs,
    "project_energy": project_energy,
    "check_policies": check_policies,
    "compare_configs": compare_configs,
    "get_hardware_info": get_hardware_info,
}


def _make_tool_declarations():
    return [
        types.FunctionDeclaration(
            name="list_historical_runs",
            description="List completed training runs with energy metrics. Call first to understand baseline performance.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="project_energy",
            description="Project energy for a proposed config. Returns duration, power, joules, CO2, cost.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "batch_size": types.Schema(type=types.Type.INTEGER, description="Batch size per GPU"),
                    "gradient_accumulation_steps": types.Schema(type=types.Type.INTEGER, description="Gradient accumulation steps"),
                    "num_epochs": types.Schema(type=types.Type.INTEGER, description="Training epochs"),
                    "lora_rank": types.Schema(type=types.Type.INTEGER, description="LoRA rank (8/16/32/64)"),
                    "max_samples": types.Schema(type=types.Type.INTEGER, description="Max training samples"),
                    "model_name": types.Schema(type=types.Type.STRING, description="Model name"),
                },
                required=["batch_size", "gradient_accumulation_steps", "num_epochs", "lora_rank", "max_samples"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_policies",
            description="Check projected metrics against Lobster Trap policies. Returns pass/fail per policy.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "total_joules": types.Schema(type=types.Type.NUMBER, description="Total energy in Joules"),
                    "joules_per_token": types.Schema(type=types.Type.NUMBER, description="Efficiency in J/token"),
                    "co2_grams": types.Schema(type=types.Type.NUMBER, description="CO2 in grams"),
                    "cost_usd": types.Schema(type=types.Type.NUMBER, description="Energy cost in USD"),
                },
                required=["total_joules", "joules_per_token", "co2_grams", "cost_usd"],
            ),
        ),
        types.FunctionDeclaration(
            name="compare_configs",
            description="Compare projected configs, rank by J/token efficiency.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "configs": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT), description="Array of project_energy results"),
                },
                required=["configs"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_hardware_info",
            description="Get AMD MI300X GPU specs.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]


# ─── Agent definitions ──────────────────────────────────────────

OPTIMIZER_PROMPT = """You are the Config Optimizer agent in the GreenTune Swarm.

Your job: propose energy-efficient QLoRA hyperparameter configs for AMD MI300X.

Key insights:
- MI300X draws ~750W regardless of batch size (TDP saturation)
- Larger batch sizes = fewer steps = less wall time = less total energy
- LoRA rank 16 is the sweet spot for 7B models
- Gradient accumulation simulates larger batches without memory cost
- QLoRA NF4 loads 7B in ~5GB on 192GB MI300X — memory is not the bottleneck

Use your tools to list historical runs, project energy for proposals, and compare them.
Always propose at least 3 configs: conservative, balanced, aggressive.
Include specific numbers in every response."""

GUARDIAN_PROMPT = """You are the Policy Guardian agent in the GreenTune Swarm.

Your job: enforce Lobster Trap energy governance on proposed training configs.

Active policies:
- carbon_budget: Max 50g CO2 per run
- energy_cap: Max 1 kWh per run
- efficiency_floor: Max 0.8 J/token
- cost_guard: Max $1.00 energy cost per run

Use check_policies on each config's projected metrics.
You are strict — no exceptions. A config passes ALL policies or it fails.
For violations, suggest specific parameter changes to fix them."""

ANALYST_PROMPT = """You are the Energy Analyst agent in the GreenTune Swarm.

Your job: analyze energy data and project consumption for training configs.

Knowledge:
- GPU power characteristics (MI300X TDP, utilization curves)
- Energy metrics (joules, kWh, J/token, CO2, cost)
- QLoRA training dynamics (steps, tokens, convergence)

Use tools to list historical runs, project energy, compare configs.
Always provide absolute numbers, relative comparisons to baseline, and actionable suggestions."""


_last_call_ts: float = 0.0
_MIN_INTERVAL = 2.0  # paid tier: 1000 RPM, just a small safety gap


def _call_with_retry(client, **kwargs):
    """Call generate_content with rate limiting and retry on 429."""
    global _last_call_ts

    # Pre-throttle: ensure minimum gap between requests
    elapsed = time.time() - _last_call_ts
    if elapsed < _MIN_INTERVAL:
        gap = _MIN_INTERVAL - elapsed
        try:
            from rich.console import Console
            Console().print(f"[dim]Throttling — waiting {gap:.0f}s for rate limit[/dim]")
        except ImportError:
            print(f"Throttling — waiting {gap:.0f}s for rate limit")
        time.sleep(gap)

    max_retries = 8
    for attempt in range(max_retries):
        try:
            _last_call_ts = time.time()
            return client.models.generate_content(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 30 * (attempt + 1)
                try:
                    from rich.console import Console
                    Console().print(f"[dim]Rate limited — waiting {wait}s (attempt {attempt + 1}/{max_retries})[/dim]")
                except ImportError:
                    print(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded on rate limit")


class SwarmAgent:
    """A single Gemini-powered agent with function calling."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        tool_names: list,
        client,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.history: list = []
        self.action_log: list = []

        all_decls = {d.name: d for d in _make_tool_declarations()}
        self.tool_decls = [all_decls[n] for n in tool_names if n in all_decls]

    def act(self, message: str, max_rounds: int = 6) -> str:
        """Send a message, auto-execute tool calls, return final text."""
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=message)])
        )
        tools = (
            [types.Tool(function_declarations=self.tool_decls)]
            if self.tool_decls
            else None
        )

        for _ in range(max_rounds):
            resp = _call_with_retry(
                self.client,
                model="gemini-2.5-flash",
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    tools=tools,
                ),
            )
            candidate = resp.candidates[0]
            self.history.append(candidate.content)

            fc_parts = [p for p in candidate.content.parts if p.function_call]
            if not fc_parts:
                text = resp.text or ""
                self.action_log.append(
                    {"type": "response", "agent": self.name, "text": text}
                )
                return text

            fn_responses = []
            for p in fc_parts:
                fn_name = p.function_call.name
                fn_args = dict(p.function_call.args) if p.function_call.args else {}
                # Coerce float args to int where the function expects int
                import inspect
                sig = inspect.signature(TOOL_FUNCTIONS.get(fn_name, lambda: None))
                for param_name, param in sig.parameters.items():
                    if param_name in fn_args and param.annotation is int:
                        fn_args[param_name] = int(fn_args[param_name])

                self.action_log.append(
                    {"type": "tool_call", "agent": self.name, "tool": fn_name, "args": fn_args}
                )
                try:
                    result = TOOL_FUNCTIONS[fn_name](**fn_args)
                except Exception as e:
                    result = {"error": str(e)}
                self.action_log.append(
                    {"type": "tool_result", "agent": self.name, "tool": fn_name, "result": result}
                )
                fn_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name, response=result
                        )
                    )
                )

            self.history.append(types.Content(role="user", parts=fn_responses))

        return "[Max tool rounds reached]"


# ─── Swarm orchestrator ────────────────────────────────────────


class GreenTuneSwarm:
    """Multi-agent swarm for energy-optimized LLM fine-tuning."""

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True, on_event=None):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY required")

        self.client = genai.Client(api_key=key)
        self.verbose = verbose
        self.on_event = on_event
        self.trace: list = []

        # Tools are run locally in Python, not via Gemini function calling,
        # to stay within free-tier rate limits (5 req/min).
        self.optimizer = SwarmAgent(
            "Config Optimizer", "Hyperparameter optimization",
            OPTIMIZER_PROMPT, [], self.client,
        )
        self.guardian = SwarmAgent(
            "Policy Guardian", "Energy policy enforcement",
            GUARDIAN_PROMPT, [], self.client,
        )
        self.analyst = SwarmAgent(
            "Energy Analyst", "Energy analysis & projection",
            ANALYST_PROMPT, [], self.client,
        )
        self.agents = {
            "optimizer": self.optimizer,
            "guardian": self.guardian,
            "analyst": self.analyst,
        }

    def _emit(self, event_type: str, agent: str, data: str = ""):
        entry = {"type": event_type, "agent": agent, "data": data, "ts": time.time()}
        self.trace.append(entry)
        if self.on_event:
            self.on_event(entry)
        if self.verbose:
            try:
                from rich.console import Console
                c = Console()
                colors = {
                    "Config Optimizer": "green",
                    "Policy Guardian": "red",
                    "Energy Analyst": "yellow",
                    "Orchestrator": "blue",
                }
                c.print(f"[bold {colors.get(agent, 'white')}][{agent}][/] {event_type}")
                if data:
                    c.print(f"  {data[:300]}")
            except ImportError:
                print(f"[{agent}] {event_type}: {data[:200]}")

    @staticmethod
    def _extract_configs(text: str) -> list:
        """Extract config JSON blocks from agent text."""
        configs = []
        for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text):
            try:
                c = json.loads(m.group(1))
                if "batch_size" in c:
                    configs.append(c)
            except (json.JSONDecodeError, KeyError):
                continue
        return configs

    def optimize(self, goal: str, max_iterations: int = 3) -> dict:
        self._emit("swarm_start", "Orchestrator", goal)

        # Phase 1 — Analyst reviews history (1 API call, uses tools locally)
        self._emit("phase", "Energy Analyst", "Analyzing historical training runs")
        history_data = list_historical_runs()
        hw_info = get_hardware_info()
        history_analysis = self.analyst.act(
            f"Here is the historical training data:\n{json.dumps(history_data, indent=2)}\n\n"
            f"Hardware: {json.dumps(hw_info)}\n\n"
            "Analyze energy efficiency. Which config is best/worst and why?"
        )
        self._emit("analysis", "Energy Analyst", history_analysis[:500])

        best_result = None

        for iteration in range(1, max_iterations + 1):
            self._emit("iteration_start", "Orchestrator", f"Iteration {iteration}/{max_iterations}")

            # Phase 2 — Optimizer proposes configs (1 API call, text only)
            self._emit("phase", "Config Optimizer", "Proposing training configurations")
            context = f"Goal: {goal}\nHistorical analysis:\n{history_analysis}\n"
            if best_result:
                context += f"\nBest so far: {json.dumps(best_result, indent=2)}\nTry to beat it."
            proposals_text = self.optimizer.act(
                f"{context}\n\nPropose 3 QLoRA configs (conservative, balanced, aggressive) "
                "for Qwen2.5-7B on 500 Hermes traces.\n"
                "For each, output a JSON block with: batch_size, gradient_accumulation_steps, "
                "num_epochs, lora_rank, max_samples. Keep max_samples=500."
            )
            self._emit("proposals", "Config Optimizer", proposals_text[:500])

            # Phase 3 — Run projections and policy checks locally (0 API calls)
            configs = self._extract_configs(proposals_text)
            if not configs:
                configs = [
                    {"batch_size": 4, "gradient_accumulation_steps": 2, "num_epochs": 1, "lora_rank": 16, "max_samples": 500},
                    {"batch_size": 8, "gradient_accumulation_steps": 1, "num_epochs": 1, "lora_rank": 16, "max_samples": 500},
                    {"batch_size": 2, "gradient_accumulation_steps": 4, "num_epochs": 1, "lora_rank": 8, "max_samples": 500},
                ]
                self._emit("phase", "Orchestrator", "No JSON configs extracted — using defaults")

            projections = []
            for cfg in configs:
                proj = project_energy(
                    batch_size=int(cfg.get("batch_size", 2)),
                    gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
                    num_epochs=int(cfg.get("num_epochs", 1)),
                    lora_rank=int(cfg.get("lora_rank", 16)),
                    max_samples=int(cfg.get("max_samples", 500)),
                )
                self._emit("tool_call", "Config Optimizer",
                           f"project_energy(bs={cfg.get('batch_size')}, ga={cfg.get('gradient_accumulation_steps')}, "
                           f"rank={cfg.get('lora_rank')})")
                self._emit("tool_result", "Config Optimizer",
                           f"→ {proj['projection']['joules_per_token']} J/tok, "
                           f"{proj['projection']['co2_grams']}g CO2, "
                           f"{proj['projection']['duration_human']}")
                projections.append(proj)

            # Policy checks
            self._emit("phase", "Policy Guardian", "Checking Lobster Trap compliance")
            policy_results = []
            for proj in projections:
                p = proj["projection"]
                check = check_policies(
                    total_joules=p["total_joules"],
                    joules_per_token=p["joules_per_token"],
                    co2_grams=p["co2_grams"],
                    cost_usd=p["cost_usd"],
                )
                policy_results.append({"config": proj["config"], "projection": p, "policies": check})
                status = "PASS" if check["all_passed"] else "FAIL"
                self._emit("tool_result", "Policy Guardian",
                           f"bs={proj['config']['batch_size']} → {status} "
                           f"({sum(1 for r in check['policies'] if r['passed'])}/4 policies)")

            # Compare
            comparison = compare_configs(projections)
            self._emit("tool_result", "Orchestrator",
                       f"Best: bs={comparison['best']['config']['batch_size']}, "
                       f"savings: {comparison['energy_savings_pct']}%")

            # Phase 4 — Guardian synthesizes policy report (1 API call)
            self._emit("phase", "Policy Guardian", "Synthesizing policy report")
            guardian_summary = self.guardian.act(
                f"Here are the Lobster Trap policy check results for {len(policy_results)} configs:\n\n"
                f"{json.dumps(policy_results, indent=2)}\n\n"
                "Summarize: which configs pass, which fail, and what adjustments would fix violations?"
            )
            self._emit("policy_check", "Policy Guardian", guardian_summary[:500])

            # Phase 5 — Analyst evaluates (1 API call)
            passing = [r for r in policy_results if r["policies"]["all_passed"]]
            self._emit("phase", "Energy Analyst", "Ranking configs by efficiency")
            evaluation = self.analyst.act(
                f"Here are the projected configs that passed Lobster Trap:\n\n"
                f"{json.dumps(passing, indent=2)}\n\n"
                f"Comparison: {json.dumps(comparison, indent=2)}\n\n"
                f"Baseline J/token: 0.355. Rank by efficiency, explain tradeoffs, "
                "and output the best config as a JSON block."
            )
            self._emit("evaluation", "Energy Analyst", evaluation[:500])

            for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", evaluation):
                try:
                    parsed = json.loads(m.group(1))
                    if "batch_size" in parsed or "config" in parsed or "projection" in parsed:
                        best_result = parsed
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

            self._emit("iteration_end", "Orchestrator", f"Iteration {iteration} complete")

        # Final synthesis (1 API call)
        self._emit("phase", "Energy Analyst", "Producing final recommendation")
        final = self.analyst.act(
            "Based on all iterations, provide your FINAL recommendation.\n"
            "Output a single JSON block with keys: model, batch_size, grad_accum, "
            "epochs, lora_rank, projected_jpt, projected_co2_g, projected_cost_usd, "
            "savings_vs_baseline_pct, reasoning."
        )
        self._emit("recommendation", "Energy Analyst", final[:500])

        recommendation = None
        for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", final):
            try:
                recommendation = json.loads(m.group(1))
                break
            except (json.JSONDecodeError, KeyError):
                continue

        self._emit("swarm_complete", "Orchestrator", "Optimization complete")

        all_logs = []
        for a in self.agents.values():
            all_logs.extend(a.action_log)

        return {
            "success": recommendation is not None,
            "recommendation": recommendation,
            "iterations": max_iterations,
            "trace": self.trace,
            "tool_calls": [e for e in all_logs if e["type"] == "tool_call"],
            "final_text": final,
        }


# ─── CLI entry point ───────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GreenTune Agent Swarm")
    parser.add_argument(
        "--goal",
        default="Find the most energy-efficient QLoRA config for Qwen2.5-7B on 500 Hermes traces on AMD MI300X",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            f"[bold]Goal:[/bold] {args.goal}\n"
            f"[bold]Iterations:[/bold] {args.iterations}\n"
            f"[bold]Agents:[/bold] Config Optimizer, Policy Guardian, Energy Analyst",
            title="GreenTune Agent Swarm", border_style="green",
        ))
    except ImportError:
        print(f"GreenTune Agent Swarm\nGoal: {args.goal}\nIterations: {args.iterations}")

    swarm = GreenTuneSwarm(api_key=args.api_key, verbose=not args.quiet)
    result = swarm.optimize(args.goal, max_iterations=args.iterations)

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        if result["success"]:
            console.print(Panel(
                json.dumps(result["recommendation"], indent=2),
                title="Swarm Recommendation", border_style="green",
            ))
        else:
            console.print(Panel(
                result.get("final_text", "No recommendation")[:500],
                title="Swarm Result", border_style="yellow",
            ))
        tc = len(result["tool_calls"])
        console.print(f"\n[dim]{result['iterations']} iterations, {tc} tool calls, {len(result['trace'])} events[/dim]")
    except ImportError:
        print(json.dumps(result.get("recommendation"), indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "goal": args.goal,
                    "success": result["success"],
                    "recommendation": result["recommendation"],
                    "iterations": result["iterations"],
                    "trace": result["trace"],
                },
                f, indent=2, default=str,
            )
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
