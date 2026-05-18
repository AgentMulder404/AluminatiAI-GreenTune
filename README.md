# AluminatiAI GreenTune Agent

**Autonomous Energy Intelligence for LLM Fine-Tuning — powered by Gemini on AMD MI300X.**

> lablab.ai Transforming Enterprise Through AI Hackathon — San Jose 2026  
> Team: AluminatiAI (Kevin Mello)  
> Live Dashboard: [aluminatiai.com/admin/fine-tuning](https://www.aluminatiai.com/admin/fine-tuning)

---

## The Problem

Every organization fine-tuning LLMs today is flying blind on energy. Teams track loss curves, learning rates, and throughput — but nobody measures the **energy cost per token produced**.

- GPU energy is invisible. MI300X draws 750W at peak but teams never see it.
- Optimization is guesswork. Without per-token energy metrics, you can't compare configs on efficiency.
- Carbon compliance is coming. EU AI Act and SEC climate disclosures will require energy reporting for AI workloads.
- No governance layer. There's no way to enforce energy budgets or reject wasteful jobs before they run.

## The Solution

GreenTune Agent is an **autonomous AI agent powered by Gemini** that turns energy measurement into energy optimization.

Instead of manually tuning hyperparameters, engineers talk to the agent:

```
greentune> Optimize Qwen-7B for lowest J/token under 50g CO2

  Based on historical data, I recommend:
  - batch_size=2, grad_accum=4 (effective=8)
  - LoRA rank 16, lr=2e-4
  - This achieves 0.355 J/token vs 0.463 J/token with bs=1

  Energy Projection:
    Duration: 138s | Energy: 87,300 J | CO2: 9.46g | Cost: $0.0024
    All policies passed. ✓

  Launch training? [y/N]
```

The agent analyzes historical energy data, recommends optimal configs, enforces **energy governance policies**, and monitors training in real-time — all autonomously.

GreenTune treats **Joules-per-token** as a first-class metric alongside loss and throughput.

```
┌──────────────────────────────────────────────────────┐
│  GreenTune Agent (Gemini 2.5)                        │
│  ┌────────────────────────────────────────────────┐  │
│  │  greentune_agent.py                            │  │
│  │  • Natural language → training config          │  │
│  │  • Historical energy analysis                  │  │
│  │  • Energy projection + policy enforcement      │  │
│  │  • Autonomous training launch + monitoring     │  │
│  └────────────────────────┬───────────────────────┘  │
│                           │ recommend / launch        │
│  ┌────────────────────────┴───────────────────────┐  │
│  │  Lobster Trap — Energy Governance              │  │
│  │  • Carbon budget (max 50g CO2)                 │  │
│  │  • Energy cap (max 1 kWh)                      │  │
│  │  • Efficiency floor (max J/token)              │  │
│  │  • Cost guard (max $/run)                      │  │
│  └────────────────────────┬───────────────────────┘  │
└───────────────────────────┼──────────────────────────┘
                            │ approved config
┌───────────────────────────┼──────────────────────────┐
│  Dashboard (Next.js + Supabase + SSE)                │
│  ┌────────────────────────┴───────────────────────┐  │
│  │  GreenTune Dashboard (/admin/fine-tuning)      │  │
│  │  • Overview — hero finding + power overlay     │  │
│  │  • Run Monitor — live power curves, loss       │  │
│  │  • Leaderboard — side-by-side run comparison   │  │
│  │  • ROI Calculator — cost projections           │  │
│  │  • Playground — model evaluation               │  │
│  └────────────────────────┬───────────────────────┘  │
└───────────────────────────┼──────────────────────────┘
                            │ real-time SSE stream
┌───────────────────────────┼──────────────────────────┐
│  GreenTune Training Pipeline (Python / ROCm)         │
│  ┌──────────────┐  ┌─────┴────────────────────────┐  │
│  │ greentune.py │  │ energy_callback.py           │  │
│  │ QLoRA + SFT  │──│ PowerSamplerThread           │  │
│  │ Qwen2.5-7B   │  │ EnergyAccumulator            │  │
│  └──────────────┘  │ J/token per step             │  │
│                     │ Live upload to dashboard API │  │
│                     └──────────┬───────────────────┘  │
│  ┌──────────────────────────┐  │                     │
│  │ amd_collector.py         │  │ amdsmi / rocm-smi   │
│  │ AMDGPUCollector          │──┘                     │
│  │ (drop-in for NVIDIA)     │                        │
│  └──────────────────────────┘                        │
│                     │                                │
│              AMD MI300X (ROCm 7.0)                   │
│              192GB HBM3 · 750W TDP · gfx942          │
└──────────────────────────────────────────────────────┘
```

## GreenTune Agent — Gemini-Powered Autonomy

The agent accepts natural language and autonomously manages the full training lifecycle:

```bash
# Interactive mode
python greentune_agent.py --interactive

# One-shot recommendation
python greentune_agent.py --request "Lowest J/token config for 500 samples"

# Recommend + launch with live dashboard
python greentune_agent.py \
  --request "Fine-tune Qwen-7B, keep CO2 under 30g" \
  --dashboard-url https://www.aluminatiai.com \
  --dashboard-api-key alum_xxx

# Analyze a completed run
python greentune_agent.py --analyze output/greentune-run/energy_metrics.json

# Show active energy policies
python greentune_agent.py --show-policies
```

The agent uses historical `energy_metrics.json` data to make informed recommendations. Each completed run improves future predictions.

## Energy Governance — Lobster Trap

Enterprise energy policies that run **before** training starts:

| Policy | Description | Default Limit |
|--------|-------------|---------------|
| `carbon_budget` | Max CO2 per run | 50g |
| `energy_cap` | Max energy per run | 1 kWh |
| `efficiency_floor` | Max J/token allowed | 0.8 J/tok |
| `cost_guard` | Max energy cost per run | $1.00 |

When a proposed config violates a policy, the agent explains why and suggests a compliant alternative — the job never launches. This is the governance layer enterprises need for EU AI Act and SEC climate disclosure compliance.

## Key Finding

**Smaller batch sizes do NOT reduce power draw on MI300X.**

We ran identical QLoRA fine-tuning jobs (Qwen2.5-7B, 500 Hermes traces, LoRA rank 16, NF4) with two batch configs:

| Metric | Baseline (bs=2, ga=4) | Small Batch (bs=1, ga=8) | Delta |
|--------|----------------------|--------------------------|-------|
| Effective Batch Size | 8 | 8 | Same |
| Training Duration | 138.5s | 178.9s | +29% |
| Avg Power Draw | 630.5 W | 637.2 W | +1% |
| Peak Power Draw | 752 W | 749 W | ~Same |
| **Total Energy** | **87,300 J** | **113,834 J** | **+30%** |
| **J/Token** | **0.355** | **0.463** | **+30%** |
| Tokens/sec | 1,774 | 1,374 | -23% |
| Energy Cost | $0.0024 | $0.0032 | +33% |
| CO2 Emissions | 9.46 g | 12.33 g | +30% |

The MI300X saturates at ~750W regardless of batch size. Smaller batches = same power x more time = **30% more energy wasted**.

This insight is invisible without per-token energy measurement. GreenTune makes it obvious.

## AMD Hardware

Built and tested on **AMD Instinct MI300X** via AMD Developer Cloud:

| Spec | Value |
|------|-------|
| GPU | AMD Instinct MI300X |
| VRAM | 192 GB HBM3 |
| TDP | 750W |
| Architecture | CDNA3 (gfx942) |
| ROCm | 7.0.51831 |
| PyTorch | 2.9.0+rocm7.0.0 |

**Why MI300X:** 192GB VRAM loads Qwen2.5-7B in 4-bit with only 5.2GB used — no model sharding needed, simplifying energy measurement. `amdsmi` provides hardware-level power telemetry with sub-second granularity.

## Quick Start

### 1. GreenTune Agent (recommended)

```bash
cd agent/finetune
pip install -r ../requirements.txt google-generativeai

# Set your Gemini API key
export GOOGLE_API_KEY=your_key_here

# Interactive agent mode
python greentune_agent.py --interactive

# One-shot: recommend optimal config
python greentune_agent.py --request "Lowest J/token for 500 samples on MI300X"

# Recommend + auto-launch with live dashboard
python greentune_agent.py \
  --request "Fine-tune Qwen-7B under 50g CO2" \
  --dashboard-url https://www.aluminatiai.com \
  --dashboard-api-key alum_xxx
```

### 2. Direct fine-tuning (without agent)

```bash
cd agent/finetune

# Run energy-aware fine-tuning on AMD
python greentune.py \
  --hermes-only --hermes-max 500 \
  --epochs 1 --batch-size 2 --grad-accum 4 \
  --logging-steps 10

# With live dashboard upload
python greentune.py \
  --hermes-only --hermes-max 500 \
  --epochs 1 --batch-size 2 --grad-accum 4 \
  --api-url https://www.aluminatiai.com \
  --api-key alum_xxx \
  --run-name "Baseline bs=2"

# Output: output/greentune-run/
#   energy_metrics.json — summary + per-step energy data
#   power_samples.json  — raw 0.5s power readings
#   run_config.json     — training configuration
#   adapter/            — LoRA adapter weights
```

### 3. GPU agent standalone

```bash
cd agent

# AMD GPU (auto-detects amdsmi or rocm-smi)
python agent.py --interval 2 --dry-run --duration 300

# NVIDIA GPU (auto-detects NVML)
python agent.py --interval 5
```

### 4. Dashboard

The fine-tuning dashboard is deployed at [aluminatiai.com/admin/fine-tuning](https://www.aluminatiai.com/admin/fine-tuning). Features real-time SSE streaming when training is active. Source is in `dashboard/`.

## Repository Structure

```
GreenTune/
├── README.md                          # This file
├── agent/
│   ├── finetune/
│   │   ├── greentune_agent.py         # Gemini-powered Energy Intelligence Agent
│   │   ├── greentune.py               # Main training pipeline (QLoRA + energy)
│   │   ├── energy_callback.py         # HF Trainer callback — power sampling + J/token + live upload
│   │   ├── rocm_power.py             # AMD power monitor (PowerSamplerThread)
│   │   ├── dataset_builder.py         # Synthetic GPU domain dataset generator
│   │   ├── eval_model.py             # Post-training evaluation
│   │   ├── smoke_test.py             # Quick validation script
│   │   ├── verify_rocm.py            # ROCm environment checker
│   │   ├── AMD_SETUP_GUIDE.md        # MI300X setup instructions
│   │   └── HACKATHON_SUBMISSION.md   # Detailed submission writeup
│   ├── agent.py                       # Main GPU monitoring agent
│   ├── amd_collector.py              # AMD GPU collector (amdsmi + CLI fallback)
│   ├── collector.py                   # NVIDIA GPU collector (pynvml)
│   ├── metrics_server.py            # Prometheus metrics exporter
│   ├── uploader.py                   # Metrics upload to AluminatiAI platform
│   ├── pyproject.toml                # Python package config
│   └── requirements.txt             # Dependencies
├── dashboard/
│   └── app/
│       ├── admin/fine-tuning/page.tsx  # 5-tab dashboard UI (Overview + Recharts)
│       └── api/admin/fine-tuning/
│           ├── route.ts               # AI-powered run analysis API
│           ├── ingest/route.ts        # Live metrics ingest from training agent
│           └── stream/route.ts        # SSE real-time stream to dashboard
└── LICENSE
```

## How Energy Measurement Works

Power is sampled via `amdsmi.amdsmi_get_power_info()` every 0.5s in a background thread. Energy per interval uses trapezoidal integration:

```
energy_delta = (power_current + power_previous) / 2 × dt
```

Per-step metrics aggregate samples between training steps:
- **step_joules** — total energy for the logging window
- **joules_per_token** — step_joules / tokens_in_window
- **cumulative_kwh** — running total for the entire run
- **cumulative_co2_grams** — kWh × 390 gCO2/kWh (US average)

The energy callback integrates with HuggingFace Trainer, requiring zero changes to training code:

```python
from energy_callback import EnergyCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[EnergyCallback()],  # <- one line
)
```

## GPU Agent — Backend Agnostic

The AluminatiAI agent auto-detects GPU vendor at startup:

```python
# Try NVIDIA first (pynvml)
collector = GPUCollector()

# Fall back to AMD (amdsmi / rocm-smi CLI)
if collector is None:
    collector = AMDGPUCollector()
```

Both return identical `GPUMetrics` dataclass instances. The same platform, dashboard, and alerting works on NVIDIA A100/H100 and AMD MI300X without code changes.

## What's Next

- **Multi-GPU fleet agent** — Agent manages training across multiple MI300X GPUs, routing jobs to the most efficient hardware
- **Cross-GPU benchmarking** — Same workload on A100, H100, MI300X to build a J/token comparison database
- **Carbon compliance reports** — Audit-ready energy/emissions reports for EU AI Act / SEC disclosures
- **Custom Lobster Trap policies** — Admin UI for defining and managing energy governance rules
- **Predictive energy modeling** — Train a model on historical energy data to predict J/token before training starts

## License

Apache 2.0

---

*Built with AMD Instinct MI300X on AMD Developer Cloud. Gemini 2.5 for agent reasoning. All energy measurements are from real training runs, not simulations.*
