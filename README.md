# AluminatiAI GreenTune

**Energy-aware fine-tuning on AMD MI300X. Measures Joules-per-token as a first-class training metric.**

> AMD Developer Hackathon 2026 — Fine-Tuning Track  
> Team: AluminatiAI (Kevin Mello)  
> Live Dashboard: [aluminatiai.com/admin/fine-tuning](https://www.aluminatiai.com/admin/fine-tuning)

---

## The Problem

Every organization fine-tuning LLMs today is flying blind on energy. Teams track loss curves, learning rates, and throughput — but nobody measures the **energy cost per token produced**.

- GPU energy is invisible. MI300X draws 750W at peak but teams never see it.
- Optimization is guesswork. Without per-token energy metrics, you can't compare configs on efficiency.
- Carbon compliance is coming. EU AI Act and SEC climate disclosures will require energy reporting for AI workloads.

## The Solution

GreenTune treats **Joules-per-token** as a first-class metric alongside loss and throughput.

```
┌──────────────────────────────────────────────────────┐
│  AluminatiAI Platform (Next.js + Supabase)           │
│  ┌────────────────────────────────────────────────┐  │
│  │  GreenTune Dashboard (/admin/fine-tuning)      │  │
│  │  • Run Monitor — power curves, loss charts     │  │
│  │  • Leaderboard — side-by-side run comparison   │  │
│  │  • ROI Calculator — cost projections           │  │
│  │  • Playground — model evaluation               │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────┘
                       │ metrics JSON
┌──────────────────────┴───────────────────────────────┐
│  GreenTune Training Pipeline (Python / ROCm)         │
│  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ greentune.py │  │ energy_callback.py           │  │
│  │ QLoRA + SFT  │──│ PowerSamplerThread           │  │
│  │ Qwen2.5-7B   │  │ EnergyAccumulator            │  │
│  └──────────────┘  │ J/token per step             │  │
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

### Fine-tuning with energy tracking

```bash
cd agent/finetune

# Install dependencies
pip install -r ../requirements.txt

# Run energy-aware fine-tuning on AMD
python greentune.py \
  --hermes-only --hermes-max 500 \
  --epochs 1 --batch-size 2 --grad-accum 4 \
  --logging-steps 10

# Output: output/greentune-run/
#   energy_metrics.json — summary + per-step energy data
#   power_samples.json  — raw 0.5s power readings
#   run_config.json     — training configuration
#   adapter/            — LoRA adapter weights
```

### Running the GPU agent standalone

```bash
cd agent

# AMD GPU (auto-detects amdsmi or rocm-smi)
python agent.py --interval 2 --dry-run --duration 300

# NVIDIA GPU (auto-detects NVML)
python agent.py --interval 5
```

### Dashboard (Next.js)

The fine-tuning dashboard is deployed at [aluminatiai.com/admin/fine-tuning](https://www.aluminatiai.com/admin/fine-tuning). Source is in `dashboard/`.

## Repository Structure

```
GreenTune/
├── README.md                          # This file
├── agent/
│   ├── finetune/
│   │   ├── greentune.py               # Main training pipeline (QLoRA + energy)
│   │   ├── energy_callback.py         # HF Trainer callback — power sampling + J/token
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
│       ├── admin/fine-tuning/page.tsx  # 4-tab dashboard UI (Recharts)
│       └── api/admin/fine-tuning/route.ts  # AI-powered run analysis API
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

- **Cross-GPU benchmarking** — Same workload on A100, H100, MI300X to build a J/token comparison database
- **Automated config recommendations** — Use energy data to suggest optimal batch size, LoRA rank, sequence length
- **Carbon compliance reports** — Audit-ready energy/emissions reports for EU AI Act / SEC disclosures
- **Fleet-level optimization** — Route workloads to hardware based on J/token efficiency

## License

Apache 2.0

---

*Built with AMD Instinct MI300X on AMD Developer Cloud. All energy measurements are from real training runs, not simulations.*
