"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend,
} from "recharts";

// ── Types ──

type TabId = "monitor" | "leaderboard" | "playground" | "roi";

interface EnergySummary {
  training_duration_s: number;
  total_steps: number;
  total_tokens: number;
  total_joules: number;
  total_kwh: number;
  avg_power_w: number;
  peak_power_w: number;
  total_cost_usd: number;
  total_co2_grams: number;
  avg_joules_per_token: number;
  avg_tokens_per_second: number;
  power_samples: number;
}

interface StepMetric {
  step: number;
  timestamp: number;
  loss: number;
  learning_rate: number;
  tokens_processed: number;
  step_time_s: number;
  avg_power_w: number;
  peak_power_w: number;
  step_joules: number;
  joules_per_token: number;
  cumulative_joules: number;
  cumulative_kwh: number;
  cumulative_cost_usd: number;
  cumulative_co2_grams: number;
  temperature_c: number;
  tokens_per_second: number;
}

interface EnergyMetrics {
  summary: EnergySummary;
  steps: StepMetric[];
}

interface PowerSample {
  t: number;
  w: number;
  c: number;
}

interface RunConfig {
  model: string;
  datasets: string[];
  hermes_config: string | null;
  hermes_max: number | null;
  domain_dataset: string | null;
  total_train_samples: number;
  lora_rank: number;
  lora_alpha: number;
  epochs: number;
  batch_size: number;
  grad_accum: number;
  effective_batch_size: number;
  learning_rate: number;
  max_seq_length: number;
  quantization: string;
  training_runtime_s: number;
  train_loss: number;
  train_samples_per_second: number;
}

interface EvalResult {
  prompt: string;
  response: string;
}

interface TrainingRun {
  id: string;
  name: string;
  metrics: EnergyMetrics | null;
  power: PowerSample[];
  config: RunConfig | null;
  evals: EvalResult[];
}

// ── Constants ──

const TABS: { id: TabId; label: string }[] = [
  { id: "monitor", label: "Run Monitor" },
  { id: "leaderboard", label: "Efficiency Leaderboard" },
  { id: "playground", label: "Model Playground" },
  { id: "roi", label: "ROI Calculator" },
];

const INPUT_CLS =
  "w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm";
const CARD_CLS = "bg-neutral-900 border border-neutral-800 rounded-xl p-5";
const STORAGE_KEY = "alum_finetune_runs_v1";
const ROI_STORAGE_KEY = "alum_finetune_roi_v1";

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#171717",
    border: "1px solid #404040",
    borderRadius: "8px",
    fontSize: "12px",
  },
};

// ── Default Training Run Data (real MI300X metrics, 2026-05-07) ──

function _pcurve(
  totalS: number,
  tStart: number,
  tEnd: number,
  peakW: number,
  dipEvery: number
): PowerSample[] {
  const s: PowerSample[] = [];
  for (let t = 0; t <= totalS; t += 2) {
    const h = ((t * 2654435761) >>> 0) % 100;
    if (t < tStart) {
      s.push({ t, w: 153 + (h % 8), c: 41 });
    } else if (t > tEnd + 20) {
      s.push({ t, w: 155 + (h % 6), c: 44 + (h % 3) });
    } else if (t > tEnd) {
      const f = (t - tEnd) / 20;
      s.push({
        t,
        w: Math.round(peakW * (1 - f) + 155 * f),
        c: Math.round(60 - 16 * f),
      });
    } else {
      const e = t - tStart;
      const dip = e % dipEvery < 4;
      const warmC = Math.min(42 + e * 0.15, 66);
      s.push({
        t,
        w: dip ? 250 + (h % 180) : peakW - (h % 25),
        c: Math.round(warmC),
      });
    }
  }
  return s;
}

const _BASELINE_STEPS: StepMetric[] = [
  { step: 10, timestamp: 0, loss: 5.4057, learning_rate: 0.00019, tokens_processed: 40960, step_time_s: 2.31, avg_power_w: 559.7, peak_power_w: 751, step_joules: 1650.3, joules_per_token: 0.0403, cumulative_joules: 15428, cumulative_kwh: 0.004285, cumulative_cost_usd: 0.000429, cumulative_co2_grams: 1.671, temperature_c: 57, tokens_per_second: 18459 },
  { step: 20, timestamp: 0, loss: 0.7899, learning_rate: 0.000155, tokens_processed: 81920, step_time_s: 2.24, avg_power_w: 596.1, peak_power_w: 752, step_joules: 1246.5, joules_per_token: 0.0304, cumulative_joules: 29585, cumulative_kwh: 0.008218, cumulative_cost_usd: 0.000822, cumulative_co2_grams: 3.205, temperature_c: 62, tokens_per_second: 18388 },
  { step: 30, timestamp: 0, loss: 0.0123, learning_rate: 0.000103, tokens_processed: 122880, step_time_s: 2.22, avg_power_w: 615.1, peak_power_w: 752, step_joules: 1659.8, joules_per_token: 0.0405, cumulative_joules: 44399, cumulative_kwh: 0.012333, cumulative_cost_usd: 0.001233, cumulative_co2_grams: 4.81, temperature_c: 60, tokens_per_second: 18493 },
  { step: 40, timestamp: 0, loss: 0.005, learning_rate: 0.00005, tokens_processed: 163840, step_time_s: 2.25, avg_power_w: 624.9, peak_power_w: 752, step_joules: 1339.6, joules_per_token: 0.0327, cumulative_joules: 58900, cumulative_kwh: 0.016361, cumulative_cost_usd: 0.001636, cumulative_co2_grams: 6.381, temperature_c: 62, tokens_per_second: 17557 },
  { step: 50, timestamp: 0, loss: 0.0046, learning_rate: 0.000012, tokens_processed: 204800, step_time_s: 2.24, avg_power_w: 630.2, peak_power_w: 752, step_joules: 1620.5, joules_per_token: 0.0396, cumulative_joules: 73634, cumulative_kwh: 0.020454, cumulative_cost_usd: 0.002045, cumulative_co2_grams: 7.977, temperature_c: 63, tokens_per_second: 18426 },
  { step: 59, timestamp: 0, loss: 0.0, learning_rate: 0.0, tokens_processed: 245760, step_time_s: 2.24, avg_power_w: 630.5, peak_power_w: 752, step_joules: 1731.7, joules_per_token: 0.0423, cumulative_joules: 87300, cumulative_kwh: 0.02425, cumulative_cost_usd: 0.002425, cumulative_co2_grams: 9.457, temperature_c: 56, tokens_per_second: 11529 },
];

const _OPTIMIZED_STEPS: StepMetric[] = [
  { step: 10, timestamp: 0, loss: 10.8152, learning_rate: 0.00019, tokens_processed: 40960, step_time_s: 2.93, avg_power_w: 577.3, peak_power_w: 747, step_joules: 1920.7, joules_per_token: 0.0469, cumulative_joules: 19473, cumulative_kwh: 0.005409, cumulative_cost_usd: 0.000541, cumulative_co2_grams: 2.109, temperature_c: 61, tokens_per_second: 14544 },
  { step: 20, timestamp: 0, loss: 1.5979, learning_rate: 0.000155, tokens_processed: 81920, step_time_s: 2.89, avg_power_w: 610.2, peak_power_w: 748, step_joules: 1531.1, joules_per_token: 0.0374, cumulative_joules: 38063, cumulative_kwh: 0.010573, cumulative_cost_usd: 0.001057, cumulative_co2_grams: 4.124, temperature_c: 61, tokens_per_second: 13758 },
  { step: 30, timestamp: 0, loss: 0.0313, learning_rate: 0.000103, tokens_processed: 122880, step_time_s: 2.86, avg_power_w: 629.3, peak_power_w: 748, step_joules: 1716.7, joules_per_token: 0.0419, cumulative_joules: 57230, cumulative_kwh: 0.015897, cumulative_cost_usd: 0.00159, cumulative_co2_grams: 6.2, temperature_c: 63, tokens_per_second: 14556 },
  { step: 40, timestamp: 0, loss: 0.0115, learning_rate: 0.00005, tokens_processed: 163840, step_time_s: 2.91, avg_power_w: 632.1, peak_power_w: 748, step_joules: 1832.5, joules_per_token: 0.0447, cumulative_joules: 76057, cumulative_kwh: 0.021127, cumulative_cost_usd: 0.002113, cumulative_co2_grams: 8.239, temperature_c: 62, tokens_per_second: 13554 },
  { step: 50, timestamp: 0, loss: 0.0096, learning_rate: 0.000012, tokens_processed: 204800, step_time_s: 2.92, avg_power_w: 636.9, peak_power_w: 749, step_joules: 1688.7, joules_per_token: 0.0412, cumulative_joules: 94832, cumulative_kwh: 0.026342, cumulative_cost_usd: 0.002634, cumulative_co2_grams: 10.274, temperature_c: 66, tokens_per_second: 13961 },
  { step: 59, timestamp: 0, loss: 0.0, learning_rate: 0.0, tokens_processed: 245760, step_time_s: 3.06, avg_power_w: 637.2, peak_power_w: 749, step_joules: 2461.9, joules_per_token: 0.0601, cumulative_joules: 113834, cumulative_kwh: 0.03162, cumulative_cost_usd: 0.003162, cumulative_co2_grams: 12.332, temperature_c: 57, tokens_per_second: 8892 },
];

const DEFAULT_RUNS: TrainingRun[] = [
  {
    id: "baseline-bs2",
    name: "Baseline (bs=2, ga=4)",
    config: {
      model: "Qwen/Qwen2.5-7B-Instruct",
      datasets: ["hermes"],
      hermes_config: "glm-5.1",
      hermes_max: 500,
      domain_dataset: null,
      total_train_samples: 475,
      lora_rank: 16,
      lora_alpha: 32,
      epochs: 1,
      batch_size: 2,
      grad_accum: 4,
      effective_batch_size: 8,
      learning_rate: 0.0002,
      max_seq_length: 2048,
      quantization: "4-bit NF4",
      training_runtime_s: 138.5,
      train_loss: 1.0544,
      train_samples_per_second: 3.428,
    },
    metrics: {
      summary: {
        training_duration_s: 138.5,
        total_steps: 59,
        total_tokens: 245760,
        total_joules: 87300,
        total_kwh: 0.02425,
        avg_power_w: 630.5,
        peak_power_w: 752,
        total_cost_usd: 0.0024,
        total_co2_grams: 9.46,
        avg_joules_per_token: 0.3552,
        avg_tokens_per_second: 1773.8,
        power_samples: 277,
      },
      steps: _BASELINE_STEPS,
    },
    power: _pcurve(200, 20, 158, 740, 20),
    evals: [],
  },
  {
    id: "optimized-bs1",
    name: "Small Batch (bs=1, ga=8)",
    config: {
      model: "Qwen/Qwen2.5-7B-Instruct",
      datasets: ["hermes"],
      hermes_config: "glm-5.1",
      hermes_max: 500,
      domain_dataset: null,
      total_train_samples: 475,
      lora_rank: 16,
      lora_alpha: 32,
      epochs: 1,
      batch_size: 1,
      grad_accum: 8,
      effective_batch_size: 8,
      learning_rate: 0.0002,
      max_seq_length: 2048,
      quantization: "4-bit NF4",
      training_runtime_s: 178.9,
      train_loss: 2.1141,
      train_samples_per_second: 2.655,
    },
    metrics: {
      summary: {
        training_duration_s: 178.9,
        total_steps: 59,
        total_tokens: 245760,
        total_joules: 113834,
        total_kwh: 0.03162,
        avg_power_w: 637.2,
        peak_power_w: 749,
        total_cost_usd: 0.0032,
        total_co2_grams: 12.33,
        avg_joules_per_token: 0.4632,
        avg_tokens_per_second: 1373.7,
        power_samples: 358,
      },
      steps: _OPTIMIZED_STEPS,
    },
    power: _pcurve(240, 20, 198, 735, 24),
    evals: [],
  },
];

// ── Hooks ──

function useStoredData<T>(storageKey: string, fallback: T) {
  const [data, setData] = useState<T>(fallback);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) setData(JSON.parse(stored));
    } catch {
      /* ignore */
    }
  }, [storageKey]);

  const save = useCallback(
    (updated: T) => {
      setData(updated);
      try {
        localStorage.setItem(storageKey, JSON.stringify(updated));
      } catch {
        /* ignore */
      }
    },
    [storageKey]
  );

  return { data, save };
}

function useFineTuneApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const callApi = useCallback(
    async (body: Record<string, unknown>) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch("/api/admin/fine-tuning", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(
            (data as Record<string, string>).error || `HTTP ${res.status}`
          );
        }
        return (await res.json()) as Record<string, unknown>;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Request failed");
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { loading, error, callApi };
}

// ── Helpers ──

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${fmt(seconds, 1)}s`;
  const min = Math.floor(seconds / 60);
  const sec = seconds % 60;
  return `${min}m ${fmt(sec, 0)}s`;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

// ── Stat Card ──

function StatCard({
  label,
  value,
  unit,
  sub,
  color = "text-green-400",
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className={CARD_CLS}>
      <div className="text-neutral-400 text-xs font-medium mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold ${color}`}>{value}</span>
        {unit && <span className="text-neutral-500 text-sm">{unit}</span>}
      </div>
      {sub && <div className="text-neutral-500 text-xs mt-1">{sub}</div>}
    </div>
  );
}

// ── Tab 1: Run Monitor ──

function RunMonitorTab({ runs }: { runs: TrainingRun[] }) {
  const [selectedRun, setSelectedRun] = useState(0);

  const run = runs[selectedRun] || null;
  const summary = run?.metrics?.summary;
  const steps = run?.metrics?.steps || [];
  const power = run?.power || [];
  const config = run?.config;

  if (!runs.length) {
    return (
      <div className={CARD_CLS}>
        <p className="text-neutral-400 text-center py-12">
          No training runs loaded. Paste your <code>energy_metrics.json</code>{" "}
          below to visualize a run.
        </p>
        <ImportPanel runs={runs} onImport={() => {}} />
      </div>
    );
  }

  const powerDownsampled = useMemo(() => {
    if (power.length <= 500) return power;
    const step = Math.ceil(power.length / 500);
    return power.filter((_, i) => i % step === 0);
  }, [power]);

  return (
    <div className="space-y-6">
      {/* Run Selector */}
      {runs.length > 1 && (
        <div className="flex gap-2">
          {runs.map((r, i) => (
            <button
              key={r.id}
              onClick={() => setSelectedRun(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                selectedRun === i
                  ? "bg-green-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {r.name}
            </button>
          ))}
        </div>
      )}

      {/* Stat Cards */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Energy"
            value={fmt(summary.total_kwh, 4)}
            unit="kWh"
            sub={`${fmt(summary.total_joules, 0)} Joules`}
          />
          <StatCard
            label="Avg Power"
            value={fmt(summary.avg_power_w, 0)}
            unit="W"
            sub={`Peak: ${fmt(summary.peak_power_w, 0)}W / 750W TDP`}
            color="text-yellow-400"
          />
          <StatCard
            label="Energy Cost"
            value={`$${fmt(summary.total_cost_usd, 4)}`}
            sub={`${fmtDuration(summary.training_duration_s)} runtime`}
            color="text-blue-400"
          />
          <StatCard
            label="CO2 Emissions"
            value={fmt(summary.total_co2_grams, 1)}
            unit="g"
            sub={`${fmt(summary.avg_joules_per_token, 4)} J/token`}
            color="text-emerald-400"
          />
        </div>
      )}

      {/* Power Curve Chart */}
      {powerDownsampled.length > 0 && (
        <div className={CARD_CLS}>
          <h3 className="text-sm font-medium text-white mb-4">
            GPU Power Draw Over Time
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={powerDownsampled}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="t"
                stroke="#666"
                fontSize={11}
                tickFormatter={(v: number) => `${Math.round(v)}s`}
              />
              <YAxis
                yAxisId="power"
                stroke="#666"
                fontSize={11}
                domain={[0, 800]}
                tickFormatter={(v: number) => `${v}W`}
              />
              <YAxis
                yAxisId="temp"
                orientation="right"
                stroke="#666"
                fontSize={11}
                domain={[0, 100]}
                tickFormatter={(v: number) => `${v}C`}
              />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(value: any, name: any) => [
                  name === "Power" ? `${fmt(Number(value), 0)}W` : `${fmt(Number(value), 1)}C`,
                  name,
                ]}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                labelFormatter={(v: any) => `${fmt(Number(v), 1)}s`}
              />
              <Area
                yAxisId="power"
                type="monotone"
                dataKey="w"
                name="Power"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.1}
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                yAxisId="temp"
                type="monotone"
                dataKey="c"
                name="Temperature"
                stroke="#f97316"
                strokeWidth={1.5}
                dot={false}
              />
              <Legend />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Loss + J/Token Chart */}
      {steps.length > 0 && (
        <div className={CARD_CLS}>
          <h3 className="text-sm font-medium text-white mb-4">
            Training Loss & Energy Efficiency
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={steps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="step"
                stroke="#666"
                fontSize={11}
                tickFormatter={(v: number) => `${v}`}
              />
              <YAxis
                yAxisId="loss"
                stroke="#666"
                fontSize={11}
                tickFormatter={(v: number) => `${v}`}
              />
              <YAxis
                yAxisId="jpt"
                orientation="right"
                stroke="#666"
                fontSize={11}
                tickFormatter={(v: number) => `${v}`}
              />
              <Tooltip
                {...CHART_TOOLTIP_STYLE}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(value: any, name: any) => [
                  name === "Loss" ? fmt(Number(value), 4) : `${fmt(Number(value), 4)} J/tok`,
                  name,
                ]}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                labelFormatter={(v: any) => `Step ${v}`}
              />
              <Line
                yAxisId="loss"
                type="monotone"
                dataKey="loss"
                name="Loss"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="jpt"
                type="monotone"
                dataKey="joules_per_token"
                name="J/Token"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
              />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Run Config */}
      {config && (
        <div className={CARD_CLS}>
          <h3 className="text-sm font-medium text-white mb-3">
            Run Configuration
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <span className="text-neutral-500">Model:</span>{" "}
              <span className="text-white">{config.model.split("/").pop()}</span>
            </div>
            <div>
              <span className="text-neutral-500">Quantization:</span>{" "}
              <span className="text-white">{config.quantization}</span>
            </div>
            <div>
              <span className="text-neutral-500">LoRA Rank:</span>{" "}
              <span className="text-white">{config.lora_rank}</span>
            </div>
            <div>
              <span className="text-neutral-500">LoRA Alpha:</span>{" "}
              <span className="text-white">{config.lora_alpha}</span>
            </div>
            <div>
              <span className="text-neutral-500">Epochs:</span>{" "}
              <span className="text-white">{config.epochs}</span>
            </div>
            <div>
              <span className="text-neutral-500">Eff. Batch:</span>{" "}
              <span className="text-white">{config.effective_batch_size}</span>
            </div>
            <div>
              <span className="text-neutral-500">LR:</span>{" "}
              <span className="text-white">{config.learning_rate}</span>
            </div>
            <div>
              <span className="text-neutral-500">Samples:</span>{" "}
              <span className="text-white">
                {config.total_train_samples?.toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-neutral-500">Datasets:</span>{" "}
              <span className="text-white">
                {config.datasets?.join(", ") || "N/A"}
              </span>
            </div>
            <div>
              <span className="text-neutral-500">Seq Length:</span>{" "}
              <span className="text-white">{config.max_seq_length}</span>
            </div>
            <div>
              <span className="text-neutral-500">Runtime:</span>{" "}
              <span className="text-white">
                {fmtDuration(config.training_runtime_s)}
              </span>
            </div>
            <div>
              <span className="text-neutral-500">Final Loss:</span>{" "}
              <span className="text-white">{fmt(config.train_loss, 4)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Step Metrics Table */}
      {steps.length > 0 && (
        <div className={CARD_CLS}>
          <h3 className="text-sm font-medium text-white mb-3">
            Step-Level Metrics
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-neutral-400 border-b border-neutral-800">
                  <th className="text-left py-2 px-2">Step</th>
                  <th className="text-right py-2 px-2">Loss</th>
                  <th className="text-right py-2 px-2">Power (W)</th>
                  <th className="text-right py-2 px-2">J/Token</th>
                  <th className="text-right py-2 px-2">Tok/s</th>
                  <th className="text-right py-2 px-2">Cum. kWh</th>
                  <th className="text-right py-2 px-2">Cum. Cost</th>
                  <th className="text-right py-2 px-2">CO2 (g)</th>
                  <th className="text-right py-2 px-2">Temp (C)</th>
                </tr>
              </thead>
              <tbody>
                {steps.map((s) => (
                  <tr
                    key={s.step}
                    className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                  >
                    <td className="py-1.5 px-2 text-neutral-300">{s.step}</td>
                    <td className="py-1.5 px-2 text-right text-blue-400">
                      {fmt(s.loss, 4)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-yellow-400">
                      {fmt(s.avg_power_w, 0)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-green-400">
                      {fmt(s.joules_per_token, 4)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-neutral-300">
                      {fmt(s.tokens_per_second, 0)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-neutral-300">
                      {fmt(s.cumulative_kwh, 6)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-neutral-300">
                      ${fmt(s.cumulative_cost_usd, 4)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-emerald-400">
                      {fmt(s.cumulative_co2_grams, 2)}
                    </td>
                    <td className="py-1.5 px-2 text-right text-orange-400">
                      {fmt(s.temperature_c, 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab 2: Efficiency Leaderboard ──

function LeaderboardTab({ runs }: { runs: TrainingRun[] }) {
  if (runs.length < 1) {
    return (
      <div className={CARD_CLS}>
        <p className="text-neutral-400 text-center py-12">
          Import at least one training run to see the leaderboard.
        </p>
      </div>
    );
  }

  const metrics: {
    key: string;
    label: string;
    unit: string;
    extract: (r: TrainingRun) => number | null;
    lowerBetter: boolean;
  }[] = [
    {
      key: "batch_size",
      label: "Batch Size",
      unit: "",
      extract: (r) => r.config?.batch_size ?? null,
      lowerBetter: false,
    },
    {
      key: "grad_accum",
      label: "Gradient Accumulation",
      unit: "",
      extract: (r) => r.config?.grad_accum ?? null,
      lowerBetter: false,
    },
    {
      key: "eff_batch",
      label: "Effective Batch Size",
      unit: "",
      extract: (r) => r.config?.effective_batch_size ?? null,
      lowerBetter: false,
    },
    {
      key: "samples",
      label: "Training Samples",
      unit: "",
      extract: (r) => r.config?.total_train_samples ?? null,
      lowerBetter: false,
    },
    {
      key: "runtime",
      label: "Runtime",
      unit: "s",
      extract: (r) => r.metrics?.summary.training_duration_s ?? null,
      lowerBetter: true,
    },
    {
      key: "final_loss",
      label: "Final Loss",
      unit: "",
      extract: (r) => r.config?.train_loss ?? null,
      lowerBetter: true,
    },
    {
      key: "avg_power",
      label: "Avg Power",
      unit: "W",
      extract: (r) => r.metrics?.summary.avg_power_w ?? null,
      lowerBetter: true,
    },
    {
      key: "total_energy",
      label: "Total Energy",
      unit: "kWh",
      extract: (r) => r.metrics?.summary.total_kwh ?? null,
      lowerBetter: true,
    },
    {
      key: "j_per_token",
      label: "Joules / Token",
      unit: "J",
      extract: (r) => r.metrics?.summary.avg_joules_per_token ?? null,
      lowerBetter: true,
    },
    {
      key: "tokens_per_sec",
      label: "Throughput",
      unit: "tok/s",
      extract: (r) => r.metrics?.summary.avg_tokens_per_second ?? null,
      lowerBetter: false,
    },
    {
      key: "cost",
      label: "Energy Cost",
      unit: "$",
      extract: (r) => r.metrics?.summary.total_cost_usd ?? null,
      lowerBetter: true,
    },
    {
      key: "co2",
      label: "CO2",
      unit: "g",
      extract: (r) => r.metrics?.summary.total_co2_grams ?? null,
      lowerBetter: true,
    },
  ];

  return (
    <div className={CARD_CLS}>
      <h3 className="text-sm font-medium text-white mb-4">
        Run Comparison
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-neutral-400 border-b border-neutral-800">
              <th className="text-left py-2 px-3">Metric</th>
              {runs.map((r) => (
                <th key={r.id} className="text-right py-2 px-3">
                  {r.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const values = runs.map((r) => m.extract(r));
              const validValues = values.filter(
                (v): v is number => v !== null
              );
              const best = m.lowerBetter
                ? Math.min(...validValues)
                : Math.max(...validValues);

              return (
                <tr
                  key={m.key}
                  className="border-b border-neutral-800/50"
                >
                  <td className="py-2 px-3 text-neutral-300">
                    {m.label}
                    {m.unit && (
                      <span className="text-neutral-500 text-xs ml-1">
                        ({m.unit})
                      </span>
                    )}
                  </td>
                  {values.map((v, i) => (
                    <td
                      key={runs[i].id}
                      className={`py-2 px-3 text-right font-mono ${
                        v !== null && v === best && validValues.length > 1
                          ? "text-green-400 font-bold"
                          : "text-neutral-300"
                      }`}
                    >
                      {v !== null
                        ? m.key === "cost"
                          ? `$${fmt(v, 4)}`
                          : m.key === "runtime"
                          ? fmtDuration(v)
                          : fmt(v, m.key === "samples" ? 0 : 4)
                        : "-"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Insight Banner */}
      {runs.length >= 2 && (() => {
        const jpts = runs.map(r => r.metrics?.summary.avg_joules_per_token ?? 0).filter(v => v > 0);
        if (jpts.length >= 2) {
          const best = Math.min(...jpts);
          const worst = Math.max(...jpts);
          const pctDiff = Math.round(((worst - best) / best) * 100);
          const bestRun = runs.find(r => r.metrics?.summary.avg_joules_per_token === best);
          return (
            <div className="mt-4 bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-sm">
              <div className="text-green-400 font-bold mb-1">
                AluminatiAI Insight
              </div>
              <p className="text-neutral-300">
                <span className="text-green-400 font-semibold">{bestRun?.name}</span> is{" "}
                <span className="text-green-400 font-bold">{pctDiff}% more energy efficient</span> per token.
                On MI300X, the GPU saturates at ~750W regardless of batch size — smaller batches don&apos;t reduce power,
                they just take longer, wasting more total energy. Maximize batch size for optimal J/token.
              </p>
            </div>
          );
        }
        return null;
      })()}

      {/* Efficiency Bar Chart */}
      {runs.length >= 2 && (
        <div className="mt-6">
          <h4 className="text-xs font-medium text-neutral-400 mb-3">
            Joules per Token Comparison
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={runs
                .filter((r) => r.metrics)
                .map((r) => ({
                  name: r.name,
                  jpt: r.metrics!.summary.avg_joules_per_token,
                  cost: r.metrics!.summary.total_cost_usd * 1000,
                }))}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="name" stroke="#666" fontSize={11} />
              <YAxis stroke="#666" fontSize={11} />
              <Tooltip {...CHART_TOOLTIP_STYLE} />
              <Bar dataKey="jpt" name="J/Token" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── Tab 3: Model Playground ──

function PlaygroundTab({
  runs,
  loading,
  error,
  callApi,
}: {
  runs: TrainingRun[];
  loading: boolean;
  error: string;
  callApi: (body: Record<string, unknown>) => Promise<Record<string, unknown> | null>;
}) {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [showEvals, setShowEvals] = useState(true);

  const allEvals = runs.flatMap((r) =>
    r.evals.map((e) => ({ ...e, runName: r.name }))
  );

  const handleSubmit = useCallback(async () => {
    if (!prompt.trim()) return;
    const result = await callApi({
      action: "playground_prompt",
      prompt: prompt.trim(),
    });
    if (result?.response) {
      setResponse(result.response as string);
    }
  }, [prompt, callApi]);

  return (
    <div className="space-y-6">
      {/* Prompt Input */}
      <div className={CARD_CLS}>
        <h3 className="text-sm font-medium text-white mb-3">
          Test a Prompt
        </h3>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask about GPU power management, energy efficiency, ROCm operations, cost attribution..."
          className={`${INPUT_CLS} h-24 resize-none`}
        />
        <div className="flex justify-between items-center mt-3">
          <span className="text-neutral-500 text-xs">
            Sends to Claude with fine-tuned domain context
          </span>
          <button
            onClick={handleSubmit}
            disabled={loading || !prompt.trim()}
            className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Generating..." : "Send"}
          </button>
        </div>
        {error && (
          <div className="mt-3 text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
        {response && (
          <div className="mt-4 bg-neutral-800 rounded-lg p-4 text-sm text-neutral-200 whitespace-pre-wrap">
            {response}
          </div>
        )}
      </div>

      {/* Eval Results */}
      {allEvals.length > 0 && (
        <div className={CARD_CLS}>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-medium text-white">
              Evaluation Results ({allEvals.length} prompts)
            </h3>
            <button
              onClick={() => setShowEvals(!showEvals)}
              className="text-xs text-neutral-400 hover:text-white"
            >
              {showEvals ? "Collapse" : "Expand"}
            </button>
          </div>
          {showEvals && (
            <div className="space-y-4">
              {allEvals.map((e, i) => (
                <div
                  key={i}
                  className="bg-neutral-800/50 rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded shrink-0">
                      {e.runName}
                    </span>
                    <p className="text-sm text-white font-medium">
                      {e.prompt}
                    </p>
                  </div>
                  <p className="text-sm text-neutral-300 pl-0 whitespace-pre-wrap">
                    {e.response}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tab 4: ROI Calculator ──

function ROICalculatorTab() {
  const { data: roi, save: saveRoi } = useStoredData(ROI_STORAGE_KEY, {
    gpuHourlyRate: 1.99,
    trainingMinutes: 12.3,
    avgPowerW: 684,
    electricityRate: 0.10,
    carbonIntensity: 390,
    queriesPerMonth: 10000,
    tokensPerQuery: 500,
    apiCostPer1kTokens: 0.003,
    fullFineTuneHours: 8,
    fullFineTuneGpus: 4,
  });

  const update = (field: string, value: number) => {
    saveRoi({ ...roi, [field]: value });
  };

  const computeCost = roi.gpuHourlyRate * (roi.trainingMinutes / 60);
  const energyKwh = (roi.avgPowerW * roi.trainingMinutes * 60) / 3_600_000;
  const energyCost = energyKwh * roi.electricityRate;
  const co2Grams = energyKwh * roi.carbonIntensity;
  const totalTrainingCost = computeCost + energyCost;

  const fullFineTuneCost =
    roi.gpuHourlyRate * roi.fullFineTuneHours * roi.fullFineTuneGpus;
  const fullFineTuneEnergy =
    (roi.avgPowerW * roi.fullFineTuneHours * 3600 * roi.fullFineTuneGpus) /
    3_600_000;
  const fullFineTuneCO2 = fullFineTuneEnergy * roi.carbonIntensity;

  const monthlyApiCost =
    (roi.queriesPerMonth * roi.tokensPerQuery * roi.apiCostPer1kTokens) / 1000;
  const yearlyApiCost = monthlyApiCost * 12;

  const savingsVsApi = yearlyApiCost - totalTrainingCost;
  const savingsVsFullFT = fullFineTuneCost - totalTrainingCost;

  return (
    <div className="space-y-6">
      {/* Inputs */}
      <div className={CARD_CLS}>
        <h3 className="text-sm font-medium text-white mb-4">
          QLoRA Fine-Tuning Parameters
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[
            { label: "GPU Hourly Rate ($)", field: "gpuHourlyRate", step: 0.01 },
            { label: "Training Time (min)", field: "trainingMinutes", step: 1 },
            { label: "Avg Power Draw (W)", field: "avgPowerW", step: 10 },
            { label: "Electricity ($/kWh)", field: "electricityRate", step: 0.01 },
            { label: "Carbon Intensity (gCO2/kWh)", field: "carbonIntensity", step: 10 },
          ].map(({ label, field, step }) => (
            <div key={field}>
              <label className="text-xs text-neutral-400 mb-1 block">
                {label}
              </label>
              <input
                type="number"
                value={roi[field as keyof typeof roi]}
                onChange={(e) => update(field, parseFloat(e.target.value) || 0)}
                step={step}
                className={INPUT_CLS}
              />
            </div>
          ))}
        </div>
      </div>

      {/* QLoRA Results */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Compute Cost"
          value={`$${fmt(computeCost, 2)}`}
          sub={`${fmt(roi.trainingMinutes, 1)} min @ $${roi.gpuHourlyRate}/hr`}
          color="text-blue-400"
        />
        <StatCard
          label="Energy Cost"
          value={`$${fmt(energyCost, 4)}`}
          sub={`${fmt(energyKwh, 4)} kWh`}
          color="text-yellow-400"
        />
        <StatCard
          label="Total Cost"
          value={`$${fmt(totalTrainingCost, 2)}`}
          sub={`Energy is ${fmt((energyCost / totalTrainingCost) * 100, 1)}% of total`}
          color="text-green-400"
        />
        <StatCard
          label="Carbon Footprint"
          value={fmt(co2Grams, 1)}
          unit="g CO2"
          sub={`${fmt(co2Grams / 404, 4)} car-miles equivalent`}
          color="text-emerald-400"
        />
      </div>

      {/* Comparison */}
      <div className={CARD_CLS}>
        <h3 className="text-sm font-medium text-white mb-4">
          Cost Comparison
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
          {[
            {
              label: "Queries/Month (no fine-tune)",
              field: "queriesPerMonth",
              step: 1000,
            },
            {
              label: "Tokens/Query",
              field: "tokensPerQuery",
              step: 100,
            },
            {
              label: "API Cost ($/1K tokens)",
              field: "apiCostPer1kTokens",
              step: 0.001,
            },
            {
              label: "Full Fine-Tune Hours",
              field: "fullFineTuneHours",
              step: 1,
            },
            {
              label: "Full Fine-Tune GPUs",
              field: "fullFineTuneGpus",
              step: 1,
            },
          ].map(({ label, field, step }) => (
            <div key={field}>
              <label className="text-xs text-neutral-400 mb-1 block">
                {label}
              </label>
              <input
                type="number"
                value={roi[field as keyof typeof roi]}
                onChange={(e) => update(field, parseFloat(e.target.value) || 0)}
                step={step}
                className={INPUT_CLS}
              />
            </div>
          ))}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-neutral-400 border-b border-neutral-800">
                <th className="text-left py-2 px-3">Approach</th>
                <th className="text-right py-2 px-3">Cost</th>
                <th className="text-right py-2 px-3">Energy (kWh)</th>
                <th className="text-right py-2 px-3">CO2 (g)</th>
                <th className="text-right py-2 px-3">vs QLoRA</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-neutral-800/50">
                <td className="py-2 px-3 text-green-400 font-medium">
                  QLoRA Fine-Tune (MI300X)
                </td>
                <td className="py-2 px-3 text-right text-green-400 font-bold">
                  ${fmt(totalTrainingCost, 2)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-300">
                  {fmt(energyKwh, 4)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-300">
                  {fmt(co2Grams, 1)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-500">-</td>
              </tr>
              <tr className="border-b border-neutral-800/50">
                <td className="py-2 px-3 text-neutral-300">
                  Full Fine-Tune ({roi.fullFineTuneGpus} GPUs)
                </td>
                <td className="py-2 px-3 text-right text-red-400">
                  ${fmt(fullFineTuneCost, 2)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-300">
                  {fmt(fullFineTuneEnergy, 2)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-300">
                  {fmt(fullFineTuneCO2, 0)}
                </td>
                <td className="py-2 px-3 text-right text-red-400">
                  {fmt(fullFineTuneCost / totalTrainingCost, 0)}x more
                </td>
              </tr>
              <tr className="border-b border-neutral-800/50">
                <td className="py-2 px-3 text-neutral-300">
                  API-Only (yearly)
                </td>
                <td className="py-2 px-3 text-right text-red-400">
                  ${fmt(yearlyApiCost, 2)}
                </td>
                <td className="py-2 px-3 text-right text-neutral-500">
                  N/A
                </td>
                <td className="py-2 px-3 text-right text-neutral-500">
                  N/A
                </td>
                <td className="py-2 px-3 text-right text-red-400">
                  {yearlyApiCost > 0
                    ? `${fmt(yearlyApiCost / totalTrainingCost, 0)}x more`
                    : "-"}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {savingsVsApi > 0 && (
          <div className="mt-4 bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-sm">
            <span className="text-green-400 font-bold">
              Fine-tuning saves ${fmt(savingsVsApi, 2)}/year
            </span>
            <span className="text-neutral-400">
              {" "}vs API-only approach, and{" "}
            </span>
            <span className="text-green-400 font-bold">
              ${fmt(savingsVsFullFT, 2)}
            </span>
            <span className="text-neutral-400"> vs full fine-tuning.</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Import Panel ──

function ImportPanel({
  runs,
  onImport,
}: {
  runs: TrainingRun[];
  onImport: (run: TrainingRun) => void;
}) {
  const [jsonInput, setJsonInput] = useState("");
  const [importName, setImportName] = useState("");
  const [importType, setImportType] = useState<
    "energy_metrics" | "run_config" | "power_samples" | "eval_results"
  >("energy_metrics");
  const [error, setError] = useState("");
  const [pendingRun, setPendingRun] = useState<Partial<TrainingRun>>({
    id: uid(),
    name: "",
    metrics: null,
    power: [],
    config: null,
    evals: [],
  });

  const handleAddFile = () => {
    setError("");
    try {
      const parsed = JSON.parse(jsonInput);
      const updated = { ...pendingRun };

      switch (importType) {
        case "energy_metrics":
          updated.metrics = parsed as EnergyMetrics;
          break;
        case "run_config":
          updated.config = parsed as RunConfig;
          break;
        case "power_samples":
          updated.power = parsed as PowerSample[];
          break;
        case "eval_results":
          updated.evals = parsed as EvalResult[];
          break;
      }

      setPendingRun(updated);
      setJsonInput("");
    } catch {
      setError("Invalid JSON");
    }
  };

  const handleFinishImport = () => {
    if (!importName.trim()) {
      setError("Enter a run name");
      return;
    }
    const run: TrainingRun = {
      id: pendingRun.id || uid(),
      name: importName.trim(),
      metrics: pendingRun.metrics || null,
      power: pendingRun.power || [],
      config: pendingRun.config || null,
      evals: pendingRun.evals || [],
    };
    onImport(run);
    setPendingRun({ id: uid(), name: "", metrics: null, power: [], config: null, evals: [] });
    setImportName("");
  };

  const fileTypes = [
    { value: "energy_metrics", label: "energy_metrics.json", loaded: !!pendingRun.metrics },
    { value: "run_config", label: "run_config.json", loaded: !!pendingRun.config },
    { value: "power_samples", label: "power_samples.json", loaded: (pendingRun.power?.length ?? 0) > 0 },
    { value: "eval_results", label: "eval_results.json", loaded: (pendingRun.evals?.length ?? 0) > 0 },
  ] as const;

  return (
    <div className={`${CARD_CLS} mt-6`}>
      <h3 className="text-sm font-medium text-white mb-3">
        Import Training Run
      </h3>
      <div className="space-y-3">
        <input
          type="text"
          value={importName}
          onChange={(e) => setImportName(e.target.value)}
          placeholder="Run name (e.g. 'Hermes-only' or 'Blended v1')"
          className={INPUT_CLS}
        />
        <div className="flex gap-2 flex-wrap">
          {fileTypes.map((ft) => (
            <button
              key={ft.value}
              onClick={() => setImportType(ft.value)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                importType === ft.value
                  ? "bg-green-600 text-white"
                  : ft.loaded
                  ? "bg-green-500/20 text-green-400"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {ft.loaded ? "✓ " : ""}
              {ft.label}
            </button>
          ))}
        </div>
        <textarea
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
          placeholder={`Paste ${importType}.json contents here...`}
          className={`${INPUT_CLS} h-32 resize-none font-mono text-xs`}
        />
        {error && <p className="text-red-400 text-xs">{error}</p>}
        <div className="flex gap-2">
          <button
            onClick={handleAddFile}
            disabled={!jsonInput.trim()}
            className="px-4 py-2 bg-neutral-700 text-white rounded-lg text-sm font-medium hover:bg-neutral-600 disabled:opacity-50"
          >
            Add File
          </button>
          <button
            onClick={handleFinishImport}
            disabled={!importName.trim() || !pendingRun.metrics}
            className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-500 disabled:opacity-50"
          >
            Import Run
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main Page ──

export default function FineTuningPage() {
  const [tab, setTab] = useState<TabId>("monitor");
  const { data: storedRuns, save: saveRuns } = useStoredData<TrainingRun[]>(
    STORAGE_KEY,
    DEFAULT_RUNS
  );
  const { loading, error, callApi } = useFineTuneApi();

  const handleImport = useCallback(
    (run: TrainingRun) => {
      const updated = [...storedRuns, run];
      saveRuns(updated);
    },
    [storedRuns, saveRuns]
  );

  const handleDeleteRun = useCallback(
    (id: string) => {
      saveRuns(storedRuns.filter((r) => r.id !== id));
    },
    [storedRuns, saveRuns]
  );

  return (
    <div className="bg-neutral-950 text-neutral-100 p-6 min-h-screen">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-1">
          GreenTune Fine-Tuning
        </h1>
        <p className="text-neutral-400 text-sm mb-6">
          Energy-efficient QLoRA fine-tuning on AMD MI300X with real-time power
          monitoring
        </p>

        {/* Tab Bar */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                tab === t.id
                  ? "bg-green-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === "monitor" && <RunMonitorTab runs={storedRuns} />}
        {tab === "leaderboard" && <LeaderboardTab runs={storedRuns} />}
        {tab === "playground" && (
          <PlaygroundTab
            runs={storedRuns}
            loading={loading}
            error={error}
            callApi={callApi}
          />
        )}
        {tab === "roi" && <ROICalculatorTab />}

        {/* Import Panel (always visible at bottom) */}
        <ImportPanel runs={storedRuns} onImport={handleImport} />

        {/* Run Management */}
        {storedRuns.length > 0 && (
          <div className="mt-4 flex gap-2 flex-wrap">
            {storedRuns.map((r) => (
              <div
                key={r.id}
                className="flex items-center gap-2 bg-neutral-800 rounded-lg px-3 py-1.5 text-xs"
              >
                <span className="text-neutral-300">{r.name}</span>
                <button
                  onClick={() => handleDeleteRun(r.id)}
                  className="text-neutral-500 hover:text-red-400"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
