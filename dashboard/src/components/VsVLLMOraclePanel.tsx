import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

// vLLM Qwen3.6-27B MTP-5 oracle, 2026-04-29 Phase 1 v4 (per BREAKTHROUGHS.md).
// Per-position acceptance — depth 1..5.
const VLLM_ORACLE = [0.927, 0.77, 0.63, 0.509, 0.43];

function isQwen36Match(modelId: string | null | undefined): boolean {
  if (!modelId) return false;
  const id = modelId.toLowerCase();
  return id.includes("qwen3.6-27b") || id.includes("qwen36-27b");
}

export function VsVLLMOraclePanel() {
  const modelId = useDashboardStore((s) => s.modelId);
  const latest = useDashboardStore((s) => s.latest);
  const meanAccept = latest?.mean_accept_probability_by_depth ?? [];
  const isMatch = isQwen36Match(modelId);

  if (!isMatch) {
    return (
      <Card
        title="vs vLLM oracle"
        subtitle="hardcoded baseline: Qwen3.6-27B MTP-5 only"
      >
        <div className="text-sm text-[var(--text-muted)] leading-relaxed">
          The vs-vLLM panel is gated on the Qwen3.6-27B family because the
          oracle baseline (per <code>BREAKTHROUGHS.md</code>, 2026-04-29
          Phase&nbsp;1&nbsp;v4) was measured on that exact model. The currently
          loaded model is <span className="text-[var(--text-primary)]">{modelId ?? "—"}</span>,
          so we render an empty state instead of a misleading comparison.
        </div>
      </Card>
    );
  }

  const rows = Array.from({ length: 5 }, (_, i) => ({
    depth: `D${i + 1}`,
    mtplx: (meanAccept[i] ?? 0) * 100,
    vllm: (VLLM_ORACLE[i] ?? 0) * 100,
  }));
  const hasMtplxData = meanAccept.length > 0;

  return (
    <Card
      title="vs vLLM oracle · Qwen3.6-27B"
      subtitle="MTPLX CyanKiwiMTP D4 vs vLLM MTP-5 Phase 1 v4 (2026-04-29)"
    >
      <div className="h-[260px]">
        {hasMtplxData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="depth" stroke="rgba(200,210,220,0.6)" />
              <YAxis
                stroke="rgba(200,210,220,0.6)"
                tickFormatter={(v) => `${v}%`}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                }}
                formatter={(v) =>
                  typeof v === "number" ? `${v.toFixed(1)}%` : String(v)
                }
              />
              <Legend
                wrapperStyle={{ color: "var(--text-muted)", fontSize: 12 }}
              />
              <Bar dataKey="mtplx" name="MTPLX" fill="rgba(0,214,143,0.9)" radius={[6, 6, 0, 0]} />
              <Bar dataKey="vllm" name="vLLM oracle" fill="rgba(79,182,243,0.65)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            Run a Qwen3.6 generation to populate the comparison.
          </div>
        )}
      </div>
    </Card>
  );
}
