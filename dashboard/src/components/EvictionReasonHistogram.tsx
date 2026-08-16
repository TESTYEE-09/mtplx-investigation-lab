import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

// Mirrors `CacheMissReason` enum (mtplx/session_bank.py:29-39).
const REASON_TOOLTIPS: Record<string, string> = {
  POLICY_MISMATCH:
    "Sampling/depth knobs changed between requests; the cached state is no longer policy-equivalent.",
  TEMPLATE_MISMATCH: "Chat template changed (system prompt, role tags); cache cannot be re-used.",
  MODEL_MISMATCH: "Different model loaded since the cached prefix was created.",
  SNAPSHOT_DESYNC: "Engine state diverged from the cached snapshot.",
  EVICTED: "LRU eviction made room for a newer prefix.",
  SESSION_BUSY: "Another request currently holds this session's lease.",
  NEW_SESSION: "First request for this session id; no prior prefix to reuse.",
};

export function EvictionReasonHistogram() {
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const log = sessionBank?.eviction_log ?? [];
  const counts: Record<string, number> = {};
  for (const entry of log) {
    counts[entry.reason] = (counts[entry.reason] ?? 0) + 1;
  }
  const rows = Object.entries(counts)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <Card
      title="Eviction reasons · last 16"
      subtitle={
        sessionBank?.last_miss_reason
          ? `most recent: ${sessionBank.last_miss_reason}`
          : "no evictions yet"
      }
    >
      <div className="h-[220px]">
        {rows.length === 0 ? (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            SessionBank stable · no evictions
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="reason"
                stroke="rgba(200,210,220,0.6)"
                tick={{ fontSize: 10 }}
                interval={0}
              />
              <YAxis stroke="rgba(200,210,220,0.6)" allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  fontSize: 12,
                  maxWidth: 320,
                }}
                labelFormatter={(label) => (
                  <span className="text-[var(--text-primary)] font-semibold">
                    {String(label)}
                  </span>
                )}
                // Recharts' Formatter generics fight a literal "count" name;
                // returning unknown[] sidesteps the constraint and the runtime
                // shape is still a [label, name] tuple.
                formatter={((value: unknown, _name: unknown, item: { payload?: { reason?: string } }) => {
                  const reason = String(item?.payload?.reason ?? "");
                  const tooltip = REASON_TOOLTIPS[reason] ?? "Cache eviction reason.";
                  return [`${value} · ${tooltip}`, "count"];
                }) as never}
              />
              <Bar dataKey="count" fill="rgba(240,180,41,0.85)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
