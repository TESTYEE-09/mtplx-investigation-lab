import { BigNumber, Card } from "./Card";
import { PrefillPanel } from "./PrefillPanel";
import { TPSGauge } from "./TPSGauge";
import { TPSTimeSeries } from "./TPSTimeSeries";
import { TokensServedTile } from "./TokensServedTile";
import { fmtNumber, fmtSeconds, fmtTokS } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function OverviewTab() {
  const latest = useDashboardStore((s) => s.latest);
  const inFlight = useDashboardStore((s) => s.inFlight);
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const contextWindow = useDashboardStore((s) => s.contextWindow);

  const contextLen = latest?.context_len ?? 0;
  const contextPct = contextWindow ? Math.min(100, (contextLen / contextWindow) * 100) : 0;

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-5">
        <TPSGauge />
      </div>
      <div className="col-span-12 lg:col-span-7">
        <TPSTimeSeries />
      </div>

      {/* Prefill panel: hero status while the model is chewing the prompt
       *  (decode TPS is meaningless during this window). Full-width so the
       *  progress bar is impossible to miss. */}
      <div className="col-span-12">
        <PrefillPanel />
      </div>

      <div className="col-span-12 lg:col-span-4">
        <TokensServedTile />
      </div>

      <div className="col-span-6 lg:col-span-4">
        <Card title="In flight">
          <BigNumber
            value={fmtNumber(inFlight.length)}
            unit="requests"
            tone={inFlight.length > 0 ? "accent" : "default"}
            caption={
              inFlight.length === 0
                ? "idle · waiting for next request"
                : `${inFlight.length} active · oldest ${fmtSeconds(
                    Math.max(...inFlight.map((h) => h.age_s)),
                  )}`
            }
          />
        </Card>
      </div>

      <div className="col-span-6 lg:col-span-4">
        <Card
          title="Cache + context"
          subtitle={
            sessionBank
              ? `${sessionBank.prefixes?.length ?? 0} of ${sessionBank.max_entries} slots`
              : "—"
          }
        >
          <BigNumber
            value={`${contextPct.toFixed(0)}%`}
            unit="context used"
            tone={contextPct >= 75 ? "warm" : contextPct >= 95 ? "hot" : "cool"}
            caption={`${fmtNumber(contextLen)} / ${fmtNumber(contextWindow)} tokens`}
          />
        </Card>
      </div>

      <div className="col-span-12 lg:col-span-6">
        <Card title="Last request" subtitle="from /metrics latest">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Field label="decode tok/s" value={fmtTokS(latest?.decode_tok_s)} highlight />
            <Field label="ttft" value={fmtSeconds(latest?.ttft_s)} />
            <Field label="prompt eval" value={fmtSeconds(latest?.prompt_eval_time_s)} />
            <Field label="decode" value={fmtSeconds(latest?.decode_elapsed_s)} />
            <Field
              label="prefill tok/s"
              value={fmtTokS(latest?.prefill_tok_s)}
            />
            <Field
              label="cached"
              value={`${fmtNumber(latest?.cached_tokens)} / ${fmtNumber(latest?.prompt_tokens)}`}
            />
          </div>
        </Card>
      </div>

      <div className="col-span-12 lg:col-span-6">
        <Card title="Session" subtitle="from latest envelope">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Field
              label="session id"
              value={latest?.session_id ? latest.session_id : "—"}
            />
            <Field
              label="cache hit"
              value={latest?.session_cache_hit ? "yes" : "no"}
              highlight={Boolean(latest?.session_cache_hit)}
            />
            <Field
              label="restore mode"
              value={latest?.session_restore_mode ?? "—"}
            />
            <Field
              label="miss reason"
              value={latest?.cache_miss_reason ?? "—"}
            />
            <Field label="mtp depth" value={fmtNumber(latest?.mtp_depth)} />
            <Field label="verify calls" value={fmtNumber(latest?.verify_calls)} />
          </div>
        </Card>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="bg-[var(--bg-elevated)] rounded-md px-3 py-2 border border-[var(--border-soft)]">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div
        className={
          "text-sm font-semibold tabular-nums " +
          (highlight ? "text-[var(--accent)]" : "text-[var(--text-primary)]")
        }
      >
        {value}
      </div>
    </div>
  );
}
