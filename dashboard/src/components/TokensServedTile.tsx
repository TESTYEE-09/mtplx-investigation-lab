import { BigNumber, Card } from "./Card";
import { fmtNumber, fmtSeconds } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function TokensServedTile() {
  const lifetime = useDashboardStore((s) => s.lifetime);
  if (!lifetime) {
    return (
      <Card title="Tokens served · this server">
        <BigNumber value="—" caption="waiting for first request" />
      </Card>
    );
  }
  return (
    <Card title="Tokens served · this server">
      <BigNumber
        value={fmtNumber(lifetime.tokens_total)}
        unit="tokens"
        tone="accent"
        caption={
          <div className="space-y-1">
            <div>
              {fmtNumber(lifetime.requests_total)} requests since {fmtSeconds(lifetime.uptime_s)} ago
            </div>
            <div className="text-[var(--text-muted)]">
              prompt: {fmtNumber(lifetime.prompt_tokens_total)} ·{" "}
              completion: {fmtNumber(lifetime.completion_tokens_total)} ·{" "}
              cached: {fmtNumber(lifetime.cached_tokens_total)}
            </div>
            {lifetime.cancelled_total > 0 ? (
              <div className="text-[var(--accent-warm)] text-xs">
                {fmtNumber(lifetime.cancelled_total)} cancelled
              </div>
            ) : null}
          </div>
        }
      />
    </Card>
  );
}
