import { AlertTriangle } from "lucide-react";
import { useDashboardStore } from "../state/store";

const FAN_RAMP_THRESHOLD_RPM = 4000;

export function UniversalThermalRuleBanner() {
  const thermal = useDashboardStore((s) => s.thermal);
  const active = useDashboardStore((s) => s.inFlight.length);
  if (active === 0) return null;
  if (!thermal || !thermal.ok) {
    return (
      <Banner>
        Thermal polling is disabled but a request is in flight. Per the project's
        Universal Thermal Rule, model work should run under verified max-fan
        mode for honest benchmark numbers.
      </Banner>
    );
  }
  if ((thermal.max_rpm ?? 0) < FAN_RAMP_THRESHOLD_RPM) {
    return (
      <Banner>
        Fans are below {FAN_RAMP_THRESHOLD_RPM} RPM while generation is active.
        Per the Universal Thermal Rule, verify max-fan ramp before treating any
        TPS reading as a real benchmark.
      </Banner>
    );
  }
  return null;
}

function Banner({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-[var(--accent-warm)]/50 bg-[var(--accent-warm)]/10 px-4 py-3 flex items-start gap-3">
      <AlertTriangle className="size-4 text-[var(--accent-warm)] shrink-0 mt-0.5" />
      <p className="text-sm text-[var(--accent-warm)] leading-relaxed">{children}</p>
    </div>
  );
}
