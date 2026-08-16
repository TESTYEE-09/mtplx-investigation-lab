import { Cpu, MemoryStick, Sparkles } from "lucide-react";
import { Card } from "./Card";
import { fmtBytes } from "../lib/utils";
import { useDashboardStore } from "../state/store";

function chipFromModel(machineModel: string | null | undefined): string {
  if (!machineModel) return "Apple Silicon";
  const id = machineModel.toLowerCase();
  if (id.includes("mac17")) return "M5 Max";
  if (id.includes("mac16")) return "M3 Ultra";
  if (id.includes("mac15")) return "M4";
  if (id.includes("mac14")) return "M3";
  if (id.includes("mac13")) return "M2";
  return "Apple Silicon";
}

export function HardwareBanner() {
  const machine = useDashboardStore((s) => s.machine);
  const profileName = useDashboardStore((s) => s.profileName);
  const modelId = useDashboardStore((s) => s.modelId);
  const contextWindow = useDashboardStore((s) => s.contextWindow);
  const chipBadge = chipFromModel(machine?.machine_model);

  return (
    <Card title="Hardware" subtitle={machine?.machine_model ?? "unknown machine model"}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Tile
          icon={<Cpu className="size-4 text-[var(--accent)]" />}
          label="chip"
          value={chipBadge}
        />
        <Tile
          icon={<MemoryStick className="size-4 text-[var(--accent-cool)]" />}
          label="unified memory"
          value={fmtBytes(machine?.unified_memory_bytes ?? null)}
        />
        <Tile
          icon={<Sparkles className="size-4 text-[var(--accent-warm)]" />}
          label="profile"
          value={profileName ?? "—"}
        />
        <Tile
          icon={<Cpu className="size-4 text-[var(--text-muted)]" />}
          label="context window"
          value={contextWindow ? `${contextWindow.toLocaleString()} tok` : "—"}
        />
      </div>
      <div className="mt-3 text-xs text-[var(--text-muted)] truncate">
        loaded model: <span className="text-[var(--text-primary)]">{modelId ?? "—"}</span>
      </div>
    </Card>
  );
}

function Tile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {icon}
        {label}
      </div>
      <div className="text-base font-semibold text-[var(--text-primary)] mt-1 truncate">
        {value}
      </div>
    </div>
  );
}
