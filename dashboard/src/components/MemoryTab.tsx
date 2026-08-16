import { BigNumber, Card } from "./Card";
import { HardwareBanner } from "./HardwareBanner";
import { MemoryStackedBar } from "./MemoryStackedBar";
import { fmtBytes } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function MemoryTab() {
  const mem = useDashboardStore((s) => s.mem);
  const latest = useDashboardStore((s) => s.latest);
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <HardwareBanner />
      </div>
      <div className="col-span-12">
        <MemoryStackedBar />
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="Active memory" subtitle="MLX active allocation">
          <BigNumber
            value={fmtBytes(mem?.active_memory_bytes ?? null)}
            tone="accent"
            caption="live MLX accessor"
          />
        </Card>
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="Cache memory" subtitle="MLX cache allocator">
          <BigNumber
            value={fmtBytes(mem?.cache_memory_bytes ?? null)}
            tone="cool"
            caption="reusable buffer cache"
          />
        </Card>
      </div>
      <div className="col-span-12 sm:col-span-6 lg:col-span-4">
        <Card title="Peak memory" subtitle="highest seen this process">
          <BigNumber
            value={fmtBytes(
              Math.max(
                Number(mem?.peak_memory_bytes ?? 0),
                Number(latest?.peak_memory_bytes ?? 0),
              ) || null,
            )}
            tone="warm"
            caption="includes last-request peak"
          />
        </Card>
      </div>
    </div>
  );
}
