import { useEffect, useRef } from "react";
import type { ThermalFan } from "../lib/types";
import { Card } from "./Card";
import { fmtNumber } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function FanRingPanel() {
  const thermal = useDashboardStore((s) => s.thermal);

  if (!thermal || !thermal.ok || thermal.fans.length === 0) {
    return (
      <Card title="Fan rings" subtitle="thermal polling disabled or unavailable">
        <p className="text-sm text-[var(--text-muted)]">
          Pass <code>--enable-thermal-poll</code> when starting the MTPLX
          server to populate live fan RPMs. The poll uses
          <code> thermalforge status</code> at 1 Hz and is off by default to
          keep the hot path clean.
        </p>
      </Card>
    );
  }

  return (
    <Card
      title="Fan rings"
      subtitle={`min ${fmtNumber(thermal.min_rpm)} RPM · max ${fmtNumber(thermal.max_rpm)} RPM`}
    >
      <div className="grid grid-cols-2 gap-4">
        {thermal.fans.map((fan, idx) => (
          <FanRing key={idx} index={idx} fan={fan} />
        ))}
      </div>
    </Card>
  );
}

function FanRing({ index, fan }: { index: number; fan: ThermalFan }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const actual = Number(fan.actual_rpm ?? fan.rpm ?? 0);
  const target = Number(fan.target_rpm ?? actual);
  const capacity = Math.max(1, Number(fan.max_capacity_rpm ?? 7800));
  const mode = String(fan.mode ?? "auto");
  const ratio = Math.min(1, actual / capacity);
  const targetRatio = Math.min(1, target / capacity);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const size = 140;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);
    const cx = size / 2;
    const cy = size / 2;
    const radius = 56;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const sweep = endAngle - startAngle;
    // background ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.stroke();
    // actual
    const actualEnd = startAngle + sweep * ratio;
    const color = ratio > 0.7 ? "rgba(240,88,106,0.9)" : ratio > 0.4 ? "rgba(240,180,41,0.9)" : "rgba(0,214,143,0.9)";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, actualEnd);
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;
    // target tick
    const tickAngle = startAngle + sweep * targetRatio;
    ctx.beginPath();
    const inner = radius - 10;
    const outer = radius + 6;
    ctx.moveTo(cx + Math.cos(tickAngle) * inner, cy + Math.sin(tickAngle) * inner);
    ctx.lineTo(cx + Math.cos(tickAngle) * outer, cy + Math.sin(tickAngle) * outer);
    ctx.strokeStyle = "rgba(255,255,255,0.65)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [actual, target, capacity, ratio, targetRatio]);

  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3 grid place-items-center">
      <div className="relative">
        <canvas ref={canvasRef} aria-hidden="true" />
        <div className="absolute inset-0 grid place-items-center pointer-events-none">
          <div className="text-center">
            <div className="text-2xl font-semibold tabular-nums text-[var(--text-primary)]">
              {fmtNumber(actual)}
            </div>
            <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] -mt-1">
              RPM
            </div>
          </div>
        </div>
      </div>
      <div className="mt-2 text-xs text-[var(--text-muted)] text-center">
        F{index} · {mode}{" "}
        <span className="text-[var(--text-primary)]">
          / {fmtNumber(capacity)} max
        </span>
      </div>
    </div>
  );
}
