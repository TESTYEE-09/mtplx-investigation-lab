import type { ReactNode } from "react";
import { cn } from "../lib/utils";

type CardProps = {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
};

export function Card({ title, subtitle, action, className, bodyClassName, children }: CardProps) {
  return (
    <section
      className={cn(
        "rounded-2xl border border-[var(--border-soft)] bg-[var(--bg-card)] shadow-[inset_0_1px_0_0_rgba(255,255,255,0.02)] overflow-hidden",
        className,
      )}
    >
      {(title || action) && (
        <header className="px-5 pt-4 pb-2 flex items-start justify-between gap-4">
          <div className="min-w-0">
            {title ? (
              <h3 className="text-sm font-semibold text-[var(--text-primary)] tracking-tight">
                {title}
              </h3>
            ) : null}
            {subtitle ? (
              <p className="text-xs text-[var(--text-muted)] mt-0.5">{subtitle}</p>
            ) : null}
          </div>
          {action ? <div className="shrink-0">{action}</div> : null}
        </header>
      )}
      <div className={cn("px-5 pb-5 pt-2", bodyClassName)}>{children}</div>
    </section>
  );
}

export function BigNumber({
  value,
  unit,
  caption,
  tone = "default",
}: {
  value: string | number;
  unit?: string;
  caption?: ReactNode;
  tone?: "default" | "accent" | "warm" | "hot" | "cool";
}) {
  const colorClass =
    tone === "accent"
      ? "text-[var(--accent)]"
      : tone === "warm"
        ? "text-[var(--accent-warm)]"
        : tone === "hot"
          ? "text-[var(--accent-hot)]"
          : tone === "cool"
            ? "text-[var(--accent-cool)]"
            : "text-[var(--text-primary)]";
  return (
    <div>
      <div className={cn("flex items-baseline gap-2", colorClass)}>
        <span className="text-4xl font-semibold tabular-nums leading-none">{value}</span>
        {unit ? <span className="text-sm text-[var(--text-muted)]">{unit}</span> : null}
      </div>
      {caption ? (
        <div className="text-xs text-[var(--text-muted)] mt-2">{caption}</div>
      ) : null}
    </div>
  );
}
