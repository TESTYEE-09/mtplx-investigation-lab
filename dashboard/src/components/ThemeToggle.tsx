import { Palette } from "lucide-react";
import { useDashboardStore } from "../state/store";

export function ThemeToggle() {
  const theme = useDashboardStore((s) => s.theme);
  const cycleTheme = useDashboardStore((s) => s.cycleTheme);
  return (
    <button
      onClick={cycleTheme}
      title={`Theme: ${theme} (press T to cycle)`}
      className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
    >
      <Palette className="size-4" />
      <span className="hidden lg:inline">{theme}</span>
    </button>
  );
}
