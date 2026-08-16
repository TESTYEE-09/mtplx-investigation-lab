import { Volume2, VolumeX } from "lucide-react";
import { useDashboardStore } from "../state/store";

export function SoundToggle() {
  const enabled = useDashboardStore((s) => s.soundEnabled);
  const toggle = useDashboardStore((s) => s.toggleSound);
  return (
    <button
      onClick={toggle}
      title={enabled ? "Mute new-max chime (S)" : "Enable new-max chime (S)"}
      className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] inline-flex items-center"
    >
      {enabled ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}
    </button>
  );
}
