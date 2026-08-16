import { useEffect, useRef } from "react";
import confetti from "canvas-confetti";
import { useDashboardStore } from "../state/store";

const COOLDOWN_MS = 1200;

export function useNewMaxTPSDetector(): {
  newMaxBanner: { tok_s: number; when_s: number } | null;
} {
  const event = useDashboardStore((s) => s.newMaxTPSEvent);
  const consume = useDashboardStore((s) => s.consumeNewMaxTPS);
  const soundEnabled = useDashboardStore((s) => s.soundEnabled);
  const lastFiredRef = useRef(0);

  useEffect(() => {
    if (!event) return;
    const now = Date.now();
    if (now - lastFiredRef.current < COOLDOWN_MS) return;
    lastFiredRef.current = now;

    try {
      confetti({
        particleCount: 90,
        spread: 70,
        origin: { y: 0.25 },
        colors: ["#00d68f", "#f0b429", "#4fb6f3", "#ffffff"],
      });
    } catch {
      // ignore in non-DOM contexts
    }

    if (soundEnabled && typeof window !== "undefined") {
      try {
        const ctx = new (window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = 880;
        g.gain.value = 0.05;
        o.connect(g).connect(ctx.destination);
        o.start();
        o.frequency.exponentialRampToValueAtTime(1320, ctx.currentTime + 0.18);
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
        o.stop(ctx.currentTime + 0.42);
      } catch {
        // ignore — audio is optional eye candy
      }
    }

    const timer = window.setTimeout(consume, 3500);
    return () => window.clearTimeout(timer);
  }, [event, consume, soundEnabled]);

  return { newMaxBanner: event };
}
