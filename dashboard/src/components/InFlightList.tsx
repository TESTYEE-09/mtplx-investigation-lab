import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Square } from "lucide-react";
import { api } from "../lib/api";
import { Card } from "./Card";
import { fmtNumber, fmtSeconds, truncateMiddle } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function InFlightList() {
  const inFlight = useDashboardStore((s) => s.inFlight);
  const queryClient = useQueryClient();
  const cancel = useMutation({
    mutationFn: (requestId: string) => api.postCancel(requestId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["metrics"] });
    },
  });

  return (
    <Card
      title="In-flight requests"
      subtitle={
        inFlight.length === 0
          ? "no active generations"
          : `${inFlight.length} active · cancel is best-effort`
      }
    >
      {inFlight.length === 0 ? (
        <div className="text-sm text-[var(--text-muted)]">
          Drive load from any client (Web UI, hippo, OpenAI SDK) to see live
          requests here.
        </div>
      ) : (
        <ul className="divide-y divide-[var(--border-soft)] -mx-2">
          {inFlight.map((handle) => {
            const progress = handle.last_progress as {
              completion_tokens?: number;
              decode_tok_s?: number;
            };
            const tokens = progress?.completion_tokens ?? 0;
            const tokS = progress?.decode_tok_s;
            return (
              <motion.li
                key={handle.request_id}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="px-2 py-3 grid grid-cols-[1fr_auto] items-center gap-3"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                    <span className="font-mono truncate">{truncateMiddle(handle.request_id, 28)}</span>
                    {handle.session_id ? (
                      <span className="text-[10px] uppercase tracking-wider text-[var(--accent-cool)]">
                        {truncateMiddle(handle.session_id, 16)}
                      </span>
                    ) : null}
                  </div>
                  <div className="text-sm text-[var(--text-primary)] truncate">
                    {handle.prompt_preview || "—"}
                  </div>
                  <div className="text-xs text-[var(--text-muted)] flex flex-wrap gap-x-3 mt-1">
                    <span>age {fmtSeconds(handle.age_s)}</span>
                    <span>{fmtNumber(tokens)} tok</span>
                    {typeof tokS === "number" && tokS > 0 ? (
                      <span className="text-[var(--accent)]">
                        {tokS.toFixed(1)} tok/s
                      </span>
                    ) : null}
                  </div>
                </div>
                <button
                  type="button"
                  className="inline-flex items-center gap-1.5 text-xs text-[var(--accent-hot)] hover:text-[var(--accent-hot)] hover:bg-[var(--accent-hot)]/10 rounded px-2 py-1 disabled:opacity-50"
                  onClick={() => cancel.mutate(handle.request_id)}
                  disabled={cancel.isPending || handle.cancelled}
                >
                  <Square className="size-3" />
                  {handle.cancelled ? "cancelling" : "cancel"}
                </button>
              </motion.li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
