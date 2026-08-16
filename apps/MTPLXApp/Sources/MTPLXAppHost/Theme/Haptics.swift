import AppKit

// MARK: - Haptics
//
// Tiny wrapper around `NSHapticFeedbackManager` so callsites read as
// "tick" intent rather than "perform a feedback pattern of type X with
// performanceTime Y". The two patterns we actually use:
//
//   - `.alignment`   → a soft, micro click. Used for high-resolution
//                      sliders where every step crossing is small
//                      (Temperature 0.05, Top K 5).
//   - `.levelChange` → a firm, deliberate click. Used for the MTP
//                      depth slider whose three positions (D1, D2, D3)
//                      are each meaningful structural changes — the
//                      user should feel them clearly.
//
// Both patterns honour the user's system-wide haptic setting in
// System Settings ▸ Trackpad ▸ Force Click & haptic feedback. We never
// fall back to alerts or beeps when haptics are unavailable.

enum Haptics {
    /// Fire a single tick using the supplied pattern. Defaults to the
    /// micro alignment tick — call sites for big detents should pass
    /// `.levelChange` explicitly so the intent is visible at the call.
    static func tick(_ pattern: NSHapticFeedbackManager.FeedbackPattern = .alignment) {
        NSHapticFeedbackManager.defaultPerformer.perform(
            pattern,
            performanceTime: .now
        )
    }
}
