import SwiftUI
import MTPLXAppCore

// MARK: - MenuBarMiniGauge
//
// Compact version of `GaugeView` for the menubar popover. ~96pt diameter,
// always TPS mode (no prefill morph — the menubar is a glance surface,
// not the hero). Inherits the same Core Animation rendering so it
// remains GPU-cheap on idle.

struct MenuBarMiniGauge: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        let rolling = backend.rolling
        let stickyMax = rolling?.stickyAllTimeMax ?? 0
        let ceiling = Swift.max(80, ceil(stickyMax / 20) * 20)
        // Same warm-up gate + held source as the hero gauge: read the
        // gated `headlineDecode` (not raw `displayDecodeTokS`) so the
        // menubar dial never flickers and never spawns at the ~30 tok/s
        // warm-up reading before a real request has run.
        let hasRealDecode = backend.observedCompletionCount > 0
            || backend.inFlight.contains { $0.hasDecodeProgress }
        let decode = hasRealDecode ? (backend.headlineDecode.value ?? 0) : 0
        // Mirror the hero gauge lifecycle so the menubar dial shows the
        // same loading spin while the model warms up and the same
        // speedtest number once it's running — not a dead dim disc.
        let mode: GaugeMode = {
            switch backend.daemonState.kind {
            case .stopped, .crashed: return .dim
            case .degraded: return .degraded
            case .starting, .warming, .stopping: return .loading(phase: .starting)
            case .running: return .tps(decode: decode, max: ceiling)
            }
        }()

        GaugeView(
            mode: mode,
            stickyMax: ceiling > 0 ? stickyMax / ceiling : 0,
            ceiling: 1.0,
            motionEnabled: !backend.configuration.performanceLock && !themeStore.reduceMotionPreference,
            diameter: 96
        )
    }
}
