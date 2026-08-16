import SwiftUI

/// Soft outer chrome glow that fades 0 → 1 → 0 on a 1.6s cycle while
/// the tile is `.running`. Mirrors the gauge's "new max" pulse
/// pattern — the trick that makes a running surface read as "alive"
/// without burning real energy.
struct BenchTilePulseModifier: ViewModifier {
    let active: Bool
    let motionEnabled: Bool

    func body(content: Content) -> some View {
        if active && motionEnabled {
            content.phaseAnimator([0.0, 1.0, 0.0]) { view, phase in
                view.shadow(
                    color: Brand.accentChrome.opacity(0.30 * phase),
                    radius: 10 * phase,
                    x: 0,
                    y: 0
                )
            } animation: { _ in
                .smooth(duration: 0.8)
            }
        } else {
            content
        }
    }
}
