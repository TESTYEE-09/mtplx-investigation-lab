import SwiftUI

// MARK: - BenchConfettiView
//
// Inline SwiftUI confetti (no SPM dep). Fires once on a 30/30 perfect
// run. 50 particles falling from the top of the summary card, fading
// out as they drop. Pure SwiftUI `TimelineView(.animation)` driven;
// respects Reduce Motion.
//
// Jet Chrome pass: the GeometryReader inside the timeline tick is
// replaced with a Canvas driven by `containerRelativeFrame` so the
// particles are drawn in a single GPU pass per frame instead of
// laying out 50 SwiftUI Rectangle views on every animation tick.
// Palette swap: cool-blue out, polished chromeAccent solid in.

struct BenchConfettiView: View {
    let start: Date
    private static let particles: [BenchConfettiParticle] = (0..<50).map { _ in
        BenchConfettiParticle.random()
    }

    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        if themeStore.reduceMotionPreference {
            EmptyView()
        } else {
            TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { context in
                let t = context.date.timeIntervalSince(start)
                if t < 0 || t > duration {
                    Color.clear
                } else {
                    Canvas { canvasContext, size in
                        for p in Self.particles {
                            draw(particle: p, t: t, into: canvasContext, size: size)
                        }
                    }
                }
            }
            .containerRelativeFrame([.horizontal, .vertical])
        }
    }

    private let duration: Double = 2.6

    private func draw(particle p: BenchConfettiParticle, t: Double, into context: GraphicsContext, size: CGSize) {
        let progress = min(1, t / duration)
        let fall = size.height * progress * p.speed
        let drift = sin(progress * .pi * p.driftRate + p.phase) * size.width * 0.12
        let rotation = Angle.degrees(progress * 360 * p.spin)
        let opacity: Double = progress > 0.85
            ? max(0, 1 - (progress - 0.85) / 0.15)
            : 1
        let x = p.originX * size.width + drift
        let y = -20 + fall

        var ctx = context
        ctx.opacity = opacity
        ctx.translateBy(x: x, y: y)
        ctx.rotate(by: rotation)
        let rect = CGRect(x: -3, y: -5, width: 6, height: 10)
        ctx.fill(Path(rect), with: .color(p.color))
    }
}
