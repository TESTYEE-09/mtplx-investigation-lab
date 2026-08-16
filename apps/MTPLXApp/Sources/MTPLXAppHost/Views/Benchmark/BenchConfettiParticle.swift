import SwiftUI

// MARK: - BenchConfettiParticle
//
// One confetti particle's deterministic seed: origin x-fraction, phase,
// color, speed, spin, and horizontal drift rate. The particle is
// drawn each frame by `BenchConfettiView`'s Canvas with these values
// interpolated against the timeline progress.
//
// Palette is Jet Chrome — success green, polished chromeAccent,
// warning amber, off-white type body. No cool-blue: the celebration
// reads as "polished metal and success" instead of "AI app celebrating."

struct BenchConfettiParticle {
    var originX: Double          // 0..1 fraction of width
    var phase: Double            // 0..2π
    var color: Color
    var speed: Double            // 0.8..1.3
    var spin: Double             // -1..1
    var driftRate: Double        // 1..3

    static func random() -> BenchConfettiParticle {
        let colors: [Color] = [
            Brand.success,
            Brand.accentChrome,
            Brand.warning,
            Brand.typeBody,
        ]
        return BenchConfettiParticle(
            originX: Double.random(in: 0...1),
            phase: Double.random(in: 0...(2 * .pi)),
            color: colors.randomElement() ?? Brand.accentChrome,
            speed: Double.random(in: 0.8...1.3),
            spin: Double.random(in: -1...1),
            driftRate: Double.random(in: 1...3)
        )
    }
}
