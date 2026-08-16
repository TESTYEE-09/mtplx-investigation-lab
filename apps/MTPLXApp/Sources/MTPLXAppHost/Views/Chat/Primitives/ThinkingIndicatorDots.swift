import SwiftUI
import MTPLXAppCore

// MARK: - ThinkingIndicatorDots
//
// Three-dot pulse used next to "Thinking" / "Searching the web" /
// "Solving" titles while a tool call, reasoning stream, or benchmark
// problem is in flight.
//
// The cadence used to be driven by a `Timer.publish(...)` stored as a
// plain `let`. Any parent that re-rendered faster than the 0.18s tick
// (the benchmark live card flushes streamed reasoning every 80ms)
// recreated the publisher and re-subscribed before it ever fired, so
// the dots froze on screen. This version self-animates with a single
// `.repeatForever` animation kicked off once in `.onAppear`, which is
// immune to parent re-renders, and is suppressed under Reduce Motion.

struct ThinkingIndicatorDots: View {
    var color: Color = Brand.typeSecondary
    var size: CGFloat = 4
    var spacing: CGFloat = 3

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animating = false

    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(color)
                    .frame(width: size, height: size)
                    .opacity(animating ? 0.95 : 0.35)
                    .scaleEffect(animating ? 1.0 : 0.78)
                    .animation(
                        reduceMotion
                            ? nil
                            : .easeInOut(duration: 0.5)
                                .repeatForever(autoreverses: true)
                                .delay(Double(index) * 0.16),
                        value: animating
                    )
            }
        }
        .onAppear { if !reduceMotion { animating = true } }
        .accessibilityLabel("Working")
    }
}
