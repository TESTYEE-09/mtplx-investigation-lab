import SwiftUI

// MARK: - MTPLXGhostButton
//
// Secondary outline pill. Hairline-stroked capsule with secondary text
// — the "Close" / "Cancel" / "Skip" companion to MTPLXPillButton.
//
// Hover bumps the stroke from `Brand.separator` to
// `Brand.separatorStrong` and the foreground from `Brand.typeSecondary`
// to `Brand.typeBody` so the affordance shows up without breaking the
// surrounding neutral surface. Press is a quiet 0.97 scale.

struct MTPLXGhostButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        MTPLXGhostButtonBody(configuration: configuration)
    }
}

private struct MTPLXGhostButtonBody: View {
    let configuration: ButtonStyle.Configuration

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
            .padding(.horizontal, Brand.Spacing.s4)
            .padding(.vertical, 9)
            .overlay {
                Capsule(style: .continuous)
                    .strokeBorder(
                        hovering ? Brand.separatorStrong : Brand.separator,
                        lineWidth: Brand.hairlineStrong
                    )
            }
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(
                reduceMotion ? nil : .spring(response: 0.18, dampingFraction: 0.74),
                value: configuration.isPressed
            )
            .animation(
                reduceMotion ? nil : .easeOut(duration: 0.15),
                value: hovering
            )
            .onHover { hovering = $0 }
            .contentShape(Capsule(style: .continuous))
    }
}

extension ButtonStyle where Self == MTPLXGhostButton {
    static var mtplxGhost: MTPLXGhostButton { MTPLXGhostButton() }
}
