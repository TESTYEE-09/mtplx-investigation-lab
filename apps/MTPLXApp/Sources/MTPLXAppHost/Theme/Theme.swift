import SwiftUI

// MARK: - ThemeStore
//
// MTPLX V1 is locked to a single brand identity (piano black + chrome
// silver). The old multi-theme picker (system/hippo/river/mono) is gone —
// the app is one curated surface, not a customizable one. What survives in
// the store are the two preferences that genuinely belong to the user:
// whether they want subtle sound cues and whether they want motion reduced
// for accessibility.

@MainActor
public final class ThemeStore: ObservableObject {
    @AppStorage("mtplx.app.soundEnabled") public var soundEnabled: Bool = false
    @AppStorage("mtplx.app.reduceMotion") public var reduceMotionPreference: Bool = false

    public init() {}
}

// MARK: - Modifier

/// Pins the window to the MTPLX V1 brand identity:
/// - `.preferredColorScheme(.dark)` so the radial / chrome reads correctly
///   regardless of the user's macOS Appearance setting.
/// - `.tint(Brand.accent)` so all standard controls pick up the chrome tone.
/// - `.background(Brand.bgOuter)` as the floor color (the piano radial
///   sits on top of this in `ContentView`).
public struct AppliesBrand: ViewModifier {
    public func body(content: Content) -> some View {
        content
            .preferredColorScheme(.dark)
            .tint(Brand.accent)
            .background(Brand.bgOuter)
    }
}

public extension View {
    /// Apply at the root of the view hierarchy to lock the MTPLX V1 brand.
    func appliesBrand() -> some View {
        modifier(AppliesBrand())
    }
}

// MARK: - Semantic color helpers
//
// Kept as `mtplx*` to preserve call-site stability across the codebase.
// They now resolve to `Brand` tokens so any future tweak ripples
// everywhere.

extension Color {
    /// Subtle separator on card boundaries and tab-bar hairlines.
    static var mtplxSeparator: Color { Brand.separator }

    /// Warning amber used by ThermalRuleBanner + NewMaxToast.
    static var mtplxWarning: Color { Brand.warning }

    /// Danger red used by ConnectionIssueBanner + degraded states.
    static var mtplxDanger: Color { Brand.danger }

    /// Calm success tint used by fan-verified / cache-hit indicators.
    static var mtplxSuccess: Color { Brand.success }

    /// Default accent — polished chrome. Use this anywhere the old code
    /// reached for `Color.accentColor`. Resolves to the desaturated
    /// cool-steel solid so toolbar tints, focus rings, and system
    /// controls all pick up the Jet Chrome identity.
    static var mtplxAccent: Color { Brand.accentChrome }
}

// MARK: - Motion

/// Wraps `withAnimation` so Performance Lock and the user's Reduce Motion
/// preference can short-circuit animation without touching every call site.
@inlinable public func animateValue<V>(
    _ animation: Animation,
    motionEnabled: Bool,
    _ body: () -> V
) -> V {
    if motionEnabled {
        return withAnimation(animation, body)
    }
    return body()
}
