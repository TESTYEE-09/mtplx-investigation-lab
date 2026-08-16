import SwiftUI
import AppKit

// MARK: - Color hex sugar

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: 1
        )
    }

    init(hex: UInt32, alpha: Double) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: alpha
        )
    }
}

// MARK: - Brand
//
// MTPLX "Jet Chrome" — single source of truth for every token the app
// reads. Identity is jet black + off-white + sparing polished-steel chrome.
// Every dark surface hex is a true neutral (R == G == B) so no surface
// reads with a blue or purple cast on a wide-gamut Apple display. The
// chrome accent is a desaturated cool-steel solid (`accentChrome`) plus a
// 5-stop polished-steel `chromeAccent` LinearGradient for the wordmark,
// hero TPS, and ALL-TIME MAX hero badge. The previous `accentBlue` cool
// blue is deprecated and aliased to `accentChrome` so any missed call site
// still renders as chrome instead of regressing to the old "AI blue."

enum Brand {
    // MARK: - Surfaces (true neutral; no +B bias)

    /// Innermost panel / card surface. Was `0x121216` (B+4); neutralized.
    static let bgInner = Color(hex: 0x121212)

    /// Middle background — the layer most views read against. Was
    /// `0x0A0A0C` (B+2); neutralized.
    static let bgMid = Color(hex: 0x0A0A0A)

    /// Outermost floor color (window background). Was `0x07070A` (B+3);
    /// neutralized to a clean piano black.
    static let bgOuter = Color(hex: 0x050505)

    /// Card / raised surface — the panel a list-row or content card sits
    /// on. Was `0x101015` (B+5); neutralized.
    static let cardSurface = Color(hex: 0x101010)

    /// Slightly elevated surface (chips, pucks, inner controls). Was
    /// `0x14141A` (B+6); neutralized and lifted 2 steps.
    static let raisedSurface = Color(hex: 0x161616)

    /// Top stop of the canonical chrome panel gradient. Used by
    /// `PanelChrome` and any per-surface panel that wants the same
    /// vertical fade.
    static let panelSurfaceTop = Color(hex: 0x1A1A1A)

    /// Bottom stop of the canonical chrome panel gradient. Matches
    /// `cardSurface` so the panel grounds into the surrounding layout.
    static let panelSurfaceBottom = Color(hex: 0x101010)

    // MARK: - Off-white type tokens (slightly cooler to match chrome)

    /// Headline white. Wordmark fallback, hero TPS digits, gauge headline.
    static let typeHi = Color(hex: 0xEFEFEF)

    /// Body white — primary running text.
    static let typeBody = Color(hex: 0xDEDEDE)

    /// Secondary type — labels, captions, supporting metadata.
    static let typeSecondary = Color(hex: 0x9A9A9A)

    /// Tertiary type — the quietest metadata tier.
    static let typeTertiary = Color(hex: 0x6A6A6A)

    /// Backwards-compat aliases for the old `textHighlight`/`accent`
    /// references. Resolved to the same off-white tiers so existing call
    /// sites keep compiling while the rest of the app moves to explicit
    /// `typeHi` / `typeBody` references.
    static let textHighlight = typeBody
    static let accent = typeHi

    // MARK: - Wordmark gradient (kept for the < 14pt fallback)

    /// Three-stop off-white gradient used only by the small-size text
    /// wordmark fallback in `WordmarkView`. The real wordmark ships as a
    /// PNG asset for any height >= 14pt.
    static let typeGradientStops: [Gradient.Stop] = [
        .init(color: Color(hex: 0xFFFFFF), location: 0.0),
        .init(color: Color(hex: 0xF0F0EA), location: 0.55),
        .init(color: Color(hex: 0xCFCFC8), location: 1.0),
    ]

    static let typeGradient = LinearGradient(
        stops: typeGradientStops,
        startPoint: .top,
        endPoint: .bottom
    )

    // MARK: - Chrome accents (the only color tone allowed)

    /// Polished-steel 5-stop sheen used for `MTPLXChromeText`, the gauge
    /// max tick, hero TPS, ALL-TIME MAX badge, and panel highlights.
    /// Bright top / dim middle / bright bottom so the gradient reads as
    /// light catching a curved bevel.
    static let chromeAccentStops: [Gradient.Stop] = [
        .init(color: Color(hex: 0xF6F6F6), location: 0.00),
        .init(color: Color(hex: 0xE0E0E0), location: 0.25),
        .init(color: Color(hex: 0x9A9A9A), location: 0.55),
        .init(color: Color(hex: 0xE0E0E0), location: 0.80),
        .init(color: Color(hex: 0xF6F6F6), location: 1.00),
    ]

    static let chromeAccent = LinearGradient(
        stops: chromeAccentStops,
        startPoint: .top,
        endPoint: .bottom
    )

    /// Desaturated cool-steel solid. Single replacement for every
    /// `accentBlue` reference. Reads as polished metal, never as blue.
    static let accentChrome = Color(hex: 0xC8D0D5)

    /// Warm-steel sister tone. Used only to distinguish prefill from
    /// decode on the gauge and to mark warm semantic states.
    static let accentWarm = Color(hex: 0xD5CFC4)

    /// Deprecated. Repointed to `accentChrome` so any missed call site
    /// still renders as polished chrome instead of regressing to the old
    /// "AI blue." Migrate references to `Brand.accentChrome` (solid) or
    /// `Brand.chromeAccent` (gradient).
    @available(*, deprecated, message: "Use Brand.accentChrome (solid) or Brand.chromeAccent (gradient).")
    static let accentBlue: Color = Color(hex: 0xC8D0D5)

    /// Cool-chrome tint reused as the gauge border when fans are pinned
    /// to max. Slightly cooler than `accentChrome` so it reads as
    /// "state," not "action."
    static let coolChrome = Color(hex: 0xBFD4E0)

    // MARK: - Status colors (already neutral; unchanged)

    static let warning = Color(hex: 0xE9C46A)
    static let danger = Color(hex: 0xE76F51)
    static let success = Color(hex: 0x88D498)

    // MARK: - Hairlines (tokenized, no more 1.4 / 1.5 drift)

    static let hairline: CGFloat = 0.5
    static let hairlineStrong: CGFloat = 0.75
    static let hairlineHeavy: CGFloat = 1.0

    /// Hairline separator at 6% white. Cards, dividers, tab-bar lines.
    static let separator = Color.white.opacity(0.06)

    /// Slightly stronger hairline for active/selected boundaries.
    static let separatorStrong = Color.white.opacity(0.14)

    /// Square diameter of every chrome-strip action button — the
    /// LaunchButton play/stop, the inference-params slider button, and
    /// the refresh button all share this so they read as a uniform row
    /// of circular controls.
    static let controlSize: CGFloat = 32

    // MARK: - Elevation tokens
    //
    // Real shadows — the previous `Brand.Depth` tuple was deliberately
    // zeroed out (`color: .clear`), which made every `.shadow(color:
    // Brand.Depth.ambient.color, ...)` call a no-op. The new tokens are
    // black at meaningful opacities so cards, panels, and overlays
    // actually feel raised. Legacy `Depth.ambient` and `Depth.near`
    // aliases below point at `Elevation.low` so existing call sites pick
    // up real elevation without a rewrite.

    enum Elevation {
        /// Tiles, chips, low-profile controls.
        static let low = (color: Color.black.opacity(0.30), radius: 6.0, x: 0.0, y: 2.0)

        /// Cards, secondary surfaces.
        static let mid = (color: Color.black.opacity(0.40), radius: 12.0, x: 0.0, y: 6.0)

        /// Panels, overlays, sheets.
        static let hi = (color: Color.black.opacity(0.50), radius: 24.0, x: 0.0, y: 12.0)
    }

    // MARK: - Spacing tokens (4pt grid)

    enum Spacing {
        static let s1: CGFloat = 4
        static let s2: CGFloat = 8
        static let s3: CGFloat = 12
        static let s4: CGFloat = 16
        static let s5: CGFloat = 20
        static let s6: CGFloat = 24
        static let s7: CGFloat = 32
    }

    // MARK: - Corner radii (concentric)

    enum Radii {
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 14
        static let panel: CGFloat = 18
    }

    // MARK: - Legacy Depth aliases (now point at real elevation)
    //
    // Preserved so the 6 existing `.shadow(color: Brand.Depth.ambient.color, ...)`
    // call sites — `BottomTabBar`, `MenuBarContent`, `TileRow`,
    // `WelcomeScreen` (dead), `Primitives`, `BottomBar` — pick up a real
    // shadow color for free until they migrate to `Brand.Elevation.*`.

    enum Depth {
        static let near = Elevation.low
        static let ambient = Elevation.low
    }

    // MARK: - Backwards-compat aliases (V1 chrome → off-white)
    //
    // These exist so the wholesale V1 view files keep compiling while we
    // simplify them. Kept untouched until each call site is migrated.

    static let chromeStops = typeGradientStops
    static let chromeFill = typeGradient
    static let warmChromeStops = typeGradientStops
    static let warmChromeFill = typeGradient
    static let shineStops: [Gradient.Stop] = [
        .init(color: Color.white.opacity(0.0), location: 0.0),
        .init(color: Color.white.opacity(0.0), location: 1.0),
    ]
    static let shineGradient = LinearGradient(
        stops: shineStops,
        startPoint: .top,
        endPoint: .bottom
    )

    /// Extrusion is dead. Empty array means callers' ForEach over
    /// `extrusionLayers` renders zero shadow layers, so the wordmark
    /// becomes a single flat off-white text.
    struct ExtrusionLayer {
        let offset: CGFloat
        let fill: Color
    }
    static let extrusionLayers: [ExtrusionLayer] = []

    /// Piano radial → flat. The radial pulled focus from the type and
    /// read "showroom carpet" rather than "tool." A flat bgOuter reads
    /// cleaner and ages better.
    static let pianoRadial = RadialGradient(
        gradient: Gradient(colors: [bgOuter, bgOuter]),
        center: .center,
        startRadius: 0,
        endRadius: 1
    )
}

// MARK: - BrandFont

/// Minimal Apple-ish typography. SF Pro Rounded for the wordmark + hero
/// numbers, SF Pro for body, SF Mono only for actual data. No Inter
/// dependency — system fonts only.
enum BrandFont {
    /// Wordmark / hero number. Lighter weight than V0 (was .black) for
    /// a less aggressive read at large sizes.
    static func wordmark(size: CGFloat) -> Font {
        Font.system(size: size, weight: .heavy, design: .rounded)
    }

    /// Tracking value used by callers that previously did manual
    /// kerning. The new typography uses default tracking, so zero.
    static func wordmarkTracking(size: CGFloat) -> CGFloat { 0 }

    /// Subtitle line under the wordmark. SF Pro Regular at 12pt.
    static func subtitle(size: CGFloat = 12) -> Font {
        Font.system(size: size, weight: .regular, design: .default)
    }
}
