import SwiftUI
import AppKit

// MARK: - HermesMark
//
// The official Hermes Agent (Nous Research) logo, used in the launch
// picker in place of a generic SF Symbol. Bundled as a white-on-
// transparent PNG (`Resources/Brands/HermesLogo.png`) so it reads on the
// jet-black surfaces, and loaded through a multi-tier bundle lookup (same
// strategy as `WordmarkView`) so it resolves whether the app runs from
// the SwiftPM build dir or the assembled `.app`. Falls back to a clean
// messenger glyph if the asset can't be found.
//
// The PNG ships pure white, so it's used here as an alpha MASK and filled
// with `tint` — this makes the mark sit at the exact same gray as every
// other launch-row icon (and flip to the accent when it's the last-used
// target) instead of glowing pure white next to them.

struct HermesMark: View {
    var size: CGFloat = 16
    var tint: Color = Brand.typeSecondary

    var body: some View {
        if let image = Self.cachedImage {
            tint
                .frame(width: size, height: size)
                .mask {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .scaledToFit()
                        .frame(width: size, height: size)
                }
                .accessibilityHidden(true)
        } else {
            Image(systemName: "paperplane")
                .font(.system(size: size * 0.85, weight: .regular))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(tint)
        }
    }

    /// Loaded once. Tries the main bundle and every loaded bundle so a
    /// missing-resource edge case degrades to the fallback glyph instead of
    /// crashing.
    private static let cachedImage: NSImage? = {
        let name = "HermesLogo"
        let ext = "png"
        if let url = Bundle.main.url(forResource: name, withExtension: ext),
           let img = NSImage(contentsOf: url) {
            return img
        }
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: name, withExtension: ext),
               let img = NSImage(contentsOf: url) {
                return img
            }
        }
        return nil
    }()
}
