import SwiftUI

// MARK: - FlowLayout
//
// Minimal wrap-around `Layout` used by `StackedBar`'s legend and
// `MathProblemRender`. Items lay out left-to-right, wrapping when
// the row width is exceeded. Available natively starting in
// macOS 13.

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    /// Memoizes the arrangement for a given proposed width. Without this the
    /// layout re-ran `subview.sizeThatFits` for every child on every layout
    /// pass — and for compact inline render children that measurement re-runs
    /// the (expensive) typesetting. SwiftUI calls `updateCache` whenever the
    /// subviews change (content edited / added / removed), which is exactly
    /// when the memoized result must be discarded, so a stable line's math
    /// is measured once and then reused on every subsequent pass.
    struct Cache {
        var width: CGFloat = .nan
        var arrangement: (size: CGSize, points: [CGPoint], sizes: [CGSize])?
    }

    func makeCache(subviews: Subviews) -> Cache { Cache() }

    func updateCache(_ cache: inout Cache, subviews: Subviews) { cache = Cache() }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) -> CGSize {
        arrangement(maxWidth: proposal.width ?? .infinity, subviews: subviews, cache: &cache).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) {
        let maxWidth = proposal.width ?? bounds.width
        let arr = arrangement(maxWidth: maxWidth, subviews: subviews, cache: &cache)
        for (index, point) in arr.points.enumerated() {
            let origin = CGPoint(x: bounds.minX + point.x, y: bounds.minY + point.y)
            subviews[index].place(at: origin, proposal: ProposedViewSize(arr.sizes[index]))
        }
    }

    private func arrangement(maxWidth: CGFloat, subviews: Subviews, cache: inout Cache) -> (size: CGSize, points: [CGPoint], sizes: [CGSize]) {
        if cache.width == maxWidth, let arrangement = cache.arrangement {
            return arrangement
        }
        let arrangement = arrange(subviews: subviews, in: maxWidth)
        cache.width = maxWidth
        cache.arrangement = arrangement
        return arrangement
    }

    private func arrange(subviews: Subviews, in maxWidth: CGFloat) -> (size: CGSize, points: [CGPoint], sizes: [CGSize]) {
        var points: [CGPoint] = []
        var sizes: [CGSize] = []
        var rowX: CGFloat = 0
        var rowY: CGFloat = 0
        var rowHeight: CGFloat = 0
        var maxRowWidth: CGFloat = 0
        // Bound height so long Text subviews wrap to multiple lines
        // instead of reporting their single-line natural width. The
        // earlier `.unspecified` proposal made `Text("a long sentence")`
        // return the full width of that sentence (often > maxWidth),
        // which FlowLayout would happily place starting from x = 0 and
        // let overflow off the right edge of the panel.
        let proposal = ProposedViewSize(width: maxWidth, height: nil)
        for view in subviews {
            let proposed = view.sizeThatFits(proposal)
            // Defensive cap: if a subview ignores the proposal (Math runs
            // sometimes do), clamp it to maxWidth so the row math below
            // still works.
            let size = CGSize(width: min(proposed.width, maxWidth), height: proposed.height)
            sizes.append(size)
            if rowX + size.width > maxWidth, rowX > 0 {
                rowY += rowHeight + spacing
                rowX = 0
                rowHeight = 0
            }
            points.append(CGPoint(x: rowX, y: rowY))
            rowX += size.width + spacing
            rowHeight = max(rowHeight, size.height)
            maxRowWidth = max(maxRowWidth, rowX - spacing)
        }
        return (CGSize(width: maxRowWidth, height: rowY + rowHeight), points, sizes)
    }
}
