import SwiftUI

// MARK: - FlowRow
//
// Simple flow layout: lays out children left-to-right and wraps to a
// new row when the next child would overflow. Used by the assistant
// bubble's collapsed tool-trace strip (so multiple `web_search` /
// `fetch_url` capsules wrap nicely under the markdown) and by the
// composer attachment chips when more than two files are pending.
//
// SwiftUI ships `Layout` in macOS 13+; we are macOS 14+ so this is
// the canonical approach. Spacing defaults match Aphanes V2's
// `FlowRow` (8pt horizontal, 8pt vertical).

struct FlowRow: Layout {
    var horizontalSpacing: CGFloat = 8
    var verticalSpacing: CGFloat = 8
    var alignment: HorizontalAlignment = .leading

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let arrangement = arrange(subviews: subviews, in: maxWidth)
        return CGSize(width: arrangement.width, height: arrangement.height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let arrangement = arrange(subviews: subviews, in: bounds.width)
        for placement in arrangement.placements {
            let origin = CGPoint(
                x: bounds.minX + placement.x,
                y: bounds.minY + placement.y
            )
            placement.subview.place(
                at: origin,
                proposal: ProposedViewSize(placement.size)
            )
        }
    }

    private struct Arrangement {
        var width: CGFloat
        var height: CGFloat
        var placements: [Placement]
    }

    private struct Placement {
        var subview: LayoutSubview
        var size: CGSize
        var x: CGFloat
        var y: CGFloat
    }

    private func arrange(subviews: Subviews, in maxWidth: CGFloat) -> Arrangement {
        var placements: [Placement] = []
        var rowY: CGFloat = 0
        var rowMaxHeight: CGFloat = 0
        var x: CGFloat = 0
        var widestRow: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                rowY += rowMaxHeight + verticalSpacing
                rowMaxHeight = 0
                x = 0
            }
            placements.append(
                Placement(subview: subview, size: size, x: x, y: rowY)
            )
            x += size.width + horizontalSpacing
            rowMaxHeight = max(rowMaxHeight, size.height)
            widestRow = max(widestRow, x - horizontalSpacing)
        }
        return Arrangement(
            width: widestRow,
            height: rowY + rowMaxHeight,
            placements: placements
        )
    }
}
