import SwiftUI

// MARK: - EqualColumnsLayout
//
// Divide the parent's horizontal space into N exactly-equal columns,
// regardless of each subview's intrinsic width. Built because SwiftUI's
// `Grid` sizes columns to fit content (so a "Depth" tile with longer
// text always read wider than a "Memory" tile despite both having
// `frame(maxWidth: .infinity)`) and `HStack` distributes leftover
// space *after* honouring each child's intrinsic width (same trap).
//
// Use:
//
//   EqualColumnsLayout(spacing: Brand.Spacing.s3) {
//       LiveTile(label: "Lifetime", ...)
//       LiveTile(label: "Cached", ...)
//       LiveTile(label: "Memory", ...)
//       LiveTile(label: "5-min Min/Max", ...)
//       LiveTile(label: "Depth", ...)
//   }
//
// Every column receives `(containerWidth - (n-1) * spacing) / n` as
// its proposed width. Height is the tallest subview, so the row
// matches the natural minimum height of the busiest tile.

struct EqualColumnsLayout: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let totalWidth = proposal.width ?? subviews.reduce(0) { acc, view in
            acc + view.sizeThatFits(.unspecified).width
        }
        let column = columnWidth(in: totalWidth, count: subviews.count)
        let columnProposal = ProposedViewSize(width: column, height: proposal.height)
        let height = subviews
            .map { $0.sizeThatFits(columnProposal).height }
            .max() ?? 0
        return CGSize(width: totalWidth, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        guard !subviews.isEmpty else { return }
        let column = columnWidth(in: bounds.width, count: subviews.count)
        let columnProposal = ProposedViewSize(width: column, height: bounds.height)
        for (index, view) in subviews.enumerated() {
            let x = bounds.minX + CGFloat(index) * (column + spacing)
            view.place(
                at: CGPoint(x: x, y: bounds.minY),
                anchor: .topLeading,
                proposal: columnProposal
            )
        }
    }

    private func columnWidth(in totalWidth: CGFloat, count: Int) -> CGFloat {
        guard count > 0 else { return 0 }
        let gaps = CGFloat(max(0, count - 1)) * spacing
        return max(0, (totalWidth - gaps) / CGFloat(count))
    }
}
