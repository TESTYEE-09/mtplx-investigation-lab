import SwiftUI
import MTPLXAppCore

// MARK: - BenchQuestionGrid
//
// 30-tile question grid. Fixed 10-column layout so AIME's 30 problems
// always lay out as a tidy 10 × 3 matrix, independent of the panel
// width. The previous `.adaptive(64-90)` approach collapsed into a
// single overflowing row in narrower windows because SwiftUI's grid
// adaptive math only wraps when the *child's intrinsic content* asks
// for less width than the parent — when the tile itself wants to
// stay 64pt wide, the row would extend past the panel and clip the
// edges. Fixed `.flexible` columns shrink each tile uniformly so the
// grid honours whatever width the panel actually gives it.

struct BenchQuestionGrid: View {
    let results: [BenchQuestionResult]
    /// Invoked when the user taps a finished tile to review its answer.
    var onSelect: ((BenchQuestionResult) -> Void)? = nil

    @EnvironmentObject private var themeStore: ThemeStore

    /// 10 columns × 3 rows = 30 tiles. Each column is `.flexible` with
    /// a 32pt minimum so tiles stay tappable / readable even when the
    /// panel narrows.
    private let columns: [GridItem] = Array(
        repeating: GridItem(.flexible(minimum: 32), spacing: 6, alignment: .center),
        count: 10
    )

    var body: some View {
        LazyVGrid(columns: columns, alignment: .leading, spacing: 6) {
            ForEach(results) { result in
                Button {
                    onSelect?(result)
                } label: {
                    BenchQuestionTile(result: result)
                }
                .buttonStyle(.plain)
                .disabled(onSelect == nil || !isReviewable(result))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A tile is reviewable once it has a landed result (correct /
    /// wrong / abstain). Pending and currently-solving tiles have
    /// nothing to open — the live card already shows the active one.
    private func isReviewable(_ result: BenchQuestionResult) -> Bool {
        result.status != .pending
    }
}
