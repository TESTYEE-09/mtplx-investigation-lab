import SwiftUI
import MTPLXAppCore

// MARK: - MathProblemRender (math-aware text + LaTeX)
//
// Splits a problem-text string on `$...$` runs and lays them out in a
// FlowLayout. Plain text runs render as SwiftUI Text; math runs render
// as styled LaTeX text. Display math (`$$...$$`) is supported by the same
// split and renders on its own row.
//
// Not a full LaTeX renderer — just enough to read AIME problems
// without raw `$` and `\frac{}` smeared across the page.

struct MathProblemRender: View, Equatable {
    let text: String
    let expanded: Bool

    // Equatable so call sites can wrap this in `.equatable()`: the problem
    // statement is stable for the whole question, so its LaTeX should be
    // typeset once, not on every re-render of the surrounding live card.
    nonisolated static func == (lhs: MathProblemRender, rhs: MathProblemRender) -> Bool {
        lhs.text == rhs.text && lhs.expanded == rhs.expanded
    }

    var body: some View {
        let runs = TextMathRuns.split(text)
        let visible = expanded ? runs : truncated(runs, charBudget: 320)
        FlowLayout(spacing: 4) {
            ForEach(Array(visible.enumerated()), id: \.offset) { _, run in
                switch run {
                case .text(let s):
                    Text(s)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(Brand.typeBody)
                        .fixedSize(horizontal: false, vertical: true)
                case .inlineMath(let latex):
                    Text(mathLabel(latex, display: false))
                        .font(.system(size: 13, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeBody)
                case .displayMath(let latex):
                    Text(mathLabel(latex, display: true))
                        .font(.system(size: 15, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeBody)
                        .padding(.vertical, 4)
                }
            }
        }
        // `.infinity` here is critical: FlowLayout's `sizeThatFits` is
        // greedy when the proposal width is nil. Forcing maxWidth =
        // .infinity in the parent chain pins it to the bounded panel
        // width and lets the layout wrap text/math runs onto new lines
        // instead of running off-screen.
        .frame(maxWidth: .infinity, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func truncated(_ runs: [TextMathRuns.Run], charBudget: Int) -> [TextMathRuns.Run] {
        var seen = 0
        var out: [TextMathRuns.Run] = []
        for run in runs {
            let count = run.approximateCharacterCount
            if seen + count > charBudget {
                // Trim the run's text if it's a text run; otherwise stop.
                if case .text(var s) = run {
                    let take = max(0, charBudget - seen)
                    if take > 0 {
                        s = String(s.prefix(take)) + "…"
                        out.append(.text(s))
                    }
                }
                break
            }
            out.append(run)
            seen += count
        }
        return out
    }

    private func mathLabel(_ latex: String, display: Bool) -> String {
        let source = display ? latex.trimmingCharacters(in: .whitespacesAndNewlines) : latex
        return StreamingMathTextFormatter.readableText(from: source)
    }
}
