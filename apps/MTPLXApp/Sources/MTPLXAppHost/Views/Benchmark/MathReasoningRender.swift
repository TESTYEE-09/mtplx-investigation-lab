import SwiftUI
import MTPLXAppCore

enum MathReasoningRenderStyle: Equatable {
    case latex
    case plain
}

// MARK: - MathReasoningRender
//
// Math-aware renderer for the AIME reasoning trace. The live path receives
// stable `StreamingDocumentBlock` rows from `BenchmarkOrchestrator`, so
// body work is row rendering, not full-transcript splitting and parsing.
//
// Static detail views can still use `init(text:)`; that compatibility path
// parses once when the detail row is built, outside the live stream hot path.

struct MathReasoningRender: View {
    let blocks: [StreamingDocumentBlock]
    let style: MathReasoningRenderStyle

    init(blocks: [StreamingDocumentBlock], style: MathReasoningRenderStyle = .latex) {
        self.blocks = blocks
        self.style = style
    }

    init(text: String) {
        self.blocks = Self.blocks(from: text)
        self.style = .latex
    }

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 5) {
            ForEach(blocks) { block in
                ReasoningBlock(block: block, style: style)
                    .equatable()
                    .id(block.id)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private static func blocks(from text: String) -> [StreamingDocumentBlock] {
        text.components(separatedBy: .newlines)
            .enumerated()
            .map { index, line in
                let runs = StreamingDocumentStore.mathRuns(in: line)
                let kind: StreamingDocumentBlockKind =
                    runs.count == 1 && runs[0].kind == .text ? .plain : .mathRuns(runs)
                return StreamingDocumentBlock(
                    id: index,
                    text: line,
                    kind: kind,
                    finalized: true
                )
            }
    }
}

// MARK: - ReasoningLine
//
// One reasoning block. `Equatable` lets SwiftUI skip unchanged finalized rows,
// and math-free rows take the plain Text path with no FlowLayout or Math view.

private struct ReasoningBlock: View, Equatable {
    let block: StreamingDocumentBlock
    let style: MathReasoningRenderStyle

    nonisolated static func == (lhs: ReasoningBlock, rhs: ReasoningBlock) -> Bool {
        lhs.block == rhs.block && lhs.style == rhs.style
    }

    var body: some View {
        if block.text.isEmpty {
            Color.clear.frame(height: 6)
        } else if style == .plain {
            plainText(block.text)
        } else {
            switch block.kind {
            case .plain, .unfinished, .markdown, .codeFence:
                plainText(block.text)
            case .mathRuns(let runs):
                mathRuns(runs)
            }
        }
    }

    private func plainText(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(Brand.typeBody.opacity(0.9))
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func mathRuns(_ runs: [StreamingMathRun]) -> some View {
        FlowLayout(spacing: 3) {
            ForEach(runs) { run in
                runView(run)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private func runView(_ run: StreamingMathRun) -> some View {
        switch run.kind {
        case .text:
            Text(run.text)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Brand.typeBody.opacity(0.9))
                .fixedSize(horizontal: false, vertical: true)
        case .inlineMath:
            Text(mathLabel(run.text, display: false))
                .font(.system(size: 12, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeBody)
        case .displayMath:
            Text(mathLabel(run.text, display: true))
                .font(.system(size: 13, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeBody)
                .padding(.vertical, 3)
        }
    }

    private func mathLabel(_ latex: String, display: Bool) -> String {
        let source = display ? latex.trimmingCharacters(in: .whitespacesAndNewlines) : latex
        return StreamingMathTextFormatter.readableText(from: source)
    }
}
