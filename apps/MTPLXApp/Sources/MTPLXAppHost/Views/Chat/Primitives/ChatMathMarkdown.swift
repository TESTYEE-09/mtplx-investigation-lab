import SwiftUI
import MarkdownUI
import MTPLXAppCore

private final class ChatMarkdownContentBox: NSObject {
    let document: MarkdownContent

    init(document: MarkdownContent) {
        self.document = document
    }
}

private enum CachedChatMarkdownDocuments {
    nonisolated(unsafe) private static let cache: NSCache<NSString, ChatMarkdownContentBox> = {
        let cache = NSCache<NSString, ChatMarkdownContentBox>()
        cache.countLimit = 256
        cache.totalCostLimit = 12_000_000
        return cache
    }()

    static func document(for source: String) -> MarkdownContent {
        let key = source as NSString
        if let cached = cache.object(forKey: key) {
            return cached.document
        }

        let document = MarkdownContent(source)
        cache.setObject(
            ChatMarkdownContentBox(document: document),
            forKey: key,
            cost: source.utf8.count
        )
        return document
    }

    static func removeAllObjects() {
        cache.removeAllObjects()
    }
}

extension ChatRenderCaches {
    static func clearMarkdownDocuments() {
        CachedChatMarkdownDocuments.removeAllObjects()
    }
}

// MARK: - ChatMathMarkdown
//
// Markdown + LaTeX renderer for chat assistant text. Chat previously
// rendered everything through MarkdownUI's `Markdown`, which has no math
// support, so `$$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$$` showed up as raw
// dollar-sign source. This reuses the exact LaTeX splitter the AIME views
// use (`StreamingDocumentStore.mathRuns`) so prose keeps full Markdown and
// math runs stay readable in a release bundle without shipping a separate
// SwiftPM math-font resource bundle.
//
// Perf: streaming text deliberately bypasses MarkdownUI and math splitting.
// Rich markdown/math rendering is a settled-message path only, so token
// generation stays close to Open WebUI's cheap append-and-paint loop.

struct ChatMathMarkdown: View {
    let text: String
    var isStreaming: Bool = false

    var body: some View {
        if isStreaming {
            plainStreamingText(text)
        } else {
            renderedMarkdown
        }
    }

    @ViewBuilder
    private var renderedMarkdown: some View {
        if text.contains("```") {
            // Code fences own their contents. Running the math splitter across
            // a fenced program can mistake `$` in source code for LaTeX and
            // break the final markdown pass.
            markdown(text)
        } else {
            let runs = StreamingDocumentStore.mathRuns(in: text)
            let hasMath = runs.contains { $0.kind != .text }
            if !hasMath {
                // Fast path: identical to the previous settled-message behavior.
                markdown(text)
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(rows(from: runs).enumerated()), id: \.offset) { _, row in
                        rowView(row)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    /// A render row is either a display-math equation (its own centered
    /// line) or a sequence of text + inline-math runs that flow together.
    private enum Row {
        case display(String)
        case inlineFlow([StreamingMathRun])
    }

    private func rows(from runs: [StreamingMathRun]) -> [Row] {
        var rows: [Row] = []
        var current: [StreamingMathRun] = []
        func flush() {
            if !current.isEmpty {
                rows.append(.inlineFlow(current))
                current = []
            }
        }
        for run in runs {
            if run.kind == .displayMath {
                flush()
                rows.append(.display(run.text))
            } else {
                current.append(run)
            }
        }
        flush()
        return rows
    }

    @ViewBuilder
    private func rowView(_ row: Row) -> some View {
        switch row {
        case .display(let latex):
            Text(mathLabel(latex, display: true))
                .font(.system(size: 15, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeHi)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 4)
        case .inlineFlow(let runs):
            // A row that is pure text keeps full block Markdown (lists,
            // headings, bold, links, inline code). Only rows that actually
            // interleave inline math drop to the inline flow layout.
            if runs.count == 1, runs[0].kind == .text {
                markdown(runs[0].text)
            } else {
                FlowLayout(spacing: 3) {
                    ForEach(runs) { run in
                        inlineRunView(run)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private func inlineRunView(_ run: StreamingMathRun) -> some View {
        switch run.kind {
        case .text:
            Text(Self.inlineAttributed(run.text))
                .font(.system(size: 14))
                .foregroundStyle(Brand.typeHi)
                .fixedSize(horizontal: false, vertical: true)
        case .inlineMath:
            Text(mathLabel(run.text, display: false))
                .font(.system(size: 14, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeHi)
        case .displayMath:
            // Defensive: a display run normally becomes its own `.display`
            // row, but render it correctly if one reaches here.
            Text(mathLabel(run.text, display: true))
                .font(.system(size: 15, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeHi)
        }
    }

    @ViewBuilder
    private func markdown(_ text: String) -> some View {
        Markdown(CachedChatMarkdownDocuments.document(for: text))
            .markdownTheme(.mtplxChat)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func plainStreamingText(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 14))
            .foregroundStyle(Brand.typeHi)
            .textSelection(.disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// Inline-only Markdown for text runs that share a row with inline
    /// math, so bold/italic/code/links survive even on a math line.
    private static func inlineAttributed(_ text: String) -> AttributedString {
        (try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .inlineOnlyPreservingWhitespace
            )
        )) ?? AttributedString(text)
    }

    private func mathLabel(_ latex: String, display: Bool) -> String {
        let source = display ? latex.trimmingCharacters(in: .whitespacesAndNewlines) : latex
        return StreamingMathTextFormatter.readableText(from: source)
    }
}
