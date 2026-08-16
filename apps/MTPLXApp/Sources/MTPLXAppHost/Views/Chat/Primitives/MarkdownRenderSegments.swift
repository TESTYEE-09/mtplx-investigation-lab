import Foundation

// MARK: - MarkdownRenderSegments
//
// Streaming-safe markdown split. Verbatim port from Aphanes V2's
// AppViews.swift (lines ~8592–8703). The job: given a possibly-growing
// markdown string, return the prefix that is safe to parse as markdown
// (`markdownContent`) and the unstable trailing bit that should be
// rendered as plain text until the stream catches up to the next
// paragraph boundary.
//
// Without this split, every newly-streamed character triggers a full
// markdown reparse, which flickers and is expensive. With it, the
// stable prefix renders through cached `Markdown(...)` views and only
// the live tail repaints per token.
//
// `StreamingOpenCodeBlock` covers the special case where an unclosed
// triple-backtick fence is in flight — render the prefix as markdown
// and the fence body as a live code block until the closing fence
// arrives.

public struct MarkdownRenderSegments: Equatable, Sendable {
    public let markdownContent: String
    public let trailingPlainText: String?
    public let streamingOpenCodeBlock: StreamingOpenCodeBlock?

    public init(content: String, isStreaming: Bool) {
        guard isStreaming else {
            self.markdownContent = content
            self.trailingPlainText = nil
            self.streamingOpenCodeBlock = nil
            return
        }

        if let streamingOpenCodeBlock = Self.streamingOpenCodeBlock(in: content) {
            self.markdownContent = streamingOpenCodeBlock.stableMarkdownPrefix
            self.trailingPlainText = nil
            self.streamingOpenCodeBlock = streamingOpenCodeBlock
            return
        }

        let splitIndex = Self.stableBoundary(in: content)
        self.markdownContent = String(content[..<splitIndex])
        let trailing = String(content[splitIndex...])
        self.trailingPlainText = trailing.isEmpty ? nil : trailing
        self.streamingOpenCodeBlock = nil
    }

    public struct StreamingOpenCodeBlock: Equatable, Sendable {
        public let stableMarkdownPrefix: String
        public let language: String?
        public let code: String

        public init(stableMarkdownPrefix: String, language: String?, code: String) {
            self.stableMarkdownPrefix = stableMarkdownPrefix
            self.language = language
            self.code = code
        }
    }

    // MARK: - Internals

    private static func hasUnclosedCodeFence(in content: String) -> Bool {
        guard !content.isEmpty else { return false }
        var index = content.startIndex
        var fenceCount = 0
        while index < content.endIndex {
            if content[index...].hasPrefix("```") {
                fenceCount += 1
                index = content.index(index, offsetBy: 3)
                continue
            }
            index = content.index(after: index)
        }
        return fenceCount % 2 != 0
    }

    private static func streamingOpenCodeBlock(in content: String) -> StreamingOpenCodeBlock? {
        guard Self.hasUnclosedCodeFence(in: content),
            let fenceRange = content.range(of: "```", options: .backwards)
        else { return nil }

        let prefix = String(content[..<fenceRange.lowerBound])
        let fenceBody = content[fenceRange.upperBound...]
        let language: String?
        let code: String

        if let newline = fenceBody.firstIndex(of: "\n") {
            let rawLanguage = fenceBody[..<newline]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            language = rawLanguage.isEmpty ? nil : rawLanguage
            code = String(fenceBody[fenceBody.index(after: newline)...])
        } else {
            let rawLanguage = fenceBody.trimmingCharacters(in: .whitespacesAndNewlines)
            language = rawLanguage.isEmpty ? nil : rawLanguage
            code = ""
        }

        return StreamingOpenCodeBlock(
            stableMarkdownPrefix: prefix,
            language: language,
            code: code
        )
    }

    private static func stableBoundary(in content: String) -> String.Index {
        guard !content.isEmpty else { return content.startIndex }
        var lastStableBoundary = content.startIndex
        var index = content.startIndex
        var openFenceCount = 0

        while index < content.endIndex {
            if content[index...].hasPrefix("```") {
                openFenceCount += 1
                index = content.index(index, offsetBy: 3)
                if openFenceCount % 2 == 0 {
                    lastStableBoundary = index
                }
                continue
            }
            if openFenceCount % 2 == 0, content[index...].hasPrefix("\n\n") {
                lastStableBoundary = content.index(index, offsetBy: 2)
                index = lastStableBoundary
                continue
            }
            index = content.index(after: index)
        }

        if openFenceCount % 2 == 0, content.hasSuffix("\n") {
            return content.endIndex
        }
        return lastStableBoundary
    }
}
