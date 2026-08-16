import Foundation

// MARK: - TextMathRuns
//
// Lightweight LaTeX-aware tokenizer. Splits a problem string into
// runs of plain text, inline math (`$...$`), and display math
// (`$$...$$`). `MathProblemRender` then turns each run into either
// a SwiftUI Text run with math styling.

enum TextMathRuns {
    enum Run {
        case text(String)
        case inlineMath(String)
        case displayMath(String)

        var approximateCharacterCount: Int {
            switch self {
            case .text(let s): return s.count
            case .inlineMath(let s): return s.count + 2
            case .displayMath(let s): return s.count + 4
            }
        }
    }

    static func split(_ source: String) -> [Run] {
        // Order matters: detect $$...$$ first so the simpler $...$
        // pass doesn't consume the inner delimiters.
        var output: [Run] = []
        var index = source.startIndex
        let end = source.endIndex

        func emitText(_ slice: Substring) {
            guard !slice.isEmpty else { return }
            output.append(.text(String(slice)))
        }

        while index < end {
            // Try display math first.
            if let display = source.range(of: "$$", range: index..<end),
               let closing = source.range(of: "$$", range: display.upperBound..<end) {
                emitText(source[index..<display.lowerBound])
                let body = source[display.upperBound..<closing.lowerBound]
                output.append(.displayMath(String(body).trimmingCharacters(in: .whitespacesAndNewlines)))
                index = closing.upperBound
                continue
            }
            // Inline math: single $...$. Skip escaped $.
            if let opening = source.range(of: "$", range: index..<end) {
                emitText(source[index..<opening.lowerBound])
                let searchStart = opening.upperBound
                if let closing = source.range(of: "$", range: searchStart..<end) {
                    let body = source[searchStart..<closing.lowerBound]
                    output.append(.inlineMath(String(body)))
                    index = closing.upperBound
                    continue
                } else {
                    // Unmatched $ — render the rest as plain text.
                    emitText(source[opening.lowerBound..<end])
                    index = end
                }
            } else {
                emitText(source[index..<end])
                index = end
            }
        }

        return output
    }
}
