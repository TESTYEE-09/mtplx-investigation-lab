import Foundation

/// Incremental boxed-answer detector for live AIME streams.
///
/// The full answer transcript can run for tens of thousands of tokens. This
/// keeps extraction bounded by scanning only a retained tail while preserving
/// enough overlap for `\boxed{277}` fragments split across SSE deltas.
public struct LiveAnswerExtractor: Sendable {
    public static let defaultTailLimit = 8_192

    private let tailLimit: Int
    private var tail: String = ""

    public private(set) var extractedAnswer: Int?
    public private(set) var hasBoxedMarker: Bool = false

    public init(tailLimit: Int = Self.defaultTailLimit) {
        self.tailLimit = max(256, tailLimit)
    }

    public mutating func reset() {
        tail = ""
        extractedAnswer = nil
        hasBoxedMarker = false
    }

    @discardableResult
    public mutating func append(_ delta: String) -> Int? {
        guard !delta.isEmpty else { return extractedAnswer }
        tail.append(delta)
        if tail.count > tailLimit {
            tail = String(tail.suffix(tailLimit))
        }
        if tail.contains("\\boxed") {
            hasBoxedMarker = true
        }
        if let parsed = BenchmarkGrader.extractBoxed(tail) {
            extractedAnswer = parsed
        }
        return extractedAnswer
    }
}
