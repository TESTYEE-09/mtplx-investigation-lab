import Foundation

// MARK: - BenchmarkModels
//
// Value types shared between the BenchmarkOrchestrator (state machine),
// BenchmarkStreamClient (SSE consumer), and the SwiftUI BenchmarkOverlay
// views. Mirrors the Python `mtplx/benchmarks/runners/aime.py` event
// schema so the two sides stay in lock step.
//
// Wire shapes are decoded out of the SSE `data:` JSON exactly as the
// Python runner emits them. The Swift side does NOT compose chat
// completion bodies itself - the backend runner owns that surface.

public struct BenchmarkStartOptions: Equatable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var enableThinking: Bool?
    public var questionProcessIsolation: String?
    public var questionLimit: Int?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        enableThinking: Bool? = nil,
        questionProcessIsolation: String? = "per_question",
        questionLimit: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.enableThinking = enableThinking
        self.questionProcessIsolation = questionProcessIsolation
        self.questionLimit = questionLimit
    }

    public init(settings: MutableSettings?) {
        self.temperature = settings?.temperature
        self.topP = settings?.topP
        self.topK = settings?.topK
        self.questionProcessIsolation = "per_question"
        self.questionLimit = nil
        switch settings?.reasoning?.lowercased() {
        case "on":
            self.enableThinking = true
        case "off":
            self.enableThinking = false
        default:
            self.enableThinking = settings?.enableThinking
        }
    }
}

// MARK: - Run state

public enum BenchRunState: String, Codable, Sendable, CaseIterable {
    case idle
    case running
    case paused
    case done
    case cancelled
    case error

    public var isTerminal: Bool {
        switch self {
        case .done, .cancelled, .error: return true
        default: return false
        }
    }

    public var isLive: Bool {
        self == .running || self == .paused
    }
}

// MARK: - Per-question types

public enum QuestionStatus: String, Codable, Sendable {
    case pending          // not yet attempted
    case correct          // extracted == expected
    case wrong            // extracted != expected
    case abstain          // no parseable answer
}

public struct BenchProblem: Identifiable, Codable, Hashable, Sendable {
    public let id: String         // e.g. "2026-I-7"
    public let set: String        // "AIME I" or "AIME II"
    public let year: Int
    public let index: Int         // 1..15 within the set
    public let problem: String    // LaTeX preserved verbatim
    public let answer: Int
    public let source: String

    public init(
        id: String,
        set: String,
        year: Int,
        index: Int,
        problem: String,
        answer: Int,
        source: String
    ) {
        self.id = id
        self.set = set
        self.year = year
        self.index = index
        self.problem = problem
        self.answer = answer
        self.source = source
    }

    /// "Q7" - shown on the question grid tile.
    public var shortTag: String { "Q\(index)" }

    /// Detects an Asymptote (`[asy]...[/asy]`) figure block that
    /// The app cannot render. The BenchLiveCard strips these from
    /// the displayed text and replaces them with a "Figure: see AoPS"
    /// link so the model still gets the raw markup but the user sees
    /// premium rendering.
    public var hasAsymptoteFigure: Bool {
        problem.contains("[asy]") && problem.contains("[/asy]")
    }
}

public struct BenchQuestionResult: Identifiable, Codable, Hashable, Sendable {
    public let idx: Int                       // 1..30 across the run
    public var problem: BenchProblem          // mutated as question_started arrives with the full text
    public var status: QuestionStatus
    public var extracted: Int?
    public var startedAt: Date?
    public var endedAt: Date?
    public var reasoningTokenCount: Int
    public var answerTokenCount: Int
    /// Full reasoning + visible answer text captured live when the
    /// question finished, so the user can reopen a solved tile and
    /// review how the model got there. Empty when a run is rehydrated
    /// from a server snapshot (the backend keeps token counts, not the
    /// transcript text).
    public var reasoning: String
    public var answer: String

    public var id: Int { idx }

    public init(
        idx: Int,
        problem: BenchProblem,
        status: QuestionStatus = .pending,
        extracted: Int? = nil,
        startedAt: Date? = nil,
        endedAt: Date? = nil,
        reasoningTokenCount: Int = 0,
        answerTokenCount: Int = 0,
        reasoning: String = "",
        answer: String = ""
    ) {
        self.idx = idx
        self.problem = problem
        self.status = status
        self.extracted = extracted
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.reasoningTokenCount = reasoningTokenCount
        self.answerTokenCount = answerTokenCount
        self.reasoning = reasoning
        self.answer = answer
    }

    public var durationMs: Int? {
        guard let s = startedAt, let e = endedAt else { return nil }
        return Int(e.timeIntervalSince(s) * 1000)
    }
}

public struct BenchAnswerVerificationState: Codable, Hashable, Sendable {
    public let idx: Int
    public let attempt: Int
    public let mode: String
    public let proposedAnswer: Int?
    public let verifiedAnswer: Int?
    public let verifierAnswers: [Int?]
    public let resolution: String?
    public let durationMs: Int?
    public let isRunning: Bool

    public init(
        idx: Int,
        attempt: Int,
        mode: String,
        proposedAnswer: Int?,
        verifiedAnswer: Int? = nil,
        verifierAnswers: [Int?] = [],
        resolution: String? = nil,
        durationMs: Int? = nil,
        isRunning: Bool
    ) {
        self.idx = idx
        self.attempt = attempt
        self.mode = mode
        self.proposedAnswer = proposedAnswer
        self.verifiedAnswer = verifiedAnswer
        self.verifierAnswers = verifierAnswers
        self.resolution = resolution
        self.durationMs = durationMs
        self.isRunning = isRunning
    }

    public var correctedAnswer: Bool {
        guard let proposedAnswer, let verifiedAnswer else { return false }
        return proposedAnswer != verifiedAnswer
    }

    public var hasVerifierAnswer: Bool {
        verifierAnswers.contains { $0 != nil }
    }
}

// MARK: - History row (from GET /history)

public struct BenchRunSummary: Identifiable, Codable, Hashable, Sendable {
    public let runID: String
    public let state: String
    public let score: Int
    public let total: Int
    public let accuracy: Double?
    public let durationMs: Int?
    public let model: String
    public let endedAt: Date?

    public var id: String { runID }

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case state
        case score
        case total
        case accuracy
        case durationMs = "duration_ms"
        case model
        case endedAt = "ended_at"
    }
}

public struct BenchHistoryResponse: Codable, Sendable {
    public let runs: [BenchRunSummary]
}

// MARK: - /start response

public struct BenchStartResponse: Codable, Sendable {
    public let runID: String
    public let total: Int
    public let model: String
    public let year: Int
    public let state: String
    public let startedAt: Date?

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case total
        case model
        case year
        case state
        case startedAt = "started_at"
    }
}

public struct BenchConcurrentError: Codable, Sendable {
    public let error: String
    public let activeRunID: String?

    enum CodingKeys: String, CodingKey {
        case error
        case activeRunID = "active_run_id"
    }
}

public struct BenchActiveResponse: Codable, Sendable {
    public let activeRunID: String?

    enum CodingKeys: String, CodingKey {
        case activeRunID = "active_run_id"
    }
}

// MARK: - /snapshot response

public struct BenchSnapshotPerQuestion: Codable, Sendable, Hashable {
    public let idx: Int
    public let id: String
    public let set: String
    public let expected: Int
    public let extracted: Int?
    public let status: String?
    public let attempts: Int?
    public let durationMs: Int?
    public let reasoningTokenCount: Int?
    public let answerTokenCount: Int?

    enum CodingKeys: String, CodingKey {
        case idx
        case id
        case set
        case expected
        case extracted
        case status
        case attempts
        case durationMs = "duration_ms"
        case reasoningTokenCount = "reasoning_token_count"
        case answerTokenCount = "answer_token_count"
    }
}

public struct BenchSnapshotResponse: Codable, Sendable {
    public let runID: String
    public let year: Int
    public let state: String
    public let model: String
    public let total: Int
    public let score: Int
    public let accuracy: Double?
    public let currentIdx: Int
    public let currentAttempt: Int?
    public let currentRequestID: String?
    public let startedAt: Date?
    public let endedAt: Date?
    public let elapsedMs: Int?
    public let paused: Bool
    public let perQuestion: [BenchSnapshotPerQuestion]

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case year
        case state
        case model
        case total
        case score
        case accuracy
        case currentIdx = "current_idx"
        case currentAttempt = "current_attempt"
        case currentRequestID = "current_request_id"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case elapsedMs = "elapsed_ms"
        case paused
        case perQuestion = "per_question"
    }
}

// MARK: - SSE events

/// Decoded SSE event payloads emitted by
/// `GET /v1/mtplx/benchmarks/aime/{run_id}/stream`. The associated values
/// mirror the Python runner's `_push({"event": ...})` payloads.
public enum BenchEvent: Sendable {
    case runStarted(runID: String, total: Int, model: String, startedAt: Date?)
    case questionStarted(
        idx: Int,
        attempt: Int,
        id: String,
        set: String,
        year: Int,
        problem: String
    )
    case reasoningDelta(idx: Int, attempt: Int, text: String)
    case answerDelta(idx: Int, attempt: Int, text: String)
    case questionProgress(
        idx: Int,
        attempt: Int,
        requestID: String,
        metrics: MetricsLatest
    )
    case answerVerificationStarted(
        idx: Int,
        attempt: Int,
        mode: String,
        proposedAnswer: Int?
    )
    case answerVerificationDone(
        idx: Int,
        attempt: Int,
        mode: String,
        proposedAnswer: Int?,
        verifiedAnswer: Int?,
        verifierAnswers: [Int?],
        resolution: String?,
        durationMs: Int?
    )
    case capRecoveryStarted(idx: Int, attempt: Int, requestID: String, mode: String)
    case questionDone(
        idx: Int,
        attempt: Int,
        id: String,
        extracted: Int?,
        expected: Int,
        status: QuestionStatus,
        durationMs: Int?,
        reasoningTokenCount: Int,
        answerTokenCount: Int
    )
    case runPaused(runID: String)
    case runResumed(runID: String)
    case runCancelled(runID: String, score: Int, total: Int)
    case runDone(
        runID: String,
        score: Int,
        total: Int,
        accuracy: Double?,
        durationMs: Int?
    )
    case error(message: String, recoverable: Bool)
    case keepAlive
}

// MARK: - Live answer grading (Swift port of validators/aime.py)

/// Swift implementation of the Python `extract_boxed` regex so the
/// BenchmarkOrchestrator can update the live answer hero numeral the
/// instant the model writes `\boxed{...}` - without waiting for the
/// terminal `question_done` event.
public enum BenchmarkGrader: Sendable {
    /// Permissive `\boxed{...}` regex with last-match-wins semantics.
    /// Matches the Python validator's behaviour exactly.
    private static let boxedRegex: NSRegularExpression = {
        let pattern = #"\\boxed\s*\{\s*\\?,?\s*(-?\d{1,4})\s*\\?,?\s*\}"#
        return try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()

    /// Fallback for "the answer is N" / "final answer is N" prose.
    private static let proseRegex: NSRegularExpression = {
        let pattern = #"(?:final\s+answer\s+is|the\s+answer\s+is|answer\s+is|answer\s*[:=])\s*\$?\s*(-?\d{1,4})"#
        return try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()

    private static let fallbackTailChars = 400

    public static func extractBoxed(_ text: String) -> Int? {
        guard !text.isEmpty else { return nil }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        let matches = boxedRegex.matches(in: text, range: range)
        if let last = matches.last, last.numberOfRanges >= 2 {
            let group = ns.substring(with: last.range(at: 1))
            if let value = Int(group) { return value }
        }
        // Tail-only prose fallback.
        let tailStart = max(0, ns.length - fallbackTailChars)
        let tailRange = NSRange(location: tailStart, length: ns.length - tailStart)
        let proseMatches = proseRegex.matches(in: text, range: tailRange)
        if let last = proseMatches.last, last.numberOfRanges >= 2 {
            let group = ns.substring(with: last.range(at: 1))
            if let value = Int(group) { return value }
        }
        return nil
    }

    public static func grade(extracted: Int?, expected: Int) -> QuestionStatus {
        guard let extracted else { return .abstain }
        return extracted == expected ? .correct : .wrong
    }
}
