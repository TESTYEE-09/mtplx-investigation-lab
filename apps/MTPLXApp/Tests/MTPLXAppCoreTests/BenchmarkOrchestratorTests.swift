import XCTest
@testable import MTPLXAppCore

/// State-machine tests for `BenchmarkOrchestrator` with a scripted SSE
/// stream. No daemon required.
@MainActor
final class BenchmarkOrchestratorTests: XCTestCase {

    // MARK: - Test seam: scripted stream factory

    private final class StubURLProtocol: URLProtocol {
        nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

        override class func canInit(with request: URLRequest) -> Bool {
            true
        }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest {
            request
        }

        override func startLoading() {
            guard let handler = Self.handler else {
                client?.urlProtocol(self, didFailWithError: MTPLXAPIClientError.invalidResponse)
                return
            }
            do {
                let (response, data) = try handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }

        override func stopLoading() {}
    }

    private final class ScriptedStream: BenchmarkStreaming, @unchecked Sendable {
        let events: [BenchEvent]
        let interEventDelayNs: UInt64
        init(_ events: [BenchEvent], interEventDelayNs: UInt64 = 1_000_000) {
            self.events = events
            self.interEventDelayNs = interEventDelayNs
        }
        func consume(_ handler: @escaping @Sendable (BenchEvent) async -> Void) async throws {
            for event in events {
                try? await Task.sleep(nanoseconds: interEventDelayNs)
                if Task.isCancelled { return }
                await handler(event)
                if event.isTerminal { return }
            }
        }
    }

    nonisolated private static func requestBodyData(_ request: URLRequest) -> Data {
        if let httpBody = request.httpBody {
            return httpBody
        }
        guard let stream = request.httpBodyStream else {
            return Data()
        }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        stream.open()
        defer { stream.close() }
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }

    private func makeOrchestrator(
        events: [BenchEvent]
    ) -> BenchmarkOrchestrator {
        // The orchestrator only calls apiClientProvider() when start()
        // is invoked. For tests we wire it to a never-used client and
        // call `attachStream` indirectly by handing the orchestrator a
        // scripted stream factory.
        let dummyURL = URL(string: "http://127.0.0.1:8000")!
        let client = MTPLXAPIClient(baseURL: dummyURL)
        return BenchmarkOrchestrator(
            apiClientProvider: { client },
            streamFactory: { _, _ in ScriptedStream(events) }
        )
    }

    /// Manually prime the orchestrator's stream by calling its internal
    /// `attachStream` path. Since `attachStream` is private, we
    /// reproduce the flow: prime via runStarted, then handle events.
    /// Easier: just start() and have the start() seed runStarted via
    /// a scripted run that begins with run_started.
    ///
    /// Even easier: call start() but make the API call fake-succeed by
    /// using a private hook. For now we exercise the flow by calling
    /// handleStreamEvent indirectly: ScriptedStream yields events that
    /// drive the state.
    private func runScript(
        orchestrator: BenchmarkOrchestrator,
        events: [BenchEvent],
        runId: String = "aime-test-1"
    ) async {
        // Prime by manually setting up internal state via runStarted +
        // attaching a fresh ScriptedStream is awkward without exposing
        // attachStream. Instead, we drive events via a publicly testable
        // path: call `start()` against a mock that succeeds, OR
        // expose a test-only entry point. For now, the cleanest seam is
        // to invoke `start()` and let the SCRIPTED stream factory drive
        // everything (start() will fail at the network call, but we
        // bypass that by using a special factory).
        //
        // Practical approach: drive directly via `attachStream` made
        // internal-visible for tests. We do that by extension below.
        await orchestrator.testDriveScript(
            runId: runId,
            events: events
        )
    }

    // MARK: - Tests

    func testFullRunCompletesEndToEnd() async throws {
        let events: [BenchEvent] = [
            .runStarted(runID: "aime-test-1", total: 3, model: "test-model", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .reasoningDelta(idx: 1, attempt: 1, text: "step a"),
            .answerDelta(idx: 1, attempt: 1, text: "Answer: \\boxed{10}"),
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 10, expected: 10, status: .correct, durationMs: 100, reasoningTokenCount: 5, answerTokenCount: 4),
            .questionStarted(idx: 2, attempt: 1, id: "2026-I-2", set: "AIME I", year: 2026, problem: "Q2"),
            .answerDelta(idx: 2, attempt: 1, text: "\\boxed{999}"),
            .questionDone(idx: 2, attempt: 1, id: "2026-I-2", extracted: 999, expected: 20, status: .wrong, durationMs: 100, reasoningTokenCount: 0, answerTokenCount: 4),
            .questionStarted(idx: 3, attempt: 1, id: "2026-I-3", set: "AIME I", year: 2026, problem: "Q3"),
            .questionDone(idx: 3, attempt: 1, id: "2026-I-3", extracted: nil, expected: 30, status: .abstain, durationMs: 100, reasoningTokenCount: 0, answerTokenCount: 0),
            .runDone(runID: "aime-test-1", score: 1, total: 3, accuracy: 1.0 / 3.0, durationMs: 300)
        ]

        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)

        // Wait for terminal state.
        try await waitFor(orch, until: { $0.state == .done })

        XCTAssertEqual(orch.state, .done)
        XCTAssertEqual(orch.total, 3)
        XCTAssertEqual(orch.score, 1)
        XCTAssertEqual(orch.resolved, 3)
        XCTAssertEqual(orch.results[0].status, .correct)
        XCTAssertEqual(orch.results[0].extracted, 10)
        XCTAssertEqual(orch.results[1].status, .wrong)
        XCTAssertEqual(orch.results[2].status, .abstain)
        XCTAssertNil(orch.results[2].extracted)
    }

    func testAnswerGradedFlagArrivesOnlyFromQuestionDone() async throws {
        let events: [BenchEvent] = [
            .runStarted(runID: "graded", total: 1, model: "test-model", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .answerDelta(idx: 1, attempt: 1, text: "candidate_answer=10\n\\boxed{10}"),
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 10, expected: 10, status: .correct, durationMs: 100, reasoningTokenCount: 0, answerTokenCount: 4)
        ]

        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events, runId: "graded")
        try await waitFor(orch, until: { $0.liveAnswerIsGraded })

        XCTAssertEqual(orch.liveExtractedAnswer, 10)
        XCTAssertTrue(orch.liveAnswerIsGraded)
    }

    func testCancelPropagates() async throws {
        let events: [BenchEvent] = [
            .runStarted(runID: "x", total: 2, model: "m", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .runCancelled(runID: "x", score: 0, total: 2)
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)
        try await waitFor(orch, until: { $0.state == .cancelled })
        XCTAssertEqual(orch.state, .cancelled)
    }

    func testAIMEStartInheritsDaemonSamplerByDefault() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        defer {
            StubURLProtocol.handler = nil
            session.invalidateAndCancel()
        }

        let baseURL = URL(string: "http://127.0.0.1:8123")!
        let client = MTPLXAPIClient(baseURL: baseURL, session: session)
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/v1/mtplx/benchmarks/aime/start")
            let body = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Self.requestBodyData(request))
                    as? [String: Any]
            )
            XCTAssertEqual(body["year"] as? Int, 2026)
            XCTAssertNil(body["temperature"])
            XCTAssertNil(body["top_p"])
            XCTAssertNil(body["top_k"])
            XCTAssertNil(body["max_tokens"])
            XCTAssertNil(body["question_limit"])

            let status = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (
                status,
                Data(
                    """
                    {
                      "run_id": "aime-2026-test",
                      "total": 30,
                      "model": "test-model",
                      "year": 2026,
                      "state": "running",
                      "started_at": "2026-05-27T10:00:00Z"
                    }
                    """.utf8
                )
            )
        }

        let response = try await client.aimeStart()
        XCTAssertEqual(response.runID, "aime-2026-test")
    }

    func testAIMEStartCanSendExplicitSamplerForAblations() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        defer {
            StubURLProtocol.handler = nil
            session.invalidateAndCancel()
        }

        let baseURL = URL(string: "http://127.0.0.1:8123")!
        let client = MTPLXAPIClient(baseURL: baseURL, session: session)
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/v1/mtplx/benchmarks/aime/start")
            let body = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Self.requestBodyData(request))
                    as? [String: Any]
            )
            XCTAssertEqual(body["year"] as? Int, 2026)
            XCTAssertEqual(body["temperature"] as? Double, 0.6)
            XCTAssertEqual(body["top_p"] as? Double, 0.95)
            XCTAssertEqual(body["top_k"] as? Int, 20)
            XCTAssertEqual(body["max_tokens"] as? Int, 4096)
            XCTAssertEqual(body["enable_thinking"] as? Bool, true)
            XCTAssertEqual(body["question_process_isolation"] as? String, "per_question")
            XCTAssertEqual(body["question_limit"] as? Int, 3)

            let status = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (
                status,
                Data(
                    """
                    {
                      "run_id": "aime-2026-test",
                      "total": 30,
                      "model": "test-model",
                      "year": 2026,
                      "state": "running",
                      "started_at": "2026-05-27T10:00:00Z"
                    }
                    """.utf8
                )
            )
        }

        let response = try await client.aimeStart(
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            maxTokens: 4096,
            enableThinking: true,
            questionProcessIsolation: "per_question",
            questionLimit: 3
        )
        XCTAssertEqual(response.runID, "aime-2026-test")
    }

    func testBenchmarkStartOptionsFollowLiveSettings() {
        let options = BenchmarkStartOptions(
            settings: MutableSettings(
                temperature: 1.0,
                topP: 0.9,
                topK: 64,
                enableThinking: true,
                reasoning: "off"
            )
        )

        XCTAssertEqual(options.temperature, 1.0)
        XCTAssertEqual(options.topP, 0.9)
        XCTAssertEqual(options.topK, 64)
        XCTAssertEqual(options.enableThinking, false)
        XCTAssertEqual(options.questionProcessIsolation, "per_question")
        XCTAssertNil(options.questionLimit)
    }

    func testAIMESkipUsesRunSpecificEndpoint() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        defer {
            StubURLProtocol.handler = nil
            session.invalidateAndCancel()
        }

        let baseURL = URL(string: "http://127.0.0.1:8123")!
        let client = MTPLXAPIClient(baseURL: baseURL, session: session)
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.url?.path, "/v1/mtplx/benchmarks/aime/aime-2026-test/skip")
            let status = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (status, Data(Self.runningSnapshotJSON(runID: "aime-2026-test").utf8))
        }

        let response = try await client.aimeSkip(runId: "aime-2026-test")
        XCTAssertEqual(response.runID, "aime-2026-test")
    }

    func testSkipCurrentSetsPendingAndClearsOnQuestionDone() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        defer {
            StubURLProtocol.handler = nil
            session.invalidateAndCancel()
        }

        let baseURL = URL(string: "http://127.0.0.1:8123")!
        let client = MTPLXAPIClient(baseURL: baseURL, session: session)
        StubURLProtocol.handler = { request in
            let status = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            if request.httpMethod == "POST" {
                XCTAssertEqual(request.url?.path, "/v1/mtplx/benchmarks/aime/x/skip")
            }
            return (status, Data(Self.runningSnapshotJSON(runID: "x").utf8))
        }

        let orch = BenchmarkOrchestrator(
            apiClientProvider: { client },
            streamFactory: { _, _ in ScriptedStream([]) }
        )
        await orch.testHandleEvent(.runStarted(runID: "x", total: 2, model: "m", startedAt: Date()))
        await orch.testHandleEvent(
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1")
        )

        await orch.skipCurrent()
        XCTAssertTrue(orch.skipPending)

        await orch.testHandleEvent(
            .questionDone(
                idx: 1,
                attempt: 1,
                id: "2026-I-1",
                extracted: nil,
                expected: 277,
                status: .abstain,
                durationMs: 42,
                reasoningTokenCount: 12,
                answerTokenCount: 0
            )
        )
        XCTAssertFalse(orch.skipPending)
        XCTAssertEqual(orch.results[0].status, .abstain)
    }

    func testCancelReconcilesReturnedSnapshotBeforeSummary() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)
        defer {
            StubURLProtocol.handler = nil
            session.invalidateAndCancel()
        }

        let baseURL = URL(string: "http://127.0.0.1:8123")!
        let client = MTPLXAPIClient(baseURL: baseURL, session: session)
        let snapshot = Self.cancelledSnapshotJSON(runID: "x", correctCount: 3)
        StubURLProtocol.handler = { request in
            let status = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (status, Data(snapshot.utf8))
        }

        let orch = BenchmarkOrchestrator(
            apiClientProvider: { client },
            streamFactory: { _, _ in ScriptedStream([]) }
        )
        await orch.testHandleEvent(.runStarted(runID: "x", total: 30, model: "m", startedAt: Date()))
        await orch.testHandleEvent(
            .questionDone(
                idx: 1,
                attempt: 1,
                id: "2026-I-1",
                extracted: 277,
                expected: 277,
                status: .correct,
                durationMs: 1,
                reasoningTokenCount: 0,
                answerTokenCount: 1
            )
        )

        XCTAssertEqual(orch.score, 1)
        XCTAssertEqual(orch.resolved, 1)

        await orch.cancel()
        await orch.testHandleEvent(.error(message: "cancelled", recoverable: false))

        try await waitFor(orch, until: { $0.state == .cancelled && $0.score == 3 && $0.resolved == 3 })
        XCTAssertNil(orch.lastError)
        XCTAssertEqual(orch.results.prefix(3).map(\.status), [.correct, .correct, .correct])
        XCTAssertEqual(orch.results[1].extracted, 62)
        XCTAssertEqual(orch.results[2].extracted, 79)
    }

    func testPauseResumeFlipsStateAndClearsPending() async throws {
        let events: [BenchEvent] = [
            .runStarted(runID: "x", total: 2, model: "m", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .runPaused(runID: "x"),
            .runResumed(runID: "x"),
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 1, expected: 1, status: .correct, durationMs: 1, reasoningTokenCount: 0, answerTokenCount: 1),
            .runDone(runID: "x", score: 1, total: 2, accuracy: 1.0, durationMs: 1)
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)
        try await waitFor(orch, until: { $0.state == .done })
        XCTAssertEqual(orch.state, .done)
        XCTAssertFalse(orch.pausePending)
    }

    func testPrepareForPresentationClearsIdleStartError() async throws {
        struct StartFailure: LocalizedError {
            var errorDescription: String? { "daemon is not ready" }
        }

        let client = MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        let orch = BenchmarkOrchestrator(
            apiClientProvider: { client },
            daemonReadinessProvider: { throw StartFailure() },
            streamFactory: { _, _ in ScriptedStream([]) }
        )

        await orch.start(questionLimit: 3)

        XCTAssertEqual(orch.state, .idle)
        XCTAssertNotNil(orch.lastError)

        orch.prepareForPresentation()

        XCTAssertNil(orch.lastError)
    }

    func testRetryAttemptIgnoresStalePausedDeltas() async throws {
        let orch = makeOrchestrator(events: [])
        await orch.testHandleEvent(.runStarted(runID: "x", total: 1, model: "m", startedAt: Date()))
        await orch.testHandleEvent(
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1")
        )
        await orch.testHandleEvent(.reasoningDelta(idx: 1, attempt: 1, text: "$k=1$: stale"))
        await orch.testHandleEvent(.runPaused(runID: "x"))
        XCTAssertEqual(orch.state, .paused)

        await orch.testHandleEvent(.runResumed(runID: "x"))
        await orch.testHandleEvent(
            .questionStarted(idx: 1, attempt: 2, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1")
        )
        await orch.testHandleEvent(.reasoningDelta(idx: 1, attempt: 1, text: " old token"))
        await orch.testHandleEvent(.answerDelta(idx: 1, attempt: 1, text: "\\boxed{999}"))
        await orch.testHandleEvent(.reasoningDelta(idx: 1, attempt: 2, text: "$k=2$: fresh"))
        await orch.testHandleEvent(.answerDelta(idx: 1, attempt: 2, text: "\\boxed{277}"))
        await orch.testHandleEvent(
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 999, expected: 277, status: .wrong, durationMs: 1, reasoningTokenCount: 1, answerTokenCount: 1)
        )
        await orch.testHandleEvent(
            .questionDone(idx: 1, attempt: 2, id: "2026-I-1", extracted: 277, expected: 277, status: .correct, durationMs: 1, reasoningTokenCount: 1, answerTokenCount: 1)
        )

        try await waitFor(orch, until: { $0.liveReasoning.contains("fresh") })
        XCTAssertFalse(orch.liveReasoning.contains("old token"))
        XCTAssertEqual(orch.liveExtractedAnswer, 277)
        XCTAssertEqual(orch.results[0].status, .correct)
        XCTAssertEqual(orch.results[0].extracted, 277)
    }

    func testLiveAnswerExtractionFires() async throws {
        // The orchestrator should run BenchmarkGrader on every answer_delta
        // so liveExtractedAnswer flips the instant a `\boxed{...}` arrives.
        let events: [BenchEvent] = [
            .runStarted(runID: "x", total: 1, model: "m", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .answerDelta(idx: 1, attempt: 1, text: "Let me think... "),
            .answerDelta(idx: 1, attempt: 1, text: "Therefore the answer is \\boxed{277}."),
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 277, expected: 277, status: .correct, durationMs: 1, reasoningTokenCount: 0, answerTokenCount: 8),
            .runDone(runID: "x", score: 1, total: 1, accuracy: 1.0, durationMs: 1)
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)
        try await waitFor(orch, until: { $0.state == .done })
        XCTAssertEqual(orch.liveExtractedAnswer, 277)
        XCTAssertEqual(orch.score, 1)
    }

    func testAnswerVerificationEventsSurfaceLiveState() async throws {
        let orch = makeOrchestrator(events: [])
        await orch.testHandleEvent(.runStarted(runID: "x", total: 2, model: "m", startedAt: Date()))
        await orch.testHandleEvent(
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1")
        )

        await orch.testHandleEvent(
            .answerVerificationStarted(
                idx: 1,
                attempt: 1,
                mode: "fast_majority",
                proposedAnswer: 57
            )
        )

        XCTAssertEqual(orch.currentVerification?.proposedAnswer, 57)
        XCTAssertEqual(orch.currentVerification?.mode, "fast_majority")
        XCTAssertEqual(orch.currentVerification?.isRunning, true)

        await orch.testHandleEvent(
            .answerVerificationDone(
                idx: 1,
                attempt: 1,
                mode: "fast_majority",
                proposedAnswer: 57,
                verifiedAnswer: 62,
                verifierAnswers: [62, 62],
                resolution: "majority_corrected",
                durationMs: 12000
            )
        )

        XCTAssertEqual(orch.currentVerification?.verifiedAnswer, 62)
        XCTAssertEqual(orch.currentVerification?.verifierAnswers, [62, 62])
        XCTAssertEqual(orch.currentVerification?.resolution, "majority_corrected")
        XCTAssertEqual(orch.currentVerification?.correctedAnswer, true)

        await orch.testHandleEvent(
            .answerVerificationDone(
                idx: 1,
                attempt: 1,
                mode: "fast_majority",
                proposedAnswer: 62,
                verifiedAnswer: 62,
                verifierAnswers: [nil, nil],
                resolution: "no_verifier_answer_keep_proposed",
                durationMs: 90000
            )
        )

        XCTAssertEqual(orch.currentVerification?.hasVerifierAnswer, false)
        XCTAssertEqual(orch.currentVerification?.correctedAnswer, false)
        XCTAssertEqual(orch.currentVerification?.verifierAnswers, [nil, nil])

        await orch.testHandleEvent(
            .questionStarted(idx: 2, attempt: 1, id: "2026-I-2", set: "AIME I", year: 2026, problem: "Q2")
        )
        XCTAssertNil(orch.currentVerification)
    }

    func testCapRecoveryStartedMovesLiveMetricsRequest() async throws {
        let orch = makeOrchestrator(events: [])
        await orch.testHandleEvent(.runStarted(runID: "x", total: 1, model: "m", startedAt: Date()))
        await orch.testHandleEvent(
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1")
        )
        XCTAssertEqual(orch.currentRequestID, "chatcmpl-x-q1-a1")

        await orch.testHandleEvent(
            .answerVerificationStarted(
                idx: 1,
                attempt: 1,
                mode: "fast_majority",
                proposedAnswer: nil
            )
        )
        await orch.testHandleEvent(
            .capRecoveryStarted(
                idx: 1,
                attempt: 1,
                requestID: "chatcmpl-x-q1-a1-finalize1",
                mode: "visible_finalizer"
            )
        )

        XCTAssertEqual(orch.currentRequestID, "chatcmpl-x-q1-a1-finalize1")
        XCTAssertNil(orch.currentVerification)
    }

    func testGraderExtractsLastBoxed() {
        XCTAssertEqual(BenchmarkGrader.extractBoxed("foo \\boxed{12} bar \\boxed{34}"), 34)
        XCTAssertEqual(BenchmarkGrader.extractBoxed("\\boxed{ 277 }"), 277)
        XCTAssertEqual(BenchmarkGrader.extractBoxed("\\boxed{\\,482\\,}"), 482)
        XCTAssertNil(BenchmarkGrader.extractBoxed("no answer here"))
        XCTAssertEqual(BenchmarkGrader.extractBoxed("the final answer is 99"), 99)
    }

    func testGraderStatusMatches() {
        XCTAssertEqual(BenchmarkGrader.grade(extracted: 5, expected: 5), .correct)
        XCTAssertEqual(BenchmarkGrader.grade(extracted: 5, expected: 6), .wrong)
        XCTAssertEqual(BenchmarkGrader.grade(extracted: nil, expected: 5), .abstain)
    }

    func testAPIClientDefaultDecoderAcceptsAIMEBackendDates() throws {
        let decoder = MTPLXAPIClient.makeDefaultDecoder()

        let started = try decoder.decode(
            BenchStartResponse.self,
            from: Data(
                """
                {
                  "run_id": "aime-2026-test",
                  "total": 30,
                  "model": "mtplx-qwen36-27b-optimized-speed",
                  "year": 2026,
                  "state": "running",
                  "started_at": "2026-05-26T20:59:57.123456Z"
                }
                """.utf8
            )
        )
        XCTAssertEqual(started.runID, "aime-2026-test")
        XCTAssertNotNil(started.startedAt)

        let history = try decoder.decode(
            BenchHistoryResponse.self,
            from: Data(
                """
                {
                  "runs": [
                    {
                      "run_id": "aime-2026-history",
                      "state": "done",
                      "score": 24,
                      "total": 30,
                      "accuracy": 0.8,
                      "duration_ms": 123456,
                      "model": "mtplx-qwen36-27b-optimized-speed",
                      "ended_at": "2026-05-26T21:09:57Z"
                    }
                  ]
                }
                """.utf8
            )
        )
        XCTAssertEqual(history.runs.first?.runID, "aime-2026-history")
        XCTAssertNotNil(history.runs.first?.endedAt)
    }

    func testResetClearsLocalState() async {
        let events: [BenchEvent] = [
            .runStarted(runID: "x", total: 1, model: "m", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .questionDone(idx: 1, attempt: 1, id: "2026-I-1", extracted: 1, expected: 1, status: .correct, durationMs: 1, reasoningTokenCount: 0, answerTokenCount: 1),
            .runDone(runID: "x", score: 1, total: 1, accuracy: 1.0, durationMs: 1)
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)
        try? await waitFor(orch, until: { $0.state == .done })
        orch.reset()
        XCTAssertEqual(orch.state, .idle)
        XCTAssertNil(orch.runID)
        XCTAssertEqual(orch.results.count, 0)
    }

    func testDiscardPresentedRunRemovesSavedTerminalRun() async throws {
        let runID = "mtplx-test-discard-\(UUID().uuidString)"
        let persistedURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("benchmarks", isDirectory: true)
            .appendingPathComponent("aime", isDirectory: true)
            .appendingPathComponent("\(runID).jsonl", isDirectory: false)
        try FileManager.default.createDirectory(
            at: persistedURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("{\"summary\": {}}\n".utf8).write(to: persistedURL)
        defer { try? FileManager.default.removeItem(at: persistedURL) }

        let events: [BenchEvent] = [
            .runStarted(runID: runID, total: 1, model: "m", startedAt: Date()),
            .runDone(runID: runID, score: 0, total: 1, accuracy: 0, durationMs: 1)
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events, runId: runID)
        try await waitFor(orch, until: { $0.state == .done })

        XCTAssertTrue(FileManager.default.fileExists(atPath: persistedURL.path))
        XCTAssertTrue(orch.discardPresentedRun())
        XCTAssertEqual(orch.state, .idle)
        XCTAssertNil(orch.runID)
        XCTAssertFalse(FileManager.default.fileExists(atPath: persistedURL.path))
    }

    func testQuestionProgressUpdatesWorkerTelemetry() async throws {
        let requestID = "chatcmpl-aime-test-q1-a1"
        let events: [BenchEvent] = [
            .runStarted(runID: "x", total: 2, model: "m", startedAt: Date()),
            .questionStarted(idx: 1, attempt: 1, id: "2026-I-1", set: "AIME I", year: 2026, problem: "Q1"),
            .questionProgress(
                idx: 1,
                attempt: 1,
                requestID: requestID,
                metrics: MetricsLatest(values: [
                    "completion_tokens": .number(64),
                    "display_decode_tok_s": .number(48.5)
                ])
            )
        ]
        let orch = makeOrchestrator(events: events)
        await runScript(orchestrator: orch, events: events)
        try await waitFor(orch, until: {
            $0.liveTelemetry.latest?.values["completion_tokens"]?.intValue == 64
        })

        XCTAssertEqual(orch.liveTelemetry.latest?.values["request_id"]?.stringValue, requestID)
        XCTAssertEqual(orch.liveTelemetry.latest?.values["display_decode_tok_s"]?.doubleValue, 48.5)

        await orch.testHandleEvent(
            .questionProgress(
                idx: 2,
                attempt: 1,
                requestID: "stale",
                metrics: MetricsLatest(values: [
                    "completion_tokens": .number(512),
                    "request_id": .string("stale")
                ])
            )
        )
        XCTAssertEqual(orch.liveTelemetry.latest?.values["request_id"]?.stringValue, requestID)

        await orch.testHandleEvent(
            .questionStarted(
                idx: 2,
                attempt: 1,
                id: "2026-I-2",
                set: "AIME I",
                year: 2026,
                problem: "Q2"
            )
        )
        XCTAssertNil(orch.liveTelemetry.latest)
    }

    // MARK: - Helpers

    private func waitFor(
        _ orch: BenchmarkOrchestrator,
        until condition: @MainActor (BenchmarkOrchestrator) -> Bool,
        timeout: TimeInterval = 2.0
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition(orch) { return }
            try await Task.sleep(nanoseconds: 5_000_000)  // 5 ms
        }
        XCTFail("Timed out waiting for orchestrator state: \(orch.state)")
    }

    private static func cancelledSnapshotJSON(runID: String, correctCount: Int) -> String {
        let rows = (1...30).map { idx -> String in
            let answer: Int
            switch idx {
            case 1: answer = 277
            case 2: answer = 62
            case 3: answer = 79
            default: answer = idx
            }
            let completed = idx <= correctCount
            return """
            {
              "idx": \(idx),
              "id": "2026-I-\(idx)",
              "set": "AIME I",
              "expected": \(answer),
              "extracted": \(completed ? String(answer) : "null"),
              "status": "\(completed ? "correct" : "pending")",
              "duration_ms": \(completed ? 100 : 0),
              "reasoning_token_count": 0,
              "answer_token_count": \(completed ? 1 : 0)
            }
            """
        }.joined(separator: ",")
        return """
        {
          "run_id": "\(runID)",
          "year": 2026,
          "state": "cancelled",
          "model": "m",
          "total": 30,
          "score": \(correctCount),
          "accuracy": 1.0,
          "current_idx": 3,
          "started_at": "2026-05-27T09:30:00Z",
          "ended_at": "2026-05-27T09:43:00Z",
          "elapsed_ms": 780000,
          "paused": false,
          "per_question": [\(rows)]
        }
        """
    }

    private static func runningSnapshotJSON(runID: String) -> String {
        """
        {
          "run_id": "\(runID)",
          "year": 2026,
          "state": "running",
          "model": "m",
          "total": 2,
          "score": 0,
          "accuracy": null,
          "current_idx": 1,
          "started_at": "2026-05-27T09:30:00Z",
          "ended_at": null,
          "elapsed_ms": 1000,
          "paused": false,
          "per_question": [
            {
              "idx": 1,
              "id": "2026-I-1",
              "set": "AIME I",
              "expected": 277,
              "extracted": null,
              "status": null,
              "duration_ms": null,
              "reasoning_token_count": 0,
              "answer_token_count": 0
            },
            {
              "idx": 2,
              "id": "2026-I-2",
              "set": "AIME I",
              "expected": 62,
              "extracted": null,
              "status": null,
              "duration_ms": null,
              "reasoning_token_count": 0,
              "answer_token_count": 0
            }
          ]
        }
        """
    }
}

// MARK: - Test seam: drive scripts without going through start()

extension BenchmarkOrchestrator {
    /// Test-only: prime the orchestrator as if start() succeeded, then
    /// pipe a scripted event sequence through `handleStreamEvent`.
    @MainActor
    func testDriveScript(runId: String, events: [BenchEvent]) async {
        // Prime via a synthetic runStarted. The orchestrator's own
        // handler will allocate placeholders + set runID/model/state.
        // We then deliver the rest of the script in order with a tiny
        // sleep so the throttled flush task runs.
        Task { @MainActor [weak self] in
            for event in events {
                try? await Task.sleep(nanoseconds: 1_000_000)  // 1 ms
                await self?.testHandleEvent(event)
            }
        }
    }

    @MainActor
    func testHandleEvent(_ event: BenchEvent) async {
        await self._testHandleStreamEvent(event)
    }
}
