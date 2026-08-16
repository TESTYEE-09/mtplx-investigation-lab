import XCTest
@testable import MTPLXAppCore

final class AIMETelemetryDisplayPolicyTests: XCTestCase {
    func testLiveDecodeAppearsAsSoonAsARealSampleExists() {
        XCTAssertEqual(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: 51.2,
                source: "inflight_exact",
                completionTokens: 1
            ),
            51.2
        )
    }

    func testCompletedDecodeDoesNotUseLiveWarmupFloor() {
        XCTAssertEqual(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: 39.0,
                source: "latest_exact_completed",
                completionTokens: 12
            ),
            39.0
        )
    }

    func testInvalidCandidateIsHidden() {
        XCTAssertNil(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: .infinity,
                source: "inflight_exact",
                completionTokens: 999
            )
        )
        XCTAssertNil(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: 0,
                source: "inflight_exact",
                completionTokens: 999
            )
        )
    }

    func testOptionalWarmupFloorStillWorksForNonAIMESurfaces() {
        XCTAssertNil(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: 51.2,
                source: "inflight_exact",
                completionTokens: 31,
                minimumCompletionTokens: 32
            )
        )
        XCTAssertEqual(
            AIMETelemetryDisplayPolicy.displayedDecodeTokS(
                candidate: 51.2,
                source: "inflight_exact",
                completionTokens: 32,
                minimumCompletionTokens: 32
            ),
            51.2
        )
    }
}
