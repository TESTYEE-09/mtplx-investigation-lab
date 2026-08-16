import XCTest
@testable import MTPLXAppCore

final class AIMEDiagnosticsTests: XCTestCase {
    func testDiagnosticsGateParsesIntentionalTruthsOnly() {
        XCTAssertTrue(AIMEDiagnostics.isEnabled(environment: ["MTPLX_AIME_DIAGNOSTICS": "1"]))
        XCTAssertTrue(AIMEDiagnostics.isEnabled(environment: ["MTPLX_AIME_DIAGNOSTICS": "true"]))
        XCTAssertTrue(AIMEDiagnostics.isEnabled(environment: ["MTPLX_AIME_DIAGNOSTICS": "YES"]))

        XCTAssertFalse(AIMEDiagnostics.isEnabled(environment: [:]))
        XCTAssertFalse(AIMEDiagnostics.isEnabled(environment: ["MTPLX_AIME_DIAGNOSTICS": "0"]))
        XCTAssertFalse(AIMEDiagnostics.isEnabled(environment: ["MTPLX_AIME_DIAGNOSTICS": "off"]))
    }

    func testRenderModeCanBeChangedWithoutDiagnosticsWriter() {
        XCTAssertEqual(
            AIMEDiagnostics.renderMode(environment: ["MTPLX_AIME_RENDER_MODE": "hidden"]),
            .hidden
        )
        XCTAssertEqual(
            AIMEDiagnostics.renderMode(environment: [
                "MTPLX_AIME_DIAGNOSTICS": "1",
                "MTPLX_AIME_RENDER_MODE": "hidden"
            ]),
            .hidden
        )
        XCTAssertEqual(
            AIMEDiagnostics.renderMode(environment: [
                "MTPLX_AIME_DIAGNOSTICS": "1",
                "MTPLX_AIME_RENDER_MODE": "tail_latex"
            ]),
            .tailLatex
        )
        XCTAssertEqual(
            AIMEDiagnostics.renderMode(environment: [
                "MTPLX_AIME_DIAGNOSTICS": "1",
                "MTPLX_AIME_RENDER_MODE": "bogus"
            ]),
            .tailLatex
        )
    }

    func testTailBlockLimitIsBounded() {
        XCTAssertEqual(AIMEDiagnostics.tailBlockLimit(environment: [:]), 48)
        XCTAssertEqual(
            AIMEDiagnostics.tailBlockLimit(environment: ["MTPLX_AIME_TAIL_BLOCKS": "8"]),
            20
        )
        XCTAssertEqual(
            AIMEDiagnostics.tailBlockLimit(environment: ["MTPLX_AIME_TAIL_BLOCKS": "120"]),
            120
        )
        XCTAssertEqual(
            AIMEDiagnostics.tailBlockLimit(environment: ["MTPLX_AIME_TAIL_BLOCKS": "5000"]),
            1_000
        )
    }
}
