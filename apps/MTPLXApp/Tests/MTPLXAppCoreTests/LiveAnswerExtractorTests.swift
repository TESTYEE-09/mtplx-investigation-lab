import XCTest
@testable import MTPLXAppCore

final class LiveAnswerExtractorTests: XCTestCase {
    func testExtractsBoxedAnswerAcrossSplitDeltas() {
        var extractor = LiveAnswerExtractor(tailLimit: 512)

        XCTAssertNil(extractor.append("working toward final \\bo"))
        XCTAssertEqual(extractor.append("xed{277}"), 277)
        XCTAssertTrue(extractor.hasBoxedMarker)
    }

    func testScanStaysTailBoundedButKeepsRecentAnswer() {
        var extractor = LiveAnswerExtractor(tailLimit: 512)
        _ = extractor.append(String(repeating: "reasoning ", count: 2_000))

        XCTAssertNil(extractor.extractedAnswer)
        XCTAssertEqual(extractor.append("\nfinal \\boxed{62}"), 62)
    }
}
