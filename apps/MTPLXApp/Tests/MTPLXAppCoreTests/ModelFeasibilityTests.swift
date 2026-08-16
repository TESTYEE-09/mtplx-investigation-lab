import XCTest
@testable import MTPLXAppCore

final class ModelFeasibilityTests: XCTestCase {
    private let evaluator = ModelFeasibility()

    private var speed: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-speed" }!
    }
    private var quality: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-quality" }!
    }
    private var fp16: MTPLXModelOption {
        MTPLXModelOption.officialCatalog.first { $0.id == "optimized-speed-fp16" }!
    }

    // Ample disk free for every case below; we exercise the disk gate
    // separately at the bottom of the file.
    private let ampleDiskGiB: Double = 500

    // MARK: - Speed (~17 GiB peak)

    func testSpeedOnLegacy8GBIsInsufficient() {
        let v = evaluator.evaluate(model: speed, chipTier: .legacyApple, ramGiB: 8, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .insufficientMemory(needsGiB: 17.0 * 1.5))
    }

    func testSpeedOnLegacy16GBIsInsufficient() {
        // 16 GiB RAM is below the 17 GiB raw peak — insufficient.
        let v = evaluator.evaluate(model: speed, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testSpeedOnModern24GBIsTightFit() {
        // 24 GiB > 17 (peak) but < 25.5 (safe floor 17 * 1.5).
        let v = evaluator.evaluate(model: speed, chipTier: .modernApple, ramGiB: 24, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testSpeedOnModern36GBIsRecommended() {
        let v = evaluator.evaluate(model: speed, chipTier: .modernApple, ramGiB: 36, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - Quality (~28 GiB peak)

    func testQualityOnLegacy16GBIsInsufficient() {
        let v = evaluator.evaluate(model: quality, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testQualityOnModern36GBIsTightFit() {
        // 36 > 28 (peak) but < 42 (safe floor 28 * 1.5).
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 36, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testQualityOnModern48GBIsRecommended() {
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 48, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    func testQualityOnModern128GBIsRecommended() {
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 128, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - FP16 (~17.5 GiB peak)

    func testFP16OnLegacy16GBIsInsufficient() {
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 16, diskFreeGiB: ampleDiskGiB)
        if case .insufficientMemory = v { return }
        XCTFail("Expected insufficientMemory, got \(v)")
    }

    func testFP16OnLegacy24GBIsTightFit() {
        // 24 > 17.5 (peak) but < 26.25 (safe floor 17.5 * 1.5).
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 24, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .tightFit)
    }

    func testFP16OnLegacy32GBIsRecommended() {
        let v = evaluator.evaluate(model: fp16, chipTier: .legacyApple, ramGiB: 32, diskFreeGiB: ampleDiskGiB)
        XCTAssertEqual(v, .recommended)
    }

    // MARK: - Intel always insufficient

    func testIntelMacAlwaysInsufficientForCuratedModels() {
        for model in [speed, quality, fp16] {
            let v = evaluator.evaluate(model: model, chipTier: .intel, ramGiB: 64, diskFreeGiB: ampleDiskGiB)
            if case .insufficientMemory = v { continue }
            XCTFail("\(model.id) on Intel should be insufficientMemory, got \(v)")
        }
    }

    // MARK: - Disk gate beats memory gate

    func testInsufficientDiskBlocksEvenWhenMemoryFits() {
        // 128 GiB RAM is plenty for Quality, but 20 GiB free disk is
        // not enough for 28 GB * 2.5 = 70 GiB required.
        let v = evaluator.evaluate(model: quality, chipTier: .modernApple, ramGiB: 128, diskFreeGiB: 20)
        if case .insufficientDisk(let needs) = v {
            XCTAssertGreaterThan(needs, 60, "Quality 28 GB on disk × 2.5 should require > 60 GiB free")
            return
        }
        XCTFail("Expected insufficientDisk, got \(v)")
    }
}
