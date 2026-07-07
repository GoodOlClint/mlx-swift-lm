// Copyright © 2025 Apple Inc.

import XCTest

@testable import MLXLMCommon

/// CI-safe coverage for `TriAttentionKVCache`'s self-contained
/// prompt-cache round-trip — the `metaState`/`fromState` codec that
/// reconstructs the eviction policy + bookkeeping — plus the
/// `resolveTriAttentionScheme`/`applyKVScheme` registration seam. Pure
/// metadata (empty `state`), so no MLX runtime is needed; the full
/// populated-array round-trip is exercised by the model-on e2e (there is
/// no CI-safe surface for MLXArray IO).
final class TriAttentionCacheTests: XCTestCase {

    func testMetaStateRoundTripsConfigAndBookkeeping() throws {
        // 9th field = the absolutePosition RoPE counter.
        let meta = ["4096", "64", "max", "false", "0", "0", "false", "0", "0"]
        let cache = try TriAttentionKVCache.fromState(
            state: [], metaState: meta)
        XCTAssertEqual(cache.config.kvBudget, 4096)
        XCTAssertEqual(cache.config.divideLength, 64)
        XCTAssertEqual(cache.config.scoreAggregation, .max)
        XCTAssertFalse(cache.config.prefillPin)
        XCTAssertEqual(cache.offset, 0)
        // Round-trip identity: the getter reproduces the input exactly.
        XCTAssertEqual(cache.metaState, meta)
    }

    func testMetaStatePreservesPrefillAndStepCounters() throws {
        // Post-eviction: 512 tokens seen (absolutePosition), 300 retained
        // (offset) — the split that keeps RoPE on the absolute position.
        let meta = [
            "2048", "128", "mean", "true", "300", "256", "true", "44", "512",
        ]
        let cache = try TriAttentionKVCache.fromState(
            state: [], metaState: meta)
        XCTAssertTrue(cache.config.prefillPin)
        XCTAssertEqual(cache.config.scoreAggregation, .mean)
        XCTAssertEqual(cache.offset, 300)
        XCTAssertEqual(cache.metaState, meta)
    }

    /// A legacy 8-field meta loads, defaulting absolutePosition to offset,
    /// and re-emits in the 9-field format.
    func testLegacyEightFieldMetaDefaultsAbsolutePosition() throws {
        let legacy = ["2048", "128", "mean", "true", "120", "16", "true", "9"]
        let cache = try TriAttentionKVCache.fromState(
            state: [], metaState: legacy)
        XCTAssertEqual(cache.offset, 120)
        XCTAssertEqual(cache.metaState, legacy + ["120"])
    }

    func testFromStateRejectsTruncatedMetaState() {
        XCTAssertThrowsError(
            try TriAttentionKVCache.fromState(
                state: [], metaState: ["2048", "128", "mean"]))
    }

    // MARK: - Geometry validation on restore

    func testFromStateRejectsZeroDivideLength() {
        // divideLength 0 would trap on `% config.divideLength`.
        XCTAssertThrowsError(
            try TriAttentionKVCache.fromState(
                state: [],
                metaState: [
                    "2048", "0", "mean", "true", "10", "0", "true", "1", "10",
                ]))
    }

    func testFromStateRejectsPrefixLengthExceedingOffset() {
        XCTAssertThrowsError(
            try TriAttentionKVCache.fromState(
                state: [],
                metaState: [
                    "2048", "128", "mean", "true", "10", "20", "true", "1", "10",
                ]))
    }

    func testFromStateRejectsAbsolutePositionBelowOffset() {
        // absolutePosition must never be < the retained-count offset.
        XCTAssertThrowsError(
            try TriAttentionKVCache.fromState(
                state: [],
                metaState: [
                    "2048", "128", "mean", "true", "300", "16", "true", "1", "10",
                ]))
    }

    // MARK: - kvScheme resolution

    func testResolveBareSchemeReturnsDefaults() {
        let config = resolveTriAttentionScheme("triattention")
        XCTAssertNotNil(config)
        XCTAssertEqual(config?.kvBudget, 2048)
        XCTAssertEqual(config?.divideLength, 128)
        XCTAssertEqual(config?.scoreAggregation, .mean)
        XCTAssertEqual(config?.prefillPin, true)
    }

    func testResolveSchemeParsesOverrides() {
        let config = resolveTriAttentionScheme(
            "triattention:kvBudget=4096,divideLength=64,agg=max,pin=false")
        XCTAssertEqual(config?.kvBudget, 4096)
        XCTAssertEqual(config?.divideLength, 64)
        XCTAssertEqual(config?.scoreAggregation, .max)
        XCTAssertEqual(config?.prefillPin, false)
    }

    func testResolveSchemeIgnoresMalformedPairs() {
        // Bad value / unknown key / missing "=" are all skipped; valid
        // pairs still apply and the rest fall back to defaults.
        let config = resolveTriAttentionScheme(
            "triattention:kvBudget=notanint,agg=max,bogus,unknown=1")
        XCTAssertEqual(config?.kvBudget, 2048)  // malformed → default
        XCTAssertEqual(config?.scoreAggregation, .max)  // valid override applied
        XCTAssertEqual(config?.divideLength, 128)  // untouched → default
    }

    func testResolveNonTriAttentionSchemesReturnNil() {
        XCTAssertNil(resolveTriAttentionScheme("affine4"))
        XCTAssertNil(resolveTriAttentionScheme(nil))
        XCTAssertNil(resolveTriAttentionScheme("triattentionish"))
    }

    // MARK: - applyKVScheme (the registration seam that the four
    // fresh-cache call sites exercise)

    func testApplyKVSchemeSwapsSimpleCachesAndLeavesOthers() {
        var params = GenerateParameters()
        params.kvScheme = "triattention:kvBudget=4096,agg=max"
        let input: [KVCache] = [KVCacheSimple(), MambaCache()]

        let result = applyKVScheme(input, parameters: params)

        let tri = result[0] as? TriAttentionKVCache
        XCTAssertNotNil(tri, "KVCacheSimple should become TriAttentionKVCache")
        XCTAssertEqual(tri?.config.kvBudget, 4096)
        XCTAssertEqual(tri?.config.scoreAggregation, .max)
        XCTAssertTrue(result[1] is MambaCache, "non-simple cache untouched")
    }

    func testApplyKVSchemeIsIdentityForNonTriAttentionSchemes() {
        var params = GenerateParameters()
        params.kvScheme = "affine4"
        let input: [KVCache] = [KVCacheSimple(), MambaCache()]

        let result = applyKVScheme(input, parameters: params)

        XCTAssertTrue(result[0] is KVCacheSimple)
        XCTAssertFalse(result[0] is TriAttentionKVCache)
        XCTAssertTrue(result[1] is MambaCache)
    }

    func testApplyKVSchemeIsIdentityForNilParameters() {
        let input: [KVCache] = [KVCacheSimple()]
        let result = applyKVScheme(input, parameters: nil)
        XCTAssertTrue(result[0] is KVCacheSimple)
        XCTAssertFalse(result[0] is TriAttentionKVCache)
    }
}
