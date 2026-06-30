// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) @testable import MLXLMCommon
import MLXNN
import Testing

// MARK: - Synthetic mocks

/// DSpark drafter stub: returns a fixed token pattern so the iterator's
/// draft/verify/accept flow can be exercised without real model numerics
/// (those are pinned separately by Qwen3DSparkParityTests).
private final class MockDSparkDrafter: DSparkDrafting {
    var targetLayerIds: [Int] { [0, 1] }
    var blockSize: Int { 4 }
    let draftedTokenValue: Int32
    private(set) var draftBlockCallCount = 0
    /// Context sequence length seen on each call — pins the accumulation/trim.
    private(set) var receivedContextLengths: [Int] = []

    init(draftedTokenValue: Int32 = 7) { self.draftedTokenValue = draftedTokenValue }

    func draftBlock(bonus: MLXArray, context: MLXArray, numDraft: Int) -> MLXArray {
        draftBlockCallCount += 1
        receivedContextLengths.append(context.dim(1))
        let batch = context.dim(0)
        return MLXArray(
            Array(repeating: draftedTokenValue, count: numDraft * batch), [batch, numDraft])
    }
}

/// Minimal `LanguageModel` that emits DSpark layer hiddens when asked, and
/// returns one-hot logits driving a planned greedy token sequence.
private final class MockDSparkTarget: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }
    var nextLogitTokens: [Int32]
    var perPositionIndex = 0
    var omitLayerHiddens = false
    private(set) var lastIncomingCaptureLayers: [Int]?
    private(set) var emittedLayerSpans: [Int] = []

    init(nextLogitTokens: [Int32]) {
        self.nextLogitTokens = nextLogitTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(positions: inputs.dim(-1))
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)
        if let first = cache?.first as? DSparkCountingCache { first.offset += positions }

        if !omitLayerHiddens, state?[mtpEmitFlagKey] ?? false {
            lastIncomingCaptureLayers = state?[mtpCaptureLayersKey]
            emittedLayerSpans.append(positions)
            var out = LMOutput.State()
            out[mtpLayerHiddenStatesKey] = [
                0: MLXArray.zeros([1, positions, 4]),
                1: MLXArray.zeros([1, positions, 4]),
            ]
            return LMOutput(logits: logits, state: out)
        }
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [DSparkCountingCache()] }

    private func makeLogits(positions: Int) -> MLXArray {
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let idx = perPositionIndex + i
            let tok = idx < nextLogitTokens.count ? Int(nextLogitTokens[idx]) : 0
            data[i * vocab + tok] = 100
        }
        perPositionIndex += positions
        return MLXArray(data, [1, positions, vocab])
    }
}

private final class DSparkCountingCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool { true }
    @discardableResult func trim(_ n: Int) -> Int {
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    { .none }
    func copy() -> any KVCache {
        let c = DSparkCountingCache()
        c.offset = offset
        return c
    }
    func innerState() -> [MLXArray] { [] }
}

// MARK: - Lossless accept invariant

@Test
func testDSparkAllDraftsAcceptedMatchesGreedy() throws {
    // Prefill bonus 7; verify [7,7,7,9]; drafter proposes [7,7,7] → 3 accepted
    // + correction 9. Emitted stream == greedy: [7,7,7,7,9].
    let main = MockDSparkTarget(nextLogitTokens: [0, 0, 7, 7, 7, 7, 9])
    let drafter = MockDSparkDrafter(draftedTokenValue: 7)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try DSparkTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8), blockSize: 4)

    let toks = [iter.next(), iter.next(), iter.next(), iter.next(), iter.next()]
    #expect(toks == [7, 7, 7, 7, 9])
    #expect(iter.tokenCount == 5)
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 3)
    #expect(drafter.draftBlockCallCount == 1)
    #expect(main.lastIncomingCaptureLayers == [0, 1])
}

@Test
func testDSparkPartialAcceptanceEmitsMainSequenceOrder() throws {
    // drafter [5,5,5]; verify [5,5,7,9] → accept 2 + correction 7.
    let main = MockDSparkTarget(nextLogitTokens: [0, 0, 5, 5, 5, 7, 9])
    let drafter = MockDSparkDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try DSparkTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8), blockSize: 4)

    let toks = [iter.next(), iter.next(), iter.next(), iter.next()]
    #expect(toks == [5, 5, 5, 7])
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 2)
}

@Test
func testDSparkContextGrowsAndTrimsOnRejection() throws {
    // Round 1: drafter [5,5,5], verify [5,7,..] → accept 1, reject 2.
    // Prefill emits span 3 (prompt). Round-1 context length seen = 3.
    // Round-1 verify emits span 4 ([bonus,d1,d2,d3]); trimmed by rejected(2)→2.
    // Round-2 context = 3 (round-1 emission, trimmed) appended → length 5.
    let main = MockDSparkTarget(nextLogitTokens: [0, 0, 5, 5, 7, 0, 0, 0, 0, 0, 0])
    let drafter = MockDSparkDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try DSparkTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 12), blockSize: 4)

    _ = iter.next()  // prefill bonus
    _ = iter.next()  // round-1 accepted draft
    _ = iter.next()  // round-1 correction
    #expect(iter.acceptedCount == 1)
    _ = iter.next()  // starts round 2 (the draftBlock under test)

    #expect(drafter.draftBlockCallCount == 2)
    // Round 1 sees the prefill emission (span 3).
    #expect(drafter.receivedContextLengths.first == 3)
    // Round 2 sees prefill(3) + round-1 emission trimmed to accepted+bonus.
    // Round-1 verify span 4, rejected 2 → trimmed to 2; context = 3 + 2 = 5.
    #expect(drafter.receivedContextLengths.last == 5)
}

@Test
func testDSparkMissingLayerHiddensFallsBackToPassthrough() throws {
    let main = MockDSparkTarget(nextLogitTokens: [0, 0, 5, 11, 12])
    main.omitLayerHiddens = true
    let drafter = MockDSparkDrafter()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try DSparkTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 3), blockSize: 4)

    let toks = [iter.next(), iter.next(), iter.next(), iter.next()]
    #expect(toks == [5, 11, 12, nil])
    #expect(drafter.draftBlockCallCount == 0)
}

// MARK: - trimLayerHiddenState contract

@Test
func testTrimLayerHiddenStateTrimsTrailingPositions() {
    func arange(_ shape: [Int]) -> MLXArray {
        MLXArray(Array(Int32(0) ..< Int32(shape.reduce(1, *))), shape)
    }
    var state: LMOutput.State? = LMOutput.State()
    state?[mtpLayerHiddenStatesKey] = [0: arange([1, 6, 4]), 1: arange([1, 6, 4])]
    trimLayerHiddenState(&state, numTokens: 2)
    let lh = state?[mtpLayerHiddenStatesKey]
    #expect(lh?[0]?.dim(1) == 4)
    #expect(lh?[1]?.dim(1) == 4)

    var nilState: LMOutput.State? = nil
    trimLayerHiddenState(&nilState, numTokens: 3)
    #expect(nilState == nil)
}
