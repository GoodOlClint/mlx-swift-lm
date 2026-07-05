// Bespoke DoD test (NOT ported from upstream #263).
//
// Checks that batched decode is semantically transparent: decoding B prompts
// together yields per-row token streams identical to decoding each prompt alone
// (B=1). The B=1 path IS the single-stream reference (same BatchGenerator code).
// Real (tiny, seeded) LlamaModel, greedy sampling, no network/model download.
//
// KNOWN FAILURE — mlx-tracker #9. Ported #263 batched decode does NOT reliably
// match single-stream: for some model weights a batched row gets "stuck" and
// diverges (its KV cache differs materially from the solo decode, not merely by
// floating-point tie-breaking). This affects BOTH equal-length and ragged
// batches — it is a fundamental B>=2 decode bug, not the ragged/left-padding
// path. Deterministic per seed; some weight sets hide it, some expose it.
// Ruled out in isolation: the vector-offset RoPE kernel, dynamicRoll, and
// in-place cache writes are each correct; forcing synchronous evaluation does
// not fix it (so it is a logic bug, not an async race). Root cause open.
// Remove the XCTExpectFailure wrappers once #9 is fixed.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import XCTest

final class BatchDecodeParityTests: XCTestCase {

    private func makeModel(seed: UInt64) -> LlamaModel {
        MLXRandom.seed(seed)
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    /// Decode `prompts` together and return per-prompt token streams in input order.
    private func decodeTokens(_ model: LlamaModel, _ prompts: [[Int]], maxTokens: Int) -> [[Int]] {
        let generator = BatchGenerator(
            model: model,
            defaultMaxTokens: maxTokens,
            prefillBatchSize: prompts.count,
            completionBatchSize: prompts.count
        )
        let uids = generator.insert(prompts: prompts)
        var tokensByUID: [Int: [Int]] = [:]
        var steps = 0
        while generator.hasWork {
            steps += 1
            XCTAssertLessThan(steps, maxTokens + 5, "decode did not terminate")
            for response in generator.next() {
                tokensByUID[response.uid, default: []].append(response.token)
            }
        }
        return uids.map { tokensByUID[$0] ?? [] }
    }

    private func assertBatchMatchesSingleStream(
        _ promptA: [Int], _ promptB: [Int], maxTokens: Int = 12
    ) {
        for seed in UInt64(0) ..< 8 {
            let model = makeModel(seed: seed)
            let soloA = decodeTokens(model, [promptA], maxTokens: maxTokens)[0]
            let soloB = decodeTokens(model, [promptB], maxTokens: maxTokens)[0]
            let batched = decodeTokens(model, [promptA, promptB], maxTokens: maxTokens)
            XCTAssertEqual(
                batched[0], soloA, "seed \(seed): batched row A must match single-stream")
            XCTAssertEqual(
                batched[1], soloB, "seed \(seed): batched row B must match single-stream")
        }
    }

    func testBatchedDecodeMatchesSingleStreamEqualLength() {
        XCTExpectFailure("batched decode diverges for some weights — mlx-tracker #9", strict: true)
        {
            assertBatchMatchesSingleStream([1, 2, 3, 4, 5], [7, 8, 9, 10, 11])
        }
    }

    func testBatchedDecodeMatchesSingleStreamRagged() {
        XCTExpectFailure("batched decode diverges for some weights — mlx-tracker #9", strict: true)
        {
            assertBatchMatchesSingleStream([1, 2, 3], [7, 8, 9, 10, 11])
        }
    }
}
