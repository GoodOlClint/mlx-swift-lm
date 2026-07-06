// Bespoke DoD test (NOT ported from upstream #263).
//
// Batched decode should be semantically transparent: decoding B prompts together
// yields per-row token streams that match decoding each prompt alone (B=1), up to
// the floating-point non-associativity of batched vs single-stream matmul (which
// can flip greedy's argmax only at near-ties). The B=1 path IS the single-stream
// reference (same BatchGenerator code). Real (tiny, seeded) LlamaModel, greedy,
// no network/model download. Swept across weight seeds.
//
// Regression pin for mlx-tracker #9: `ropeOffset` is a KVCache protocol
// requirement; its default witness (KVCache extension) is `.scalar(offset)`, and
// a subclass/refined-protocol extension could not override that once BaseKVCache
// bound it — so BatchKVCache silently fed every row the scalar `_idx` instead of
// its per-row `.batch(batchOffset)`, corrupting all rows after the first. Fixed by
// making `ropeOffset` an open class property on BaseKVCache and overriding it in
// the batched caches. Before the fix, most seeds diverged materially; after, they
// match single-stream.

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

    /// Asserts batched decode matches single-stream for all but at most
    /// `tieBudget` weight seeds. Exact token parity is NOT guaranteed even by the
    /// reference implementation: batched matmul accumulates in a different order
    /// than single-stream, so a near-tie can flip greedy's argmax (a 1-token
    /// shift). mlx-lm Python diverges for exactly 1 of these 8 seeds on the same
    /// prompts; we allow the same floor. Before the #9 fix, 6+/8 seeds diverged
    /// materially (a row got "stuck"), so this still discriminates the bug.
    private func assertBatchMatchesSingleStream(
        _ promptA: [Int], _ promptB: [Int], maxTokens: Int = 12, tieBudget: Int = 1
    ) {
        var diverged: [UInt64] = []
        for seed in UInt64(0) ..< 8 {
            let model = makeModel(seed: seed)
            let soloA = decodeTokens(model, [promptA], maxTokens: maxTokens)[0]
            let soloB = decodeTokens(model, [promptB], maxTokens: maxTokens)[0]
            let batched = decodeTokens(model, [promptA, promptB], maxTokens: maxTokens)
            if batched[0] != soloA || batched[1] != soloB { diverged.append(seed) }
        }
        XCTAssertLessThanOrEqual(
            diverged.count, tieBudget,
            "batched decode diverged from single-stream for seeds \(diverged) "
                + "(> \(tieBudget) allowed FP-tie divergences)")
    }

    func testBatchedDecodeMatchesSingleStreamEqualLength() {
        assertBatchMatchesSingleStream([1, 2, 3, 4, 5], [7, 8, 9, 10, 11])
    }

    func testBatchedDecodeMatchesSingleStreamRagged() {
        assertBatchMatchesSingleStream([1, 2, 3], [7, 8, 9, 10, 11])
    }
}
