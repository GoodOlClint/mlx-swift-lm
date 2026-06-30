// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// MARK: - DSpark Qwen3 drafter parity vs the DeepSpec reference (D2)
//
// Loads a tiny seeded fixture recorded from DeepSpec's `Qwen3DSparkModel`
// (research/dspark-fixtures/record_fixtures.py) and asserts the MLX port
// reproduces each stage: context fusion (Eq. 2), the KV-injection backbone
// (Eq. 3), lm_head, and Markov greedy sampling (Eq. 5). Tiny model, no
// checkpoint — runs in CI.

private struct TinyMeta {
    static let H = 16, V = 32, L = 2, m = 2, block = 3, ctx = 4
    static let anchor = 5, headDim = 8, nH = 2, nKV = 1, rank = 4
}

@Test
func testQwen3DSparkParityVsDeepSpecReference() throws {
    guard
        let url = Bundle.module.url(
            forResource: "dspark_qwen3_tiny", withExtension: "safetensors")
    else {
        Issue.record("dspark_qwen3_tiny.safetensors fixture not found in test bundle")
        return
    }
    let all = try loadArrays(url: url)

    let cfg = Qwen3DSparkConfiguration(
        hiddenSize: TinyMeta.H, vocabSize: TinyMeta.V, numHiddenLayers: TinyMeta.L,
        numAttentionHeads: TinyMeta.nH, numKeyValueHeads: TinyMeta.nKV, headDim: TinyMeta.headDim,
        intermediateSize: 32, rmsNormEps: 1e-6, ropeTheta: 1e6,
        targetLayerIds: Array(0 ..< TinyMeta.m), blockSize: TinyMeta.block,
        markovRank: TinyMeta.rank, maskTokenId: TinyMeta.V - 1, maxPositionEmbeddings: 64)
    let model = Qwen3DSparkModel(cfg)

    // Load reference weights (strip "w." prefix; confidence head is deferred).
    var params: [String: MLXArray] = [:]
    for (k, v) in all where k.hasPrefix("w.") {
        let key = String(k.dropFirst(2))
        if key.hasPrefix("confidence_head") || key.hasPrefix("rotary_emb") { continue }
        params[key] = v
    }
    try model.update(parameters: ModuleParameters.unflattened(params), verify: [])
    eval(model)

    let targetHidden = all["in.target_hidden"]!
    let noiseEmb = all["in.noise_embedding"]!
    let mask: MLXFast.ScaledDotProductAttentionMaskMode = .array(all["in.attn_mask"]!)

    // Cross-framework (torch f32 ↔ MLX f32 through rsqrt/matmul/SDPA) parity tol.
    let tol: Double = 2e-3

    // (1) context fusion — Eq. 2
    let hctx = model.contextFusion(targetHidden)
    eval(hctx)
    let hctxDiff = (hctx - all["out.h_ctx"]!).abs().max().item(Float.self)
    #expect(
        allClose(hctx, all["out.h_ctx"]!, rtol: tol, atol: tol).item(Bool.self),
        "context fusion mismatch, max-diff \(hctxDiff)")

    // (2) backbone — Eq. 3 KV injection + layers + final norm
    let back = model.backbone(noiseEmbedding: noiseEmb, targetHidden: targetHidden, mask: mask)
    eval(back)
    let backDiff = (back - all["out.backbone_out"]!).abs().max().item(Float.self)
    #expect(
        allClose(back, all["out.backbone_out"]!, rtol: tol, atol: tol).item(Bool.self),
        "backbone mismatch, max-diff \(backDiff)")

    // (3) lm_head
    let base = model.logits(back)
    eval(base)
    let baseDiff = (base - all["out.base_logits"]!).abs().max().item(Float.self)
    #expect(
        allClose(base, all["out.base_logits"]!, rtol: tol, atol: tol).item(Bool.self),
        "lm_head mismatch, max-diff \(baseDiff)")

    // (4) Markov greedy sampling — Eq. 5. Feed the REFERENCE base logits so this
    // isolates the Markov recurrence from any backbone float drift.
    let toks = model.markovSampleGreedy(
        base: all["out.base_logits"]!, anchor: MLXArray([Int32(TinyMeta.anchor)]))
    eval(toks)
    #expect(
        allClose(toks.asType(.int32), all["out.markov_tokens"]!.asType(.int32), rtol: 0, atol: 0)
            .item(Bool.self),
        "Markov greedy token sequence mismatch")
}
