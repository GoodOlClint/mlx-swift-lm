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

// MARK: - Config decoding

@Test
func testQwen3DSparkConfigDecodesRealShape() throws {
    // Representative of deepseek-ai/dspark_qwen3_4b_block7/config.json, incl. the
    // nested rope_parameters that the decoder must flatten to ropeTheta.
    let json = """
        {
          "architectures": ["Qwen3DSparkModel"],
          "block_size": 7, "head_dim": 128, "hidden_size": 2560,
          "intermediate_size": 9728, "markov_rank": 256, "mask_token_id": 151669,
          "max_position_embeddings": 40960, "model_type": "qwen3",
          "num_attention_heads": 32, "num_hidden_layers": 5,
          "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
          "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
          "target_layer_ids": [1, 9, 17, 25, 33], "vocab_size": 151936
        }
        """
    let cfg = try JSONDecoder().decode(Qwen3DSparkConfiguration.self, from: Data(json.utf8))
    #expect(cfg.hiddenSize == 2560)
    #expect(cfg.vocabSize == 151936)
    #expect(cfg.numHiddenLayers == 5)
    #expect(cfg.numKeyValueHeads == 8)
    #expect(cfg.headDim == 128)
    #expect(cfg.targetLayerIds == [1, 9, 17, 25, 33])
    #expect(cfg.blockSize == 7)
    #expect(cfg.markovRank == 256)
    #expect(cfg.maskTokenId == 151669)
    #expect(cfg.ropeTheta == 1_000_000)  // flattened from rope_parameters
}

// MARK: - Loader + pairing

@Test
func testQwen3DSparkLoaderRoundTripsAndSanitizes() throws {
    let cfg = Qwen3DSparkConfiguration(
        hiddenSize: 8, vocabSize: 16, numHiddenLayers: 1, numAttentionHeads: 2,
        numKeyValueHeads: 1, headDim: 4, intermediateSize: 16, rmsNormEps: 1e-6,
        ropeTheta: 1e6, targetLayerIds: [0, 1], blockSize: 3, markovRank: 4,
        maskTokenId: 15, maxPositionEmbeddings: 64)
    let model = Qwen3DSparkModel(cfg)
    eval(model)

    // The model's own params form a valid checkpoint; add an unmodeled
    // confidence_head key that the loader's sanitize MUST drop (else
    // `verify: [.all]` would throw on the unexpected key → load fails).
    var weights = [String: MLXArray]()
    for (k, v) in model.parameters().flattened() { weights[k] = v }
    weights["confidence_head.proj.weight"] = MLXArray.zeros([1, cfg.hiddenSize + cfg.markovRank])
    weights["confidence_head.proj.bias"] = MLXArray.zeros([1])

    let dir = FileManager.default.temporaryDirectory
        .appending(component: "dspark_load_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try save(arrays: weights, url: dir.appending(component: "model.safetensors"))
    let configJSON = """
        {"hidden_size":8,"vocab_size":16,"num_hidden_layers":1,
         "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
         "intermediate_size":16,"rms_norm_eps":1e-6,"block_size":3,
         "markov_rank":4,"mask_token_id":15,"max_position_embeddings":64,
         "rope_parameters":{"rope_theta":1000000},"target_layer_ids":[0,1]}
        """
    try Data(configJSON.utf8).write(to: dir.appending(component: "config.json"))

    let loaded = try Qwen3DSparkDrafter.load(directory: dir)
    eval(loaded)
    // Round-trip fidelity: a representative weight survived load unchanged.
    #expect(allClose(loaded.fc.weight, model.fc.weight, rtol: 0, atol: 0).item(Bool.self))
    // Seeded pairing.
    #expect(
        Qwen3DSparkDrafter.drafterForTarget["Qwen/Qwen3-4B"]
            == "deepseek-ai/dspark_qwen3_4b_block7")
}
