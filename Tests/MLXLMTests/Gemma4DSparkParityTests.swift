// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// MARK: - DSpark Gemma 4 drafter parity vs the DeepSpec reference
//
// Sibling of Qwen3DSparkParityTests. Loads a tiny seeded fixture recorded from
// DeepSpec's `Gemma4DSparkModel`
// (research/dspark-fixtures/record_fixtures_gemma4.py) and asserts the MLX port
// reproduces each stage: context fusion (Eq. 2), the KV-injection Gemma 4
// backbone (Eq. 3) — exercising k_eq_v, partial/proportional RoPE, the
// gelu_pytorch_tanh MLP, Gemma 4 RMSNorm, and layer_scalar — lm_head with the
// final_logit_softcapping, and Markov greedy sampling (Eq. 5). Tiny model, no
// checkpoint — runs in CI.

private enum Gemma4TinyMeta {
    static let H = 16, V = 32, L = 2, m = 2, block = 3, ctx = 4
    static let anchor = 5, headDim = 8, nH = 2, nKV = 1, rank = 4
    static let softcap: Float = 30.0, intermediate = 32
}

private func gemma4TinyConfig() -> Gemma4DSparkConfiguration {
    Gemma4DSparkConfiguration(
        hiddenSize: Gemma4TinyMeta.H, vocabSize: Gemma4TinyMeta.V,
        numHiddenLayers: Gemma4TinyMeta.L, numAttentionHeads: Gemma4TinyMeta.nH,
        numKeyValueHeads: Gemma4TinyMeta.nKV, numGlobalKeyValueHeads: Gemma4TinyMeta.nKV,
        headDim: Gemma4TinyMeta.headDim, globalHeadDim: Gemma4TinyMeta.headDim,
        intermediateSize: Gemma4TinyMeta.intermediate, rmsNormEps: 1e-6, attentionKEqV: true,
        finalLogitSoftcapping: Gemma4TinyMeta.softcap,
        ropeParameters: [
            "full_attention": [
                "rope_type": .string("proportional"),
                "rope_theta": .float(1_000_000),
                "partial_rotary_factor": .float(0.25),
            ]
        ],
        maxPositionEmbeddings: 64, targetLayerIds: Array(0 ..< Gemma4TinyMeta.m),
        blockSize: Gemma4TinyMeta.block, markovRank: Gemma4TinyMeta.rank,
        maskTokenId: Gemma4TinyMeta.V - 1,
        enableConfidenceHead: true, confidenceHeadWithMarkov: true)
}

@Test
func testGemma4DSparkParityVsDeepSpecReference() throws {
    guard
        let url = Bundle.module.url(
            forResource: "dspark_gemma4_tiny", withExtension: "safetensors")
    else {
        Issue.record("dspark_gemma4_tiny.safetensors fixture not found in test bundle")
        return
    }
    let all = try loadArrays(url: url)
    let model = Gemma4DSparkModel(gemma4TinyConfig())

    // Load reference weights (strip "w." prefix; confidence head is deferred).
    var params: [String: MLXArray] = [:]
    for (k, v) in all where k.hasPrefix("w.") {
        let key = String(k.dropFirst(2))
        if key.hasPrefix("rotary_emb") { continue }
        params[key] = v
    }
    try model.update(parameters: ModuleParameters.unflattened(params), verify: [])
    eval(model)

    let targetHidden = all["in.target_hidden"]!
    let noiseEmb = all["in.noise_embedding"]!
    let mask: MLXFast.ScaledDotProductAttentionMaskMode = .array(all["in.attn_mask"]!)

    // Cross-framework parity tolerances. Context fusion (fc + one RMSNorm) is
    // tight. The backbone/logits stages are looser: Gemma 4's block has
    // unnormalized attention (scale = 1.0) and 4 RMSNorms/layer, so torch-CPU-f32
    // ↔ MLX-Metal-f32 kernel differences (rsqrt, gelu-tanh, SDPA softmax)
    // accumulate ~0.25% across the 2-layer backbone — more than Qwen3's
    // normalized block. The reference is f64-stable to ~1e-6, so this is kernel
    // drift, not a structural gap (rope, attention, and fusion each verified to
    // <4e-4 in isolation). The tolerance-free argmax anchor below is what pins
    // end-to-end correctness despite the drift.
    let fuseTol: Double = 2e-3
    let driftTol: Double = 2e-2

    // (1) context fusion — Eq. 2 (Gemma 4 RMSNorm)
    let hctx = model.contextFusion(targetHidden)
    eval(hctx)
    let hctxDiff = (hctx - all["out.h_ctx"]!).abs().max().item(Float.self)
    #expect(
        allClose(hctx, all["out.h_ctx"]!, rtol: fuseTol, atol: fuseTol).item(Bool.self),
        "context fusion mismatch, max-diff \(hctxDiff)")

    // (2) backbone — Eq. 3 KV injection (k_eq_v, partial RoPE) + layers + norm
    let back = model.backbone(noiseEmbedding: noiseEmb, targetHidden: targetHidden, mask: mask)
    eval(back)
    let backDiff = (back - all["out.backbone_out"]!).abs().max().item(Float.self)
    #expect(
        allClose(back, all["out.backbone_out"]!, rtol: driftTol, atol: driftTol).item(Bool.self),
        "backbone mismatch, max-diff \(backDiff)")

    // (3) lm_head + final_logit_softcapping
    let base = model.logits(back)
    eval(base)
    let baseDiff = (base - all["out.base_logits"]!).abs().max().item(Float.self)
    #expect(
        allClose(base, all["out.base_logits"]!, rtol: driftTol, atol: driftTol).item(Bool.self),
        "lm_head/softcap mismatch, max-diff \(baseDiff)")

    // (4) Markov greedy sampling — Eq. 5. Feed the REFERENCE base logits so this
    // isolates the Markov recurrence from any backbone float drift.
    let toks = model.markovSampleGreedy(
        base: all["out.base_logits"]!, anchor: MLXArray([Int32(Gemma4TinyMeta.anchor)]))
    eval(toks)
    #expect(
        allClose(toks.asType(.int32), all["out.markov_tokens"]!.asType(.int32), rtol: 0, atol: 0)
            .item(Bool.self),
        "Markov greedy token sequence mismatch")

    // (5) End-to-end correctness anchor (tolerance-free): the full MLX pipeline —
    // backbone → lm_head(+softcap) → Markov — must produce the SAME draft tokens
    // as the reference. argmax is robust to the sub-1% logit drift above, so this
    // pins that the drift is benign and the architecture is faithful end to end.
    let e2eToks = model.markovSampleGreedy(
        base: base, anchor: MLXArray([Int32(Gemma4TinyMeta.anchor)]))
    eval(e2eToks)
    #expect(
        allClose(e2eToks.asType(.int32), all["out.markov_tokens"]!.asType(.int32), rtol: 0, atol: 0)
            .item(Bool.self),
        "end-to-end draft tokens diverged from reference, diff \((e2eToks.asType(.int32) - all["out.markov_tokens"]!.asType(.int32)).abs().max().item(Int32.self))"
    )

    // (6) Confidence head logits — Eq. 7. Feed reference backbone hiddens + tokens
    // so this isolates the head (proj over [hidden ; w1[prev]]).
    let refMarkov = all["out.markov_tokens"]!.asType(.int32)
    var confParts: [MLXArray] = []
    var cprev = MLXArray([Int32(Gemma4TinyMeta.anchor)])
    for k in 0 ..< Gemma4TinyMeta.block {
        let hiddenK = all["out.backbone_out"]![0..., k, 0...]
        let logit = model.confidenceLogit(hidden: hiddenK, prev: cprev)!
        confParts.append(logit.reshaped(1, 1))
        cprev = refMarkov[0..., k]
    }
    let conf = concatenated(confParts, axis: 1)
    eval(conf)
    let confDiff = (conf - all["out.confidence"]!).abs().max().item(Float.self)
    #expect(
        allClose(conf, all["out.confidence"]!, rtol: fuseTol, atol: fuseTol).item(Bool.self),
        "confidence head mismatch, max-diff \(confDiff)")
}

// MARK: - Config decoding

@Test
func testGemma4DSparkConfigDecodesRealShape() throws {
    // Representative of dspark_gemma4_12b_block7/config.json, incl. the nested
    // per-layer-type rope_parameters and the Gemma 4 deltas.
    let json = """
        {
          "architectures": ["Gemma4DSparkModel"],
          "block_size": 7, "head_dim": 256, "global_head_dim": 512,
          "hidden_size": 3840, "intermediate_size": 15360, "markov_rank": 256,
          "mask_token_id": 4, "max_position_embeddings": 131072, "model_type": "gemma4_text",
          "num_attention_heads": 16, "num_hidden_layers": 5,
          "num_key_value_heads": 8, "num_global_key_value_heads": 1,
          "attention_k_eq_v": true, "final_logit_softcapping": 30.0, "rms_norm_eps": 1e-06,
          "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_type": "proportional", "rope_theta": 1000000},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
            "rope_theta": null, "rope_type": "default"
          },
          "target_layer_ids": [5, 17, 29, 41, 46], "vocab_size": 262144
        }
        """
    let cfg = try JSONDecoder().decode(Gemma4DSparkConfiguration.self, from: Data(json.utf8))
    #expect(cfg.hiddenSize == 3840)
    #expect(cfg.vocabSize == 262_144)
    #expect(cfg.numHiddenLayers == 5)
    #expect(cfg.numAttentionHeads == 16)
    #expect(cfg.numKeyValueHeads == 8)
    #expect(cfg.numGlobalKeyValueHeads == 1)
    #expect(cfg.headDim == 256)
    #expect(cfg.globalHeadDim == 512)
    #expect(cfg.attentionHeadDim == 512)  // reference: head_dim = global_head_dim
    #expect(cfg.attentionKVHeads == 1)  // k_eq_v ⇒ num_global_key_value_heads
    #expect(cfg.attentionKEqV == true)
    #expect(cfg.finalLogitSoftcapping == 30.0)
    #expect(cfg.targetLayerIds == [5, 17, 29, 41, 46])
    #expect(cfg.blockSize == 7)
    #expect(cfg.markovRank == 256)
    #expect(cfg.maskTokenId == 4)
    #expect(cfg.ropeParameters["full_attention"]?["rope_theta"]?.asFloat() == 1_000_000)
}

// MARK: - Loader + pairing

@Test
func testGemma4DSparkLoaderRoundTripsAndSanitizes() throws {
    let cfg = gemma4TinyConfig()
    let model = Gemma4DSparkModel(cfg)
    eval(model)

    // The model's own params form a valid checkpoint; add an unmodeled
    // confidence_head key that the loader's sanitize MUST drop (else
    // `verify: [.all]` would throw on the unexpected key → load fails).
    var weights = [String: MLXArray]()
    for (k, v) in model.parameters().flattened() { weights[k] = v }
    weights["confidence_head.proj.weight"] = MLXArray.zeros([1, cfg.hiddenSize + cfg.markovRank])
    weights["confidence_head.proj.bias"] = MLXArray.zeros([1])

    let dir = FileManager.default.temporaryDirectory
        .appending(component: "dspark_gemma4_load_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try save(arrays: weights, url: dir.appending(component: "model.safetensors"))
    let configJSON = """
        {"hidden_size":16,"vocab_size":32,"num_hidden_layers":2,
         "num_attention_heads":2,"num_key_value_heads":1,"num_global_key_value_heads":1,
         "head_dim":8,"global_head_dim":8,"intermediate_size":32,"rms_norm_eps":1e-6,
         "attention_k_eq_v":true,"final_logit_softcapping":30.0,"block_size":3,
         "markov_rank":4,"mask_token_id":31,"max_position_embeddings":64,
         "rope_parameters":{"full_attention":{"rope_type":"proportional","rope_theta":1000000,"partial_rotary_factor":0.25}},
         "target_layer_ids":[0,1]}
        """
    try Data(configJSON.utf8).write(to: dir.appending(component: "config.json"))

    let loaded = try Gemma4DSparkDrafter.load(directory: dir)
    eval(loaded)
    // Round-trip fidelity: a representative weight survived load unchanged.
    #expect(allClose(loaded.fc.weight, model.fc.weight, rtol: 0, atol: 0).item(Bool.self))
    // Seeded pairing.
    #expect(
        Gemma4DSparkDrafter.drafterForTarget["mlx-community/gemma-4-12B-it-bf16"]
            == "deepseek-ai/dspark_gemma4_12b_block7")
}
