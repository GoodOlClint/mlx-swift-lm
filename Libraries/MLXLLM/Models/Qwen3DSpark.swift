// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// DSpark drafter for Qwen3 targets (DeepSeek DeepSpec). Semi-autoregressive
// speculative decoding: a DFlash-style parallel backbone conditioned on the
// target's multi-layer hidden states (Eq. 2/3) plus a rank-r Markov head that
// makes the block distribution autoregressive (Eq. 5).
//
// Port of DeepSpec `deepspec/modeling/dspark/qwen3/modeling.py`
// (architecture "Qwen3DSparkModel"). Inference path only — the confidence
// head / STS / hardware scheduler are deferred (fixed-length verify is
// lossless; paper §3.2). See research/dspark-fixtures/DSPARK-SWIFT-BLUEPRINT.md.
//
// This file is the D2 backbone + heads, pinned by fixture-parity tests against
// the DeepSpec reference. Wiring into the speculative accept loop is D3.

// MARK: - Configuration

public struct Qwen3DSparkConfiguration: Sendable {
    public var hiddenSize: Int
    public var vocabSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var targetLayerIds: [Int]
    public var blockSize: Int
    public var markovRank: Int
    public var maskTokenId: Int
    public var maxPositionEmbeddings: Int
    public var enableConfidenceHead: Bool
    public var confidenceHeadWithMarkov: Bool

    /// Input width of the confidence head's `proj`: the backbone hidden, plus the
    /// Markov prev-token embedding when `confidence_head_with_markov`.
    public var confidenceInputDim: Int {
        hiddenSize + (confidenceHeadWithMarkov ? markovRank : 0)
    }

    public init(
        hiddenSize: Int, vocabSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, headDim: Int, intermediateSize: Int, rmsNormEps: Float,
        ropeTheta: Float, targetLayerIds: [Int], blockSize: Int, markovRank: Int,
        maskTokenId: Int, maxPositionEmbeddings: Int,
        enableConfidenceHead: Bool = false, confidenceHeadWithMarkov: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.targetLayerIds = targetLayerIds
        self.blockSize = blockSize
        self.markovRank = markovRank
        self.maskTokenId = maskTokenId
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.enableConfidenceHead = enableConfidenceHead
        self.confidenceHeadWithMarkov = confidenceHeadWithMarkov
    }
}

extension Qwen3DSparkConfiguration: Decodable {
    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case targetLayerIds = "target_layer_ids"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case maskTokenId = "mask_token_id"
        case maxPositionEmbeddings = "max_position_embeddings"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        private enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // `rope_theta` may be top-level (older configs) or nested under
        // `rope_parameters` (Qwen3DSpark checkpoints).
        let ropeTheta =
            (try? c.decode(Float.self, forKey: .ropeTheta))
            ?? ((try? c.decode(RopeParameters.self, forKey: .ropeParameters))?.ropeTheta)
            ?? 1_000_000
        self.init(
            hiddenSize: try c.decode(Int.self, forKey: .hiddenSize),
            vocabSize: try c.decode(Int.self, forKey: .vocabSize),
            numHiddenLayers: try c.decode(Int.self, forKey: .numHiddenLayers),
            numAttentionHeads: try c.decode(Int.self, forKey: .numAttentionHeads),
            numKeyValueHeads: try c.decode(Int.self, forKey: .numKeyValueHeads),
            headDim: try c.decode(Int.self, forKey: .headDim),
            intermediateSize: try c.decode(Int.self, forKey: .intermediateSize),
            rmsNormEps: try c.decode(Float.self, forKey: .rmsNormEps),
            ropeTheta: ropeTheta,
            targetLayerIds: try c.decode([Int].self, forKey: .targetLayerIds),
            blockSize: try c.decode(Int.self, forKey: .blockSize),
            markovRank: try c.decode(Int.self, forKey: .markovRank),
            maskTokenId: try c.decode(Int.self, forKey: .maskTokenId),
            maxPositionEmbeddings: try c.decodeIfPresent(
                Int.self, forKey: .maxPositionEmbeddings) ?? 40960,
            enableConfidenceHead: try c.decodeIfPresent(
                Bool.self, forKey: .enableConfidenceHead) ?? false,
            confidenceHeadWithMarkov: try c.decodeIfPresent(
                Bool.self, forKey: .confidenceHeadWithMarkov) ?? false
        )
    }
}

// MARK: - Attention with target-context KV injection (Eq. 3)

final class Qwen3DSparkAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ c: Qwen3DSparkConfiguration) {
        self.nHeads = c.numAttentionHeads
        self.nKVHeads = c.numKeyValueHeads
        self.headDim = c.headDim
        self.scale = pow(Float(c.headDim), -0.5)

        _wq.wrappedValue = Linear(c.hiddenSize, nHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, nKVHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, nKVHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(nHeads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        self.rope = initializeRope(
            dims: headDim, base: c.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: c.maxPositionEmbeddings)
    }

    /// Draft tokens (`x`, length `q`) attend to the injected target context
    /// (`hCtx`, length `C`) **and** themselves. Keys/values are the projected
    /// context concatenated with the projected draft, along the sequence axis.
    /// RoPE: context K at positions [0, C), draft Q/K at [C, C+q).
    func callAsFunction(
        _ x: MLXArray, hCtx: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, q) = (x.dim(0), x.dim(1))
        let C = hCtx.dim(1)

        var queries = qNorm(wq(x).reshaped(B, q, nHeads, headDim)).transposed(0, 2, 1, 3)

        let kCat = concatenated([wk(hCtx), wk(x)], axis: 1)  // [B, C+q, nKV*d]
        let vCat = concatenated([wv(hCtx), wv(x)], axis: 1)
        var keys = kNorm(kCat.reshaped(B, C + q, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vCat.reshaped(B, C + q, nKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(C))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(0))

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, q, -1)
        return wo(out)
    }
}

// MARK: - Decoder layer

final class Qwen3DSparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3DSparkAttention
    let mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: Qwen3DSparkConfiguration) {
        _attention.wrappedValue = Qwen3DSparkAttention(c)
        self.mlp = Qwen3MLP(dimensions: c.hiddenSize, hiddenDimensions: c.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, hCtx: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = x + attention(inputLayerNorm(x), hCtx: hCtx, mask: mask)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

// MARK: - Markov head (Eq. 5)

final class Qwen3DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var w1: Embedding  // [V, r] lookup
    @ModuleInfo(key: "markov_w2") var w2: Linear  // r -> V

    init(_ c: Qwen3DSparkConfiguration) {
        _w1.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.markovRank)
        _w2.wrappedValue = Linear(c.markovRank, c.vocabSize, bias: false)
    }

    /// First-order transition bias B(prev, ·) = w2(w1[prev]) for tokens `[B]`.
    func bias(_ prev: MLXArray) -> MLXArray { w2(w1(prev)) }
}

// MARK: - Confidence head (Eq. 7 — per-position acceptance estimator)

/// DSpark `AcceptRatePredictor`: a single `Linear(inputDim → 1)` over the
/// per-position feature `[hidden ; markov_w1[prev]]` (or just `hidden` when not
/// `with_markov`). Returns the raw acceptance **logit**; the sigmoid + STS
/// temperature are applied by the verify scheduler (C2/C3), matching the
/// reference `AcceptRatePredictor.forward` (`proj(features).squeeze(-1)`).
/// Checkpoint keys: `confidence_head.proj.{weight,bias}`.
final class Qwen3DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(inputDim: Int) {
        _proj.wrappedValue = Linear(inputDim, 1)
    }

    /// `features` `[B, inputDim]` → acceptance logit `[B]`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        proj(features).squeezed(axis: -1)
    }
}

// MARK: - DSpark drafter backbone + heads

public final class Qwen3DSparkModel: Module {
    public let config: Qwen3DSparkConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear  // [H, m*H], W_c
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Qwen3DSparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "markov_head") var markovHead: Qwen3DSparkMarkovHead
    @ModuleInfo(key: "confidence_head") var confidenceHead: Qwen3DSparkConfidenceHead?

    public init(_ c: Qwen3DSparkConfiguration) {
        self.config = c
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _fc.wrappedValue = Linear(c.targetLayerIds.count * c.hiddenSize, c.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in Qwen3DSparkDecoderLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
        _markovHead.wrappedValue = Qwen3DSparkMarkovHead(c)
        if c.enableConfidenceHead {
            _confidenceHead.wrappedValue = Qwen3DSparkConfidenceHead(inputDim: c.confidenceInputDim)
        }
    }

    /// Per-position acceptance **logit** for the confidence head (Eq. 7): feature
    /// is `[hidden ; markov_w1[prev]]` (or `hidden` alone when not with-markov).
    /// `hidden` `[B, H]`, `prev` `[B]` → logit `[B]`. `nil` if no confidence head.
    public func confidenceLogit(hidden: MLXArray, prev: MLXArray) -> MLXArray? {
        guard let confidenceHead else { return nil }
        let features =
            config.confidenceHeadWithMarkov
            ? concatenated([hidden, markovHead.w1(prev)], axis: -1)
            : hidden
        return confidenceHead(features)
    }

    /// Context fusion (Eq. 2): `H_ctx = hidden_norm(fc(concat[target hiddens]))`.
    /// `targetHidden` is the m captured layer hiddens concatenated on the
    /// feature axis: `[B, C, m*H]`.
    public func contextFusion(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    /// One backbone pass over a draft block. `noiseEmbedding` is `[B, block, H]`
    /// (embedding of anchor + mask tokens); `targetHidden` is `[B, C, m*H]`.
    public func backbone(
        noiseEmbedding: MLXArray, targetHidden: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let hCtx = contextFusion(targetHidden)
        var h = noiseEmbedding
        for layer in layers {
            h = layer(h, hCtx: hCtx, mask: mask)
        }
        return norm(h)
    }

    public func logits(_ h: MLXArray) -> MLXArray { lmHead(h) }

    /// Keep the confidence-head weights when the head is modeled
    /// (`enable_confidence_head`); otherwise drop them so `update(verify: [.all])`
    /// doesn't fail on the unmodeled keys.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if confidenceHead != nil { return weights }
        return weights.filter { !$0.key.hasPrefix("confidence_head") }
    }

    /// Sequential greedy Markov sampling (temp 0) over a block of `base` logits
    /// `[B, block, V]`, seeded by the `anchor` token `[B]`. Returns `[B, block]`.
    public func markovSampleGreedy(base: MLXArray, anchor: MLXArray) -> MLXArray {
        var prev = anchor
        var toks: [MLXArray] = []
        for k in 0 ..< base.dim(1) {
            let step = base[0..., k, 0...] + markovHead.bias(prev)
            let tok = argMax(step, axis: -1)
            toks.append(tok.reshaped(tok.dim(0), 1))
            prev = tok
        }
        return concatenated(toks, axis: 1)
    }

    /// Like ``markovSampleGreedy`` but also returns per-position confidence logits
    /// (Eq. 7) when the confidence head is present. `hidden` `[B, block, H]` is the
    /// backbone output; confidence at position k uses `hidden[:, k]` and the prev
    /// token (anchor for k=0). Returns `(tokens [B, block], confidence [B, block]?)`.
    public func markovSampleGreedyWithConfidence(
        base: MLXArray, hidden: MLXArray, anchor: MLXArray
    ) -> (tokens: MLXArray, confidence: MLXArray?) {
        let hasConfidence = confidenceHead != nil
        var prev = anchor
        var toks: [MLXArray] = []
        var confs: [MLXArray] = []
        for k in 0 ..< base.dim(1) {
            if hasConfidence, let cl = confidenceLogit(hidden: hidden[0..., k, 0...], prev: prev) {
                confs.append(cl.reshaped(cl.dim(0), 1))
            }
            let step = base[0..., k, 0...] + markovHead.bias(prev)
            let tok = argMax(step, axis: -1)
            toks.append(tok.reshaped(tok.dim(0), 1))
            prev = tok
        }
        return (concatenated(toks, axis: 1), hasConfidence ? concatenated(confs, axis: 1) : nil)
    }
}

// MARK: - DSparkDrafting conformance

extension Qwen3DSparkModel: DSparkDrafting {
    public var targetLayerIds: [Int] { config.targetLayerIds }
    public var blockSize: Int { config.blockSize }

    /// Produce `numDraft` draft tokens conditioned on the `bonus` anchor `[B]`
    /// and the accumulated multi-layer target context `[B, C, m*H]`. Builds the
    /// noise embedding (anchor + mask tokens), runs the backbone with an
    /// all-visible single-block-at-frontier mask, applies lm_head, and samples
    /// the block greedily via the Markov head. Returns `[B, numDraft]`.
    public func draftBlock(bonus: MLXArray, context: MLXArray, numDraft: Int) -> DSparkDraft {
        let B = context.dim(0)
        let C = context.dim(1)
        let bonus2 = bonus.ndim == 1 ? bonus.reshaped(bonus.dim(0), 1) : bonus
        // Noise ids: [bonus, mask, mask, ...] of length numDraft.
        let maskIds = MLXArray.full(
            [B, numDraft - 1], values: MLXArray(Int32(config.maskTokenId)))
        let noiseIds = numDraft > 1 ? concatenated([bonus2, maskIds], axis: 1) : bonus2
        let noiseEmb = embedTokens(noiseIds)
        // Single block at the generation frontier: every draft position attends
        // to all context + all draft (additive all-zeros mask).
        let mask = MLXArray.zeros([B, 1, numDraft, C + numDraft], dtype: noiseEmb.dtype)
        let hidden = backbone(noiseEmbedding: noiseEmb, targetHidden: context, mask: .array(mask))
        let base = logits(hidden)
        let (tokens, confidence) = markovSampleGreedyWithConfidence(
            base: base, hidden: hidden, anchor: bonus2.reshaped(B))
        return DSparkDraft(tokens: tokens, confidence: confidence)
    }
}

// MARK: - Loading + target pairing

public enum Qwen3DSparkDrafter {
    /// Seeded target-model-id → DSpark drafter-id map, from DeepSpec's released
    /// checkpoints. The target config doesn't advertise its drafter, so — like
    /// the Gemma 4 assistant — pairing is explicit (not sniffed in-config).
    public static let drafterForTarget: [String: String] = [
        "Qwen/Qwen3-4B": "deepseek-ai/dspark_qwen3_4b_block7",
        "Qwen/Qwen3-8B": "deepseek-ai/dspark_qwen3_8b_block7",
        "Qwen/Qwen3-14B": "deepseek-ai/dspark_qwen3_14b_block7",
    ]

    /// Load a DSpark drafter from a checkpoint directory (`config.json` +
    /// `*.safetensors`). `sanitize` drops the deferred confidence head so
    /// `verify: [.all]` doesn't fail on the unmodeled keys.
    public static func load(directory: URL) throws -> Qwen3DSparkModel {
        let configURL = directory.appending(component: "config.json")
        let cfg = try JSONDecoder().decode(
            Qwen3DSparkConfiguration.self, from: Data(contentsOf: configURL))
        let model = Qwen3DSparkModel(cfg)

        var weights: [String: MLXArray] = [:]
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            for (k, v) in try loadArrays(url: url) { weights[k] = v }
        }
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
