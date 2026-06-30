// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// DSpark drafter for Gemma 4 targets (DeepSeek DeepSpec). Second DSpark target
// after Qwen3 — same semi-autoregressive shape (DFlash-style parallel backbone
// conditioned on the target's multi-layer hidden states, Eq. 2/3, + a rank-r
// Markov head, Eq. 5), but with the Gemma 4 transformer block.
//
// Port of DeepSpec `deepspec/modeling/dspark/gemma4/modeling.py`
// (architecture "Gemma4DSparkModel"). Inference path only — the confidence
// head / STS / hardware scheduler are deferred (fixed-length verify is
// lossless; paper §3.2).
//
// Reuses the SHARED seam unchanged: conforms to `DSparkDrafting`, plugs into
// `DSparkTokenIterator`. Deltas from Qwen3DSpark (per the Gemma4 reference):
//   - context fusion / norms use Gemma 4 RMSNorm (zero-centered `(1+weight)`)
//   - attention `head_dim = global_head_dim`, `scale = 1.0` (not d^-0.5)
//   - `attention_k_eq_v`: values reuse the k_proj path; a no-scale `v_norm`
//   - MLP activation gelu_pytorch_tanh; per-layer `layer_scalar`
//   - partial/proportional RoPE (reuses `ProportionalRoPE` via `initializeRope`)
//   - `final_logit_softcapping` applied after `lm_head`
// The fusion (`fc` + `hidden_norm`), Markov head, and KV-injection pattern are
// otherwise identical to Qwen3DSpark. Pinned by fixture-parity tests.

// MARK: - Configuration

public struct Gemma4DSparkConfiguration: Sendable {
    public var hiddenSize: Int
    public var vocabSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numGlobalKeyValueHeads: Int
    public var headDim: Int
    public var globalHeadDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var attentionKEqV: Bool
    public var finalLogitSoftcapping: Float?
    public var ropeParameters: [String: [String: StringOrNumber]]
    public var maxPositionEmbeddings: Int
    public var targetLayerIds: [Int]
    public var blockSize: Int
    public var markovRank: Int
    public var maskTokenId: Int

    /// The drafter's attention head dim. The reference uses
    /// `head_dim = global_head_dim` unconditionally (gemma4/modeling.py:40).
    public var attentionHeadDim: Int { globalHeadDim }

    /// K/V head count for the drafter's attention. With `attention_k_eq_v` the
    /// reference switches to `num_global_key_value_heads` (gemma4/modeling.py:43).
    public var attentionKVHeads: Int {
        attentionKEqV ? numGlobalKeyValueHeads : numKeyValueHeads
    }

    public init(
        hiddenSize: Int, vocabSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, numGlobalKeyValueHeads: Int, headDim: Int, globalHeadDim: Int,
        intermediateSize: Int, rmsNormEps: Float, attentionKEqV: Bool,
        finalLogitSoftcapping: Float?, ropeParameters: [String: [String: StringOrNumber]],
        maxPositionEmbeddings: Int, targetLayerIds: [Int], blockSize: Int, markovRank: Int,
        maskTokenId: Int
    ) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.attentionKEqV = attentionKEqV
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.ropeParameters = ropeParameters
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.targetLayerIds = targetLayerIds
        self.blockSize = blockSize
        self.markovRank = markovRank
        self.maskTokenId = maskTokenId
    }
}

extension Gemma4DSparkConfiguration: Decodable {
    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case targetLayerIds = "target_layer_ids"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case maskTokenId = "mask_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let headDim = try c.decode(Int.self, forKey: .headDim)
        let numKVHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        self.init(
            hiddenSize: try c.decode(Int.self, forKey: .hiddenSize),
            vocabSize: try c.decode(Int.self, forKey: .vocabSize),
            numHiddenLayers: try c.decode(Int.self, forKey: .numHiddenLayers),
            numAttentionHeads: try c.decode(Int.self, forKey: .numAttentionHeads),
            numKeyValueHeads: numKVHeads,
            // `attention_k_eq_v` configs default the global KV count to 1 (a single
            // shared K/V head); fall back to the standard count otherwise.
            numGlobalKeyValueHeads: try c.decodeIfPresent(
                Int.self, forKey: .numGlobalKeyValueHeads) ?? numKVHeads,
            headDim: headDim,
            // ponytail: the reference's attention head dim is global_head_dim; the
            // handoff lists the model's head_dim (256). Fall back to head_dim if
            // the checkpoint omits global_head_dim. config.json wins when present.
            globalHeadDim: try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? headDim,
            intermediateSize: try c.decode(Int.self, forKey: .intermediateSize),
            rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            attentionKEqV: try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false,
            finalLogitSoftcapping: try c.decodeIfPresent(
                Float.self, forKey: .finalLogitSoftcapping),
            ropeParameters: try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:],
            maxPositionEmbeddings: try c.decodeIfPresent(
                Int.self, forKey: .maxPositionEmbeddings) ?? 131_072,
            targetLayerIds: try c.decode([Int].self, forKey: .targetLayerIds),
            blockSize: try c.decode(Int.self, forKey: .blockSize),
            markovRank: try c.decode(Int.self, forKey: .markovRank),
            maskTokenId: try c.decode(Int.self, forKey: .maskTokenId)
        )
    }
}

// MARK: - Attention with target-context KV injection (Eq. 3)

final class Gemma4DSparkAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear?
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale

    @ModuleInfo var rope: OffsetLayer

    init(_ c: Gemma4DSparkConfiguration) {
        self.nHeads = c.numAttentionHeads
        self.nKVHeads = c.attentionKVHeads
        self.headDim = c.attentionHeadDim
        self.useKEqV = c.attentionKEqV
        self.scale = 1.0  // reference: self.scaling = 1.0 (not d^-0.5)

        _wq.wrappedValue = Linear(c.hiddenSize, nHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, nKVHeads * headDim, bias: false)
        if !useKEqV {
            _wv.wrappedValue = Linear(c.hiddenSize, nKVHeads * headDim, bias: false)
        }
        _wo.wrappedValue = Linear(nHeads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: headDim, eps: c.rmsNormEps)
        _vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: c.rmsNormEps)

        // Partial/proportional RoPE over the full_attention rope_parameters (all
        // drafter layers are full_attention). Reuses the exact helper the Gemma 4
        // target attention uses.
        let ropeConfig = c.ropeParameters["full_attention"]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? 1_000_000
        _rope.wrappedValue = initializeRope(
            dims: headDim, base: ropeTheta, traditional: false,
            scalingConfig: ropeConfig, maxPositionEmbeddings: c.maxPositionEmbeddings)
    }

    /// Draft tokens (`x`, length `q`) attend to the injected target context
    /// (`hCtx`, length `C`) **and** themselves. K/V are the projected context
    /// concatenated with the projected draft along the sequence axis. With
    /// `attention_k_eq_v`, values reuse the k_proj output (then a no-scale
    /// v_norm); keys get the zero-shift k_norm. RoPE: context K at [0, C), draft
    /// Q/K at [C, C+q).
    func callAsFunction(
        _ x: MLXArray, hCtx: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, q) = (x.dim(0), x.dim(1))
        let C = hCtx.dim(1)

        var queries = qNorm(wq(x).reshaped(B, q, nHeads, headDim)).transposed(0, 2, 1, 3)

        let kCat = concatenated([wk(hCtx), wk(x)], axis: 1)  // [B, C+q, nKV*d]
        var keys = kNorm(kCat.reshaped(B, C + q, nKVHeads, headDim)).transposed(0, 2, 1, 3)

        // attention_k_eq_v: values reuse the k_proj path; otherwise the v_proj.
        let vCat = useKEqV ? kCat : concatenated([wv!(hCtx), wv!(x)], axis: 1)
        let values = vNorm(vCat.reshaped(B, C + q, nKVHeads, headDim)).transposed(0, 2, 1, 3)

        queries = rope(queries, offset: C)
        keys = rope(keys, offset: 0)

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, q, -1)
        return wo(out)
    }
}

// MARK: - MLP (gelu_pytorch_tanh)

/// The drafter MLP — mirrors `Gemma4TextMLP`'s non-double-wide path (the drafter
/// has no KV-shared layers). Kept local to avoid threading a full
/// `Gemma4TextConfiguration` just to build a gate/up/down GLU.
final class Gemma4DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ c: Gemma4DSparkConfiguration) {
        _gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer (Gemma 4 norm sandwich + layer_scalar)

final class Gemma4DSparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Gemma4DSparkAttention
    @ModuleInfo var mlp: Gemma4DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ c: Gemma4DSparkConfiguration) {
        _attention.wrappedValue = Gemma4DSparkAttention(c)
        _mlp.wrappedValue = Gemma4DSparkMLP(c)
        _inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(
        _ x: MLXArray, hCtx: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = x + postAttentionLayerNorm(attention(inputLayerNorm(x), hCtx: hCtx, mask: mask))
        h = h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        return h * layerScalar
    }
}

// MARK: - Markov head (Eq. 5)

final class Gemma4DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var w1: Embedding  // [V, r] lookup
    @ModuleInfo(key: "markov_w2") var w2: Linear  // r -> V

    init(_ c: Gemma4DSparkConfiguration) {
        _w1.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.markovRank)
        _w2.wrappedValue = Linear(c.markovRank, c.vocabSize, bias: false)
    }

    /// First-order transition bias B(prev, ·) = w2(w1[prev]) for tokens `[B]`.
    func bias(_ prev: MLXArray) -> MLXArray { w2(w1(prev)) }
}

// MARK: - DSpark drafter backbone + heads

public final class Gemma4DSparkModel: Module {
    public let config: Gemma4DSparkConfiguration
    let embedScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear  // [H, m*H], W_c
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "layers") var layers: [Gemma4DSparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "markov_head") var markovHead: Gemma4DSparkMarkovHead

    public init(_ c: Gemma4DSparkConfiguration) {
        self.config = c
        self.embedScale = pow(Float(c.hiddenSize), 0.5)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _fc.wrappedValue = Linear(c.targetLayerIds.count * c.hiddenSize, c.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in Gemma4DSparkDecoderLayer(c) }
        _norm.wrappedValue = Gemma4RMSNormZeroShift(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
        _markovHead.wrappedValue = Gemma4DSparkMarkovHead(c)
    }

    /// Context fusion (Eq. 2): `H_ctx = hidden_norm(fc(concat[target hiddens]))`.
    /// `targetHidden` is the m captured layer hiddens concatenated on the
    /// feature axis: `[B, C, m*H]`.
    public func contextFusion(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    /// One backbone pass over a draft block. `noiseEmbedding` is `[B, block, H]`
    /// (scaled embedding of anchor + mask tokens); `targetHidden` is `[B, C, m*H]`.
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

    /// `lm_head` followed by the `final_logit_softcapping` tanh squash (Gemma 4
    /// caps logits; Qwen3 does not). Matches the reference `compute_logits`.
    public func logits(_ h: MLXArray) -> MLXArray {
        let raw = lmHead(h)
        guard let cap = config.finalLogitSoftcapping else { return raw }
        return tanh(raw / cap) * cap
    }

    /// Drop checkpoint weights for components this v1 doesn't model — the
    /// confidence head (deferred; fixed-length verify is lossless). Keeps
    /// `update(verify: [.all])` from failing on unmodeled keys.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.hasPrefix("confidence_head") }
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
}

// MARK: - DSparkDrafting conformance

extension Gemma4DSparkModel: DSparkDrafting {
    public var targetLayerIds: [Int] { config.targetLayerIds }
    public var blockSize: Int { config.blockSize }

    /// Produce `numDraft` draft tokens conditioned on the `bonus` anchor `[B]`
    /// and the accumulated multi-layer target context `[B, C, m*H]`. Builds the
    /// scaled noise embedding (anchor + mask tokens), runs the backbone with an
    /// all-visible single-block-at-frontier mask, applies lm_head (+softcap), and
    /// samples the block greedily via the Markov head. Returns `[B, numDraft]`.
    public func draftBlock(bonus: MLXArray, context: MLXArray, numDraft: Int) -> MLXArray {
        let B = context.dim(0)
        let C = context.dim(1)
        let bonus2 = bonus.ndim == 1 ? bonus.reshaped(bonus.dim(0), 1) : bonus
        // Noise ids: [bonus, mask, mask, ...] of length numDraft.
        let maskIds = MLXArray.full(
            [B, numDraft - 1], values: MLXArray(Int32(config.maskTokenId)))
        let noiseIds = numDraft > 1 ? concatenated([bonus2, maskIds], axis: 1) : bonus2
        let noiseEmbRaw = embedTokens(noiseIds)
        // Gemma 4 scales the word embeddings by sqrt(hidden_size).
        let noiseEmb = noiseEmbRaw * MLXArray(embedScale, dtype: noiseEmbRaw.dtype)
        // Single block at the generation frontier: every draft position attends
        // to all context + all draft (additive all-zeros mask).
        let mask = MLXArray.zeros([B, 1, numDraft, C + numDraft], dtype: noiseEmb.dtype)
        let hidden = backbone(noiseEmbedding: noiseEmb, targetHidden: context, mask: .array(mask))
        let base = logits(hidden)
        return markovSampleGreedy(base: base, anchor: bonus2.reshaped(B))
    }
}

// MARK: - Loading + target pairing

public enum Gemma4DSparkDrafter {
    /// Seeded target-model-id → DSpark drafter-id map, from DeepSpec's released
    /// checkpoints. Like the Gemma 4 assistant, pairing is explicit (the target
    /// config does not advertise its drafter).
    // ponytail: only the released 12B drafter id is known; add the exact MLX
    // Gemma4-12B target id here once confirmed against the checkpoint pair.
    public static let drafterForTarget: [String: String] = [
        "mlx-community/gemma-4-12b": "deepseek-ai/dspark_gemma4_12b_block7"
    ]

    /// Load a DSpark drafter from a checkpoint directory (`config.json` +
    /// `*.safetensors`). `sanitize` drops the deferred confidence head so
    /// `verify: [.all]` doesn't fail on the unmodeled keys.
    public static func load(directory: URL) throws -> Gemma4DSparkModel {
        let configURL = directory.appending(component: "config.json")
        let cfg = try JSONDecoder().decode(
            Gemma4DSparkConfiguration.self, from: Data(contentsOf: configURL))
        let model = Gemma4DSparkModel(cfg)

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
