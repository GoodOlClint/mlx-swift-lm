// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A DSpark-family drafter: a semi-autoregressive draft model that proposes a
/// block of tokens conditioned on an *accumulated multi-layer target context*
/// (DSpark paper Eq. 2/3) rather than a single per-round hidden. The iterator
/// owns the growing context; the drafter is stateless with respect to it (the
/// whole context is threaded as a method argument), so drafter instances are
/// safe to share across streams.
///
/// Distinct from ``MTPDrafterModel`` (single `lastHidden` + shared K/V) — DSpark
/// injects target hidden states from multiple layers over the entire prefix as
/// K/V into every draft layer, which the MTP capture seam does not carry. See
/// `docs/decisions/0006`.
public protocol DSparkDrafting {
    /// Target decoder-layer indices whose hidden states form the injected
    /// context (DSpark `target_layer_ids`). The iterator requests capture at
    /// exactly these via ``mtpCaptureLayersKey``.
    var targetLayerIds: [Int] { get }

    /// Draft block size γ (anchor + γ−1 positions).
    var blockSize: Int { get }

    /// Produce `numDraft` draft tokens `[B, numDraft]` conditioned on the
    /// `bonus` anchor `[B]` and the accumulated context `[B, C, m*H]` (the m
    /// target-layer hiddens concatenated on the feature axis).
    func draftBlock(bonus: MLXArray, context: MLXArray, numDraft: Int) -> MLXArray
}

/// Speculative token iterator for DSpark drafters. Mirrors
/// ``MTPSpeculativeTokenIterator``'s prefill → draft → verify → accept → rewind
/// cycle and reuses the same greedy/longest-prefix acceptance (so the
/// bit-exact-equivalence-to-greedy guarantee holds), but threads an accumulated
/// multi-layer target context instead of single-hidden + shared K/V.
///
/// The context grows by each main-model emission's
/// ``mtpLayerHiddenStatesKey`` chunk and is trimmed in lockstep with the cache
/// on partial acceptance. Falls back to single-token passthrough if the target
/// stops emitting layer hiddens (e.g. KV quantization onset).
public struct DSparkTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    let mainModel: any LanguageModel
    let drafter: any DSparkDrafting
    let targetLayerIds: [Int]

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    public let blockSize: Int

    /// Accumulated injected context `[B, C, m*H]`, grown per round from emitted
    /// layer hiddens and trimmed on rejection. `nil` until the first emission.
    // ponytail: accumulates + re-projects the full prefix each round (O(seq)).
    // The perf upgrade is dflash-mlx's projected-K/V draft cache (append, no
    // re-projection) — deferred; correctness-equivalent.
    var context: MLXArray?

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var passthrough = false
    private var passthroughLoggedOnce = false

    private var telemetry = SpeculativeDecodingTelemetry()
    public private(set) var acceptedCount: Int = 0
    public private(set) var proposedCount: Int = 0
    public private(set) var passthroughReason: String?

    public var promptPrefillTime: TimeInterval = 0.0
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DSparkDrafting,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int? = nil
    ) throws {
        let bs = blockSize ?? drafter.blockSize
        precondition(bs >= 2, "DSparkTokenIterator requires blockSize >= 2 (1 bonus + K-1 drafted)")

        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter
        self.targetLayerIds = drafter.targetLayerIds

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache) else {
            throw KVCacheError(
                message: "DSpark speculative decoding requires a trimmable main KV cache.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.blockSize = bs

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache, kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart)
        }

        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prepare(input: input, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    /// State the iterator pushes into each main-model call to opt the target
    /// into emitting layer hiddens at `targetLayerIds`.
    private func emitState() -> LMOutput.State {
        var s = LMOutput.State()
        s[mtpEmitFlagKey] = true
        s[mtpCaptureLayersKey] = targetLayerIds
        return s
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: emitState())
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
            pendingTokens.append(token.item(Int.self))
        case .logits(let prefillResult):
            var logits = prefillResult.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = prefillResult.state
            if mainState?[mtpLayerHiddenStatesKey] == nil {
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: emitState())
                mainState = primed.state
                var newLogits = primed.logits[0..., -1, 0...]
                newLogits = processor?.process(logits: newLogits) ?? newLogits
                let newToken = sampler.sample(logits: newLogits)
                processor?.didSample(token: newToken)
                y = .init(tokens: newToken)
                pendingTokens.append(token.item(Int.self))
                pendingTokens.append(newToken.item(Int.self))
            } else {
                pendingTokens.append(token.item(Int.self))
            }
        }
    }

    /// Concatenate the m target-layer hiddens (in `targetLayerIds` order) on the
    /// feature axis → `[B, seq, m*H]`. Returns nil if any requested layer is
    /// absent (drives passthrough).
    private func concatLayers(_ d: [Int: MLXArray]) -> MLXArray? {
        var parts: [MLXArray] = []
        for id in targetLayerIds {
            guard let h = d[id] else { return nil }
            parts.append(h)
        }
        return concatenated(parts, axis: -1)
    }

    mutating func speculateRound() {
        guard !passthrough else { return }

        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }
            let draftBudget = Swift.min(remaining - 1, blockSize - 1)
            guard draftBudget > 0 else {
                if let token = passthroughStep() { pendingTokens.append(token) }
                return
            }
            numDraft = draftBudget
        } else {
            numDraft = blockSize - 1
        }

        guard
            let state = mainState,
            let layerHiddens = state[mtpLayerHiddenStatesKey],
            let chunk = concatLayers(layerHiddens)
        else {
            switchToPassthrough(reason: "main model did not emit layer hiddens")
            return
        }

        // Grow the accumulated context with this emission's chunk.
        context = context == nil ? chunk : concatenated([context!, chunk], axis: 1)
        let ctx = context!

        let bonusToken = y.tokens
        let draftTokens = drafter.draftBlock(bonus: bonusToken, context: ctx, numDraft: numDraft)
        let flatDraftTokens = draftTokens.flattened()

        // Verify pass: [bonus, d_1 ... d_numDraft] in one forward, emitting next
        // round's layer hiddens.
        let verifyTokens = concatenated([bonusToken, flatDraftTokens])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: mainCache, state: emitState())
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let mainTokens: MLXArray
        if var verifyProcessorCopy = processor {
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessorCopy.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessorCopy.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        eval(mainTokens, flatDraftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = flatDraftTokens.asArray(Int.self)

        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else { break }
            let drafted = flatDraftTokens[i ..< (i + 1)]
            processor?.didSample(token: drafted)
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        proposedCount += numDraft
        acceptedCount += accepted
        telemetry.recordRound(
            drafted: numDraft, accepted: accepted,
            targetVerified: numDraft + 1, draftModelCalls: 1)

        // Rewind the cache and this round's emission by the rejected count, in
        // lockstep, so next round's appended chunk excludes rejected positions.
        let rejected = numDraft - accepted
        let trimmed = trimPromptCache(mainCache, numTokens: rejected)
        trimLayerHiddenState(&mainState, numTokens: trimmed)
        quantizeKVCache(&mainCache)

        if let context { eval(context) }  // bound the lazy graph each round
        y = .init(tokens: finalToken)
    }

    private mutating func switchToPassthrough(reason: String) {
        if !passthroughLoggedOnce {
            print("[DSparkTokenIterator] passthrough mode: \(reason)")
            passthroughLoggedOnce = true
        }
        passthroughReason = reason
        passthrough = true
    }

    private mutating func passthroughStep() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        let result = mainModel(y[text: .newAxis], cache: mainCache, state: nil)
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        let tokenInt = token.item(Int.self)
        y = .init(tokens: token)
        quantizeKVCache(&mainCache)
        return tokenInt
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        if passthrough {
            if let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            if passthrough, let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

extension DSparkTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
}

/// Rewinds the emitted layer-hidden snapshot by `numTokens` trailing sequence
/// positions, mirroring ``trimSharedKVState`` for the DSpark capture key. No-op
/// when `numTokens <= 0`, the state is nil, or the key is absent.
func trimLayerHiddenState(_ state: inout LMOutput.State?, numTokens: Int) {
    guard numTokens > 0, let layerHiddens = state?[mtpLayerHiddenStatesKey] else { return }
    state?[mtpLayerHiddenStatesKey] = layerHiddens.mapValues { h in
        let newLen = h.dim(1) - numTokens
        return h[0..., ..<newLen, 0...]
    }
}
