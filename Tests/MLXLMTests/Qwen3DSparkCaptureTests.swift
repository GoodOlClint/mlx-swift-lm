// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// MARK: - DSpark multi-layer capture-seam tests for Qwen3 (target)
//
// D1 of the DSpark drafter work (see docs/decisions/0006-…): the Qwen3 target
// gains an MTP-aware `callAsFunction(_:cache:state:)` that, when a drafter sets
// `mtpEmitFlagKey` + `mtpCaptureLayersKey`, captures post-block hidden states at
// the requested layer indices and emits them via `mtpLayerHiddenStatesKey`
// (DSpark paper Eq. 2/3 multi-layer context injection). With the flag/keys
// absent the path must be bit-identical to ordinary generation, so the existing
// drafters and plain Qwen3 inference are unaffected.
//
// Construction uses a tiny synthetic `Qwen3Configuration` (random weights) — we
// assert key presence, shapes, and the bit-identical invariant, not numerics.

@Test
func testQwen3EmitAbsentReturnsNoState() {
    let model = makeSyntheticQwen3Model(layers: 4)
    let input = LMInput.Text(tokens: tokens(8))

    let out = model(input, cache: nil, state: nil)

    #expect(out.state == nil)
    eval(out.logits)
    #expect(out.logits.shape == [1, 8, 10])
}

@Test
func testQwen3CaptureLayersPopulatesLayerHiddens() {
    let model = makeSyntheticQwen3Model(layers: 4)
    let input = LMInput.Text(tokens: tokens(6))

    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    state[mtpCaptureLayersKey] = [0, 2]
    let out = model(input, cache: nil, state: state)

    #expect(out.state != nil, "emit + captureLayers should populate LMOutput.state")
    guard let s = out.state,
        let captured = s[mtpLayerHiddenStatesKey],
        let lastHidden = s[mtpLastHiddenStatesKey]
    else {
        Issue.record("expected captured layer hiddens + last hidden")
        return
    }
    // Only the requested layers are captured.
    #expect(Set(captured.keys) == [0, 2])
    for (_, h) in captured {
        eval(h)
        #expect(h.shape == [1, 6, 4])
    }
    eval(lastHidden)
    #expect(lastHidden.shape == [1, 6, 4])
}

@Test
func testQwen3EmitWithEmptyCaptureLayersTakesDefaultPath() {
    let model = makeSyntheticQwen3Model(layers: 4)
    let input = LMInput.Text(tokens: tokens(4))

    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    state[mtpCaptureLayersKey] = []  // nothing to capture ⇒ no state emitted
    let out = model(input, cache: nil, state: state)

    #expect(out.state == nil)
    eval(out.logits)
}

@Test
func testQwen3EmitDisabledIsBitIdenticalRegression() {
    let model = makeSyntheticQwen3Model(layers: 4)
    let toks = tokens(6)

    // Bare 2-arg path (ordinary generation) vs the new state overload with no
    // emit. The emit-off branch MUST route through the identical forward.
    let bare = model(toks, cache: nil)
    let viaState = model(LMInput.Text(tokens: toks), cache: nil, state: nil).logits

    eval(bare, viaState)
    #expect(
        allClose(bare, viaState, rtol: 0, atol: 0).item(Bool.self),
        "emit-off state overload must be bit-identical to bare callAsFunction"
    )
}

// MARK: - Helpers

private func tokens(_ n: Int) -> MLXArray {
    MLXArray((0 ..< n).map { Int32($0) }).reshaped([1, n])
}

/// Tiny random-weight Qwen3 model — sufficient for capture-seam plumbing and the
/// bit-identical regression; not a numeric reference for the model itself.
private func makeSyntheticQwen3Model(layers: Int) -> Qwen3Model {
    let json =
        """
        {
          "hidden_size": 4,
          "num_hidden_layers": \(layers),
          "intermediate_size": 8,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 10,
          "tie_word_embeddings": true,
          "max_position_embeddings": 16
        }
        """
    let cfg = try! JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
    return Qwen3Model(cfg)
}
