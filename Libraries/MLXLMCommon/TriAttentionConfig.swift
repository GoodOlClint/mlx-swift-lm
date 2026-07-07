// Copyright © 2025 Apple Inc.

import Foundation

// TriAttention — trigonometric KV-cache token EVICTION (arXiv:2604.04921;
// reference WeianMao/triattention, MLX port by @DeadByDawn101, MIT).
//
// This ships the *norm-only* mode (the reference's `disable_trig=True`
// path — the only one actually functional upstream: its calibration
// script is a non-functional stub). Norm-only scoring is `‖k‖` and uses
// NEITHER token positions NOR cross-layer aggregation, so each
// attention-layer cache can self-evict independently. The calibrated
// trigonometric scorer (pre-RoPE Q-center stats + cross-layer global
// score + an offline calibration step) is a deliberate follow-up.

/// Configuration for TriAttention norm-only KV eviction. Defaults mirror
/// the reference `TriAttentionMLXConfig`.
public struct TriAttentionConfig: Sendable, Equatable {
    /// Max KV pairs to retain per attention layer.
    public var kvBudget: Int
    /// Run an eviction pass every N decode steps.
    public var divideLength: Int
    /// Aggregate per-head scores by mean (default) or max.
    public var scoreAggregation: ScoreAggregation
    /// Never evict prefill (prompt) tokens.
    public var prefillPin: Bool

    public enum ScoreAggregation: String, Sendable, Equatable {
        case mean
        case max
    }

    public init(
        kvBudget: Int = 2048,
        divideLength: Int = 128,
        scoreAggregation: ScoreAggregation = .mean,
        prefillPin: Bool = true
    ) {
        self.kvBudget = kvBudget
        self.divideLength = divideLength
        self.scoreAggregation = scoreAggregation
        self.prefillPin = prefillPin
    }
}
