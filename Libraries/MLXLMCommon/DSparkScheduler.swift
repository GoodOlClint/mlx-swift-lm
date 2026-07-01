// Copyright © 2026 Apple Inc.
//
// RESEARCH / FORK-ONLY — do NOT upstream (ADR 0010). Hardware-aware prefix
// scheduler for DSpark confidence-scheduled verify (paper §3.2.2, Alg 1). This
// is the pure decision component: given per-request prefix-survival probabilities
// and a profiled batched-forward cost curve, choose how many draft tokens to
// verify for each request to maximize aggregate goodput. It has no MLX or serving
// dependencies — the measurement harness (S2/S3) feeds it a real SPS curve and
// real confidence traces.

import Foundation

/// Hardware-aware prefix scheduler (DSpark Alg 1), offline/exact form.
///
/// Goodput model per batched decode step:
///   accepted(ℓ) = R + Σ_r Σ_{j<ℓ_r} survivals[r][j]     // R bonus tokens + expected surviving drafts
///   tokens(ℓ)   = R + Σ_r ℓ_r                            // total tokens in the batched forward
///   goodput(ℓ)  = accepted(ℓ) · stepThroughput(tokens(ℓ))
/// where `stepThroughput(B)` is steps/sec at a batched forward of `B` tokens
/// (from the profiled SPS curve; non-increasing in `B`).
///
/// Greedy admission (Alg 1): repeatedly extend the request whose next draft
/// position has the highest survival, while doing so increases goodput. Because
/// survivals are non-increasing within a request and `stepThroughput` is
/// non-increasing in `B`, the marginal value is monotone, so stopping at the
/// first non-improving admission is correct. Uses only pre-token survival
/// probabilities (the non-anticipating property).
public enum DSparkScheduler {

    /// Per-request draft verify lengths `ℓ_r ∈ 0...survivals[r].count`.
    ///
    /// - Parameters:
    ///   - survivals: `survivals[r][j]` = P(request r's prefix survives through
    ///     draft position j) = ∏_{i≤j} c_{r,i}. Non-increasing in j, in (0, 1].
    ///   - stepThroughput: relative steps/sec at a batched forward of `B` total
    ///     tokens (`B ≥ 1`). Must be positive and non-increasing.
    /// - Returns: `ℓ_r` per request. Every request always verifies its bonus
    ///   token (counted in the token budget), so `ℓ_r == 0` still makes progress.
    public static func selectVerifyLengths(
        survivals: [[Double]], stepThroughput: (Int) -> Double
    ) -> [Int] {
        let r = survivals.count
        guard r > 0 else { return [] }

        var lengths = [Int](repeating: 0, count: r)
        var tokens = r  // R bonus tokens always verified
        var accepted = Double(r)  // bonuses always "accepted"
        var goodput = accepted * stepThroughput(tokens)

        while true {
            // Highest-survival admittable next position across requests.
            var bestReq = -1
            var bestVal = -1.0
            for i in 0 ..< r where lengths[i] < survivals[i].count {
                let v = survivals[i][lengths[i]]
                if v > bestVal {
                    bestVal = v
                    bestReq = i
                }
            }
            if bestReq < 0 { break }  // all requests fully extended

            let newAccepted = accepted + bestVal
            let newGoodput = newAccepted * stepThroughput(tokens + 1)
            if newGoodput > goodput {
                lengths[bestReq] += 1
                tokens += 1
                accepted = newAccepted
                goodput = newGoodput
            } else {
                break  // monotone marginal value ⇒ no later admission can improve
            }
        }
        return lengths
    }

    /// Convenience: cumulative prefix-survival products from per-position
    /// acceptance probabilities `c_{r,j} ∈ (0,1]`.
    public static func survivals(fromConfidence confidence: [[Double]]) -> [[Double]] {
        confidence.map { row in
            var acc = 1.0
            return row.map { c in
                acc *= c
                return acc
            }
        }
    }
}
