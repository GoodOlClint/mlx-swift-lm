// Copyright © 2026 Apple Inc.

import Foundation

/// Sequential Temperature Scaling for DSpark confidence-scheduled verify
/// (DSpark paper §3.2). The confidence head emits a raw per-position acceptance
/// **logit** `z_k`; STS applies a per-position temperature `T_k` so the calibrated
/// probability `σ(z_k / T_k)` matches the empirical acceptance rate. Because the
/// cumulative survival `∏ σ(z_i / T_i)` is what the verify scheduler thresholds,
/// calibration makes a threshold value mean the same thing across positions and
/// models. `T_k` rescales monotonically, so it never reorders tokens — pruning
/// stays lossless (ADR 0009).
///
/// The released checkpoints do **not** ship temperatures, so they are fit offline
/// on held-out (logit, accepted) pairs collected from a real drafter+target run.
public enum DSparkSTS {

    /// Fit one temperature per draft position via temperature scaling: `T_k`
    /// minimizes the binary cross-entropy of `σ(z / T)` against the accepted
    /// labels at that position. Positions with no data or a single class (no
    /// calibration signal) default to `1` (identity).
    ///
    /// - Parameters:
    ///   - logitsPerPosition: `[position][sample]` confidence logits.
    ///   - acceptedPerPosition: `[position][sample]` accepted labels.
    /// - Returns: `T_k` per position (same length as the inputs).
    public static func fitTemperatures(
        logitsPerPosition: [[Float]], acceptedPerPosition: [[Bool]]
    ) -> [Float] {
        precondition(
            logitsPerPosition.count == acceptedPerPosition.count,
            "STS fit: position count mismatch")
        return zip(logitsPerPosition, acceptedPerPosition).map { logits, labels in
            fitOne(logits: logits, labels: labels)
        }
    }

    /// Temperature scaling for a single position. BCE(T) is unimodal in `T` on
    /// `(0, ∞)`, so a ternary search converges to the minimizer.
    static func fitOne(logits: [Float], labels: [Bool]) -> Float {
        guard logits.count == labels.count, !logits.isEmpty else { return 1 }
        let positives = labels.lazy.filter { $0 }.count
        // No calibration signal without both classes present.
        if positives == 0 || positives == labels.count { return 1 }

        func bce(_ t: Double) -> Double {
            var sum = 0.0
            for (z, y) in zip(logits, labels) {
                let p = 1.0 / (1.0 + exp(-Double(z) / t))
                let pc = Swift.min(Swift.max(p, 1e-7), 1 - 1e-7)
                sum += y ? -log(pc) : -log(1 - pc)
            }
            return sum / Double(logits.count)
        }

        // Wide bound: late draft positions are strongly over-confident, so their
        // fitted T can be large (a large T → σ ≈ 0.5, i.e. "this position's score
        // is uninformative"). Keep the ceiling well above realistic values.
        var lo = 0.05
        var hi = 200.0
        for _ in 0 ..< 100 {
            let m1 = lo + (hi - lo) / 3
            let m2 = hi - (hi - lo) / 3
            if bce(m1) < bce(m2) { hi = m2 } else { lo = m1 }
        }
        return Float((lo + hi) / 2)
    }

    /// Load a per-position temperature vector from a JSON `[Float]` file, or nil
    /// if absent/unreadable (⇒ the iterator falls back to `T = 1`).
    public static func load(from url: URL) -> [Float]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode([Float].self, from: data)
    }

    /// Write a per-position temperature vector as JSON `[Float]`.
    public static func save(_ temperatures: [Float], to url: URL) throws {
        try JSONEncoder().encode(temperatures).write(to: url)
    }
}
