// Copyright © 2026 Apple Inc.
//
// RESEARCH / FORK-ONLY (ADR 0010). Unit tests for the pure hardware-aware prefix
// scheduler (DSpark Alg 1) against synthetic SPS curves.

import Foundation
import Testing

@testable import MLXLMCommon

@Test
func testDSparkSchedulerPartialAdmissionPrioritizesHighSurvival() {
    // Two requests, γ=2. High-survival r0 vs low-survival r1; sublinear-but-real
    // cost sps(t)=1/(1+0.3t). Hand-worked: admit both of r0 (goodput rises
    // 1.25→1.526→1.682), then r1's first draft fails (4.2/2.5=1.680 < 1.682) →
    // stop. Expected [2, 0]: capacity goes to the request most likely to survive.
    let survivals = [[0.9, 0.8], [0.5, 0.4]]
    let lengths = DSparkScheduler.selectVerifyLengths(survivals: survivals) { b in
        1.0 / (1.0 + 0.3 * Double(b))
    }
    #expect(lengths == [2, 0])
}

@Test
func testDSparkSchedulerCheapStepAdmitsEverything() {
    // Flat throughput (a batched forward costs the same regardless of width): any
    // positive-survival draft raises goodput, so verify the whole block.
    let survivals = [[0.9, 0.7, 0.5], [0.6, 0.3, 0.1]]
    let lengths = DSparkScheduler.selectVerifyLengths(survivals: survivals) { _ in 1.0 }
    #expect(lengths == [3, 3])
}

@Test
func testDSparkSchedulerLinearCostAdmitsNothing() {
    // sps(t)=1/t: cost grows exactly linearly in tokens, so verifying k tokens
    // costs k separate forwards — speculation can never win. Admit no drafts
    // (each request still emits its bonus).
    let survivals = [[0.99, 0.98], [0.95, 0.9]]
    let lengths = DSparkScheduler.selectVerifyLengths(survivals: survivals) { b in
        1.0 / Double(b)
    }
    #expect(lengths == [0, 0])
}

@Test
func testDSparkSchedulerYieldsContiguousMonotonePrefixes() {
    // Whatever the mix, each ℓ_r is a valid prefix length in 0...γ (no skipping),
    // and a strictly higher-survival request never gets fewer tokens than a
    // uniformly-lower one under the same cost curve.
    let survivals = [[0.95, 0.9, 0.85, 0.8], [0.4, 0.3, 0.2, 0.1]]
    let lengths = DSparkScheduler.selectVerifyLengths(survivals: survivals) { b in
        1.0 / (1.0 + 0.1 * Double(b))
    }
    #expect(lengths.count == 2)
    for (i, l) in lengths.enumerated() { #expect(l >= 0 && l <= survivals[i].count) }
    #expect(lengths[0] >= lengths[1], "higher-survival request should get ≥ capacity")
}

@Test
func testDSparkSchedulerSurvivalsFromConfidence() {
    let s = DSparkScheduler.survivals(fromConfidence: [[0.9, 0.9, 0.5], [1.0, 0.5]])
    #expect(abs(s[0][0] - 0.9) < 1e-9)
    #expect(abs(s[0][1] - 0.81) < 1e-9)
    #expect(abs(s[0][2] - 0.405) < 1e-9)
    #expect(abs(s[1][1] - 0.5) < 1e-9)
}

@Test
func testDSparkSchedulerEmpty() {
    #expect(DSparkScheduler.selectVerifyLengths(survivals: []) { _ in 1.0 } == [])
}
