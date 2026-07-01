// Copyright © 2026 Apple Inc.
//
// RESEARCH / FORK-ONLY (ADR 0010, S2+S3). Measures whether the DSpark
// hardware-aware prefix scheduler improves aggregate throughput on THIS hardware.
//
// S2: empirical SPS curve — decode-forward time vs tokens-per-forward on the real
//     target (the batched-forward cost the scheduler consumes). Measured at
//     batch 1 over token count L; total-tokens-per-forward is the first-order
//     cost variable (weight loading dominates on Apple Silicon, so the batch-vs-
//     length split is second-order — real B>1 execution would need the engine).
// S3: trace-driven comparison — real per-round confidence→survival traces (via
//     the C4 collector) + the SPS curve fed into DSparkScheduler, reporting
//     scheduled-vs-fixed-length aggregate goodput across simulated batch sizes.
//
// Cache-gated; prints the curve + the decisive table. No wall-clock floor.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
@_spi(Testing) import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

@Suite(.serialized)
struct Gemma4DSparkSchedulerMeasurementTests {

    @Test
    func measureSchedulerVsFixedLength12BBF16() async throws {
        guard let targetDir = hfSnapshotDir(modelId: "mlx-community/gemma-4-12B-it-bf16"),
            let drafterDir = hfSnapshotDir(modelId: "deepseek-ai/dspark_gemma4_12b_block7")
        else {
            Issue.record("bf16 12B target or DSpark drafter not in HF cache; skipping")
            return
        }
        let context = try await VLMModelFactory.shared.load(
            from: targetDir, using: #huggingFaceTokenizerLoader())
        let drafter = try Gemma4DSparkDrafter.load(directory: drafterDir)
        quantize(model: drafter, groupSize: 64, bits: 8, filter: { _, m in m is Linear })
        eval(drafter)
        let gamma = drafter.blockSize - 1  // draft positions per block

        // ---- S2: SPS curve — decode-forward time vs tokens-per-forward ----
        let primePrompt = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 8)
        let Ls = [1, 2, 4, 7, 12, 20, 32]
        var timeByTokens: [(t: Int, sec: Double)] = []
        print("\n=== S2 SPS curve (12B bf16, batch 1) ===")
        for L in Ls {
            // Prime a cache through the model's own prepare (correct per-layer /
            // mask setup), then time decode forwards on the exact LMInput.Text +
            // state path the DSpark iterator uses for verify.
            let primed = try await context.processor.prepare(input: UserInput(chat: [.user(primePrompt)]))
            let cache = context.model.newCache(parameters: GenerateParameters())
            _ = try context.model.prepare(primed, cache: cache, windowSize: nil)
            eval(cache.map { $0.state }.flatMap { $0 })
            let step = LMInput.Text(tokens: MLXArray(Array(repeating: Int32(1), count: L)))
            _ = context.model(step[text: .newAxis], cache: cache, state: nil).logits  // warm
            eval(cache.map { $0.state }.flatMap { $0 })
            var times: [Double] = []
            for _ in 0 ..< 20 {
                let t0 = Date()
                let out = context.model(step[text: .newAxis], cache: cache, state: nil)
                eval(out.logits)
                times.append(Date().timeIntervalSince(t0))
            }
            times.sort()
            let med = times[times.count / 2]
            timeByTokens.append((L, med))
            print(String(format: "  tokens=%2d  step=%.4f ms  tput=%.1f tok/s(rel)", L, med * 1000, Double(L) / med))
        }

        // stepThroughput(B tokens) = 1 / interpolated forward time. Non-increasing.
        func forwardTime(_ tokens: Int) -> Double {
            let t = max(1, tokens)
            if t <= timeByTokens.first!.t { return timeByTokens.first!.sec }
            if t >= timeByTokens.last!.t { return timeByTokens.last!.sec }
            for i in 1 ..< timeByTokens.count where t <= timeByTokens[i].t {
                let (t0, s0) = timeByTokens[i - 1]
                let (t1, s1) = timeByTokens[i]
                let f = Double(t - t0) / Double(t1 - t0)
                return s0 + f * (s1 - s0)
            }
            return timeByTokens.last!.sec
        }
        let stepThroughput: (Int) -> Double = { 1.0 / forwardTime($0) }

        // ---- fit STS temps (needed for realistic survival products) ----
        var logitsByPos = [[Float]](repeating: [], count: gamma)
        var acceptedByPos = [[Bool]](repeating: [], count: gamma)
        var survivalPool: [[Double]] = []
        for prompt in [
            "Why is the sky blue? Explain in one paragraph.",
            "Natalia sold 48 clips in April then half as many in May. Total? Step by step.",
            "Write a Swift function returning the nth Fibonacci number iteratively.",
        ] {
            let input = try await context.processor.prepare(input: UserInput(chat: [.user(prompt)]))
            var iter = try DSparkTokenIterator(
                input: input, mainModel: context.model, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 128, temperature: 0),
                confidenceThreshold: 0, collectConfidenceCalibration: true)
            var n = 0
            while n < 128, iter.next() != nil { n += 1 }
            for s in iter.confidenceCalibrationSamples where s.position < gamma {
                logitsByPos[s.position].append(s.logit)
                acceptedByPos[s.position].append(s.accepted)
            }
            // Segment the ordered samples into per-round logit vectors (position
            // resets to 0 each round) for correlated survival traces.
            var current: [Float] = []
            for s in iter.confidenceCalibrationSamples {
                if s.position == 0, !current.isEmpty {
                    survivalPool.append(current.map { Double($0) })
                    current = []
                }
                current.append(s.logit)
            }
            if !current.isEmpty { survivalPool.append(current.map { Double($0) }) }
        }
        let temps = DSparkSTS.fitTemperatures(
            logitsPerPosition: logitsByPos, acceptedPerPosition: acceptedByPos)

        // Convert per-round logit vectors → survival products with STS temps.
        func survivals(_ logits: [Double]) -> [Double] {
            var acc = 1.0
            var out: [Double] = []
            for (k, z) in logits.enumerated() where k < gamma {
                let t = k < temps.count ? Double(temps[k]) : 1.0
                acc *= 1.0 / (1.0 + exp(-z / t))
                out.append(acc)
            }
            return out
        }
        let pool = survivalPool.filter { $0.count == gamma }.map(survivals)
        guard pool.count >= 8 else {
            Issue.record("too few survival traces (\(pool.count))"); return
        }

        // ---- S3: scheduler vs fixed-length across batch sizes ----
        func goodput(lengths: [Int], batch: [[Double]]) -> Double {
            var accepted = Double(batch.count)  // bonus per request
            var tokens = batch.count
            for (r, l) in lengths.enumerated() {
                tokens += l
                for j in 0 ..< l { accepted += batch[r][j] }
            }
            return accepted * stepThroughput(tokens)
        }

        print("\n=== S3 scheduler vs fixed-length (trace-driven) ===")
        print(
            String(
                format: "%-7@ %14@ %14@ %10@", "batch" as NSString, "sched-verify" as NSString,
                "fixed-verify" as NSString, "goodput" as NSString))
        // Deterministic sampling from the pool (index stride) so the run is stable.
        for b in [1, 2, 4, 8, 16] {
            let trials = 40
            var ratioSum = 0.0
            var schedVerifySum = 0.0
            for tr in 0 ..< trials {
                var batch: [[Double]] = []
                for i in 0 ..< b { batch.append(pool[(tr * b + i) % pool.count]) }
                let sched = DSparkScheduler.selectVerifyLengths(
                    survivals: batch, stepThroughput: stepThroughput)
                let fixed = [Int](repeating: gamma, count: b)
                ratioSum += goodput(lengths: sched, batch: batch)
                    / goodput(lengths: fixed, batch: batch)
                schedVerifySum += Double(sched.reduce(0, +)) / Double(b)
            }
            print(
                String(
                    format: "%-7d %14.2f %14d %9.3fx", b, schedVerifySum / Double(trials), gamma,
                    ratioSum / Double(trials)))
        }
        print("============================================================\n")
        #expect(!pool.isEmpty)
    }
}
