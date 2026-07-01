// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
@_spi(Testing) import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - DSpark STS calibration harness (ADR 0009 C3)

/// Fits per-position Sequential Temperature Scaling temperatures for the Gemma4
/// DSpark drafter. The released checkpoint ships the confidence head but no STS
/// temperatures, so we collect `(confidence logit, accepted)` samples from a real
/// drafter+target run at `threshold = 0` (no truncation → every position's
/// acceptance is observed), then fit one temperature per draft position via
/// ``DSparkSTS/fitTemperatures``. Writes `sts_temperatures.json` and logs the
/// vector. Cache-gated; does not assert a numeric target.
@Suite(.serialized)
struct Gemma4DSparkSTSCalibrationTests {

    @Test
    func fitGemma4DSparkSTSTemperatures12BBF16() async throws {
        let targetId = "mlx-community/gemma-4-12B-it-bf16"
        let drafterId = "deepseek-ai/dspark_gemma4_12b_block7"
        guard let targetDir = hfSnapshotDir(modelId: targetId),
            let drafterDir = hfSnapshotDir(modelId: drafterId)
        else {
            Issue.record("bf16 12B target or DSpark drafter not in HF cache; skipping")
            return
        }
        let context = try await VLMModelFactory.shared.load(
            from: targetDir, using: #huggingFaceTokenizerLoader())
        let drafter = try Gemma4DSparkDrafter.load(directory: drafterDir)

        // Domain-mixed calibration prompts (non-thinking mode, matching training).
        let prompts = [
            "Why is the sky blue? Explain in one paragraph.",
            "Natalia sold clips to 48 friends in April, then half as many in May. "
                + "How many clips did she sell altogether? Think step by step.",
            "Write a Swift function returning the nth Fibonacci number iteratively.",
            "Summarize the causes of the French Revolution in a few sentences.",
        ]

        var samples: [(logit: Float, position: Int, accepted: Bool)] = []
        for prompt in prompts {
            let lmInput = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)]))
            var iter = try DSparkTokenIterator(
                input: lmInput,
                mainModel: context.model,
                drafter: drafter,
                parameters: GenerateParameters(maxTokens: 128, temperature: 0),
                confidenceThreshold: 0,  // observe acceptance at every position
                collectConfidenceCalibration: true)
            var produced = 0
            while produced < 128, iter.next() != nil { produced += 1 }
            samples.append(contentsOf: iter.confidenceCalibrationSamples)
            if let reason = iter.passthroughReason {
                Issue.record("drafter fell to passthrough (\(reason)); calibration may be thin")
            }
        }

        guard !samples.isEmpty else {
            Issue.record("no calibration samples collected"); return
        }

        // Group by draft position and fit one temperature each.
        let blockSize = drafter.blockSize
        var logitsByPos = [[Float]](repeating: [], count: blockSize - 1)
        var acceptedByPos = [[Bool]](repeating: [], count: blockSize - 1)
        for s in samples where s.position < blockSize - 1 {
            logitsByPos[s.position].append(s.logit)
            acceptedByPos[s.position].append(s.accepted)
        }
        let temps = DSparkSTS.fitTemperatures(
            logitsPerPosition: logitsByPos, acceptedPerPosition: acceptedByPos)

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("dspark_gemma4_12b_sts_temperatures.json")
        try DSparkSTS.save(temps, to: outURL)

        for p in 0 ..< (blockSize - 1) {
            let n = acceptedByPos[p].count
            let acc = n > 0 ? Double(acceptedByPos[p].filter { $0 }.count) / Double(n) : 0
            print(
                "[STS pos \(p)] samples=\(n) empirical-accept=\(String(format: "%.2f", acc)) "
                    + "T=\(String(format: "%.3f", temps[p]))")
        }
        print("[STS] temperatures=\(temps)")
        print("[STS] wrote \(outURL.path)")

        #expect(temps.count == blockSize - 1)
        #expect(temps.allSatisfy { $0 > 0 }, "temperatures must be positive")
    }
}
