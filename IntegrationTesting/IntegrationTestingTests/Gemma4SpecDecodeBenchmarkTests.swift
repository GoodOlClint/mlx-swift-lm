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

// MARK: - Gemma 4 spec-decode tok/s benchmark (8-bit targets)
//
// Measures decode tok/s for Gemma 4 at 8-bit across three modes — plain
// autoregressive (default), MTP (Gemma4 assistant drafter), and DSpark
// (deepseek-ai dspark_gemma4_12b_block7) — over the available weight sizes.
//
// Matrix (drafter availability gates the cells):
//   12B-8bit:      default | MTP (12B assistant) | DSpark (12B drafter)
//   26B-A4B-8bit:  default | MTP (26B assistant) | —  (no DSpark drafter)
//   31B-8bit:      default | MTP (31B assistant) | —  (no DSpark drafter)
//
// Each mode is warmed up (8 tokens, excluded) before the measured run so Metal
// kernel compilation doesn't pollute the number. temp=0, fixed prompt. Gated on
// the checkpoints being in the HF cache; prints a summary table.

private let benchPrompt = "Why is the sky blue? Explain in one paragraph."
private let warmupTokens = 8
private let measureTokens = 128

private struct BenchRow {
    let target: String
    let mode: String
    let tps: Double
    let genTokens: Int
    let acceptRate: Double?  // nil for plain decode
    let passthrough: String?
}

private func tokPerSec(_ info: GenerateCompletionInfo) -> Double {
    info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0
}

@Suite(.serialized)
struct Gemma4SpecDecodeBenchmarkTests {

    @Test
    func testGemma4SpecDecodeTokensPerSecond8Bit() async throws {
        // (target id, MTP assistant id or nil, DSpark drafter id or nil)
        let matrix: [(target: String, mtp: String?, dspark: String?)] = [
            (
                "mlx-community/gemma-4-12B-it-8bit",
                "mlx-community/gemma-4-12B-it-assistant-bf16",
                "deepseek-ai/dspark_gemma4_12b_block7"
            ),
            (
                "mlx-community/gemma-4-26b-a4b-it-8bit",
                "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
                nil
            ),
            (
                "mlx-community/gemma-4-31b-it-8bit",
                "mlx-community/gemma-4-31B-it-assistant-bf16",
                nil
            ),
        ]

        var rows: [BenchRow] = []

        for entry in matrix {
            guard let targetDir = hfSnapshotDir(modelId: entry.target) else {
                print("[bench] SKIP \(entry.target) — not in HF cache")
                continue
            }
            // Load the target once; reuse across all three modes. Scoped so the
            // context (and its weights) release before the next, larger target.
            let context = try await VLMModelFactory.shared.load(
                from: targetDir, using: #huggingFaceTokenizerLoader())
            let lmInput = try await context.processor.prepare(
                input: UserInput(chat: [.user(benchPrompt)]))

            // --- default (plain autoregressive) ---
            if let row = try await measurePlain(target: entry.target, context: context, input: lmInput) {
                rows.append(row)
            }

            // --- MTP (Gemma4 assistant) ---
            if let mtpId = entry.mtp, let drafterDir = hfSnapshotDir(modelId: mtpId) {
                let cfg = try JSONDecoder().decode(
                    Gemma4AssistantConfiguration.self,
                    from: Data(contentsOf: drafterDir.appendingPathComponent("config.json")))
                let drafter = Gemma4AssistantDraftModel(cfg)
                try loadWeights(modelDirectory: drafterDir, model: drafter)
                rows.append(
                    try await measureMTP(
                        target: entry.target, context: context, input: lmInput, drafter: drafter))
            } else if entry.mtp != nil {
                print("[bench] SKIP MTP for \(entry.target) — drafter not in cache")
            }

            // --- DSpark ---
            if let dsId = entry.dspark, let drafterDir = hfSnapshotDir(modelId: dsId) {
                let drafter = try Gemma4DSparkDrafter.load(directory: drafterDir)
                rows.append(
                    try await measureDSpark(
                        target: entry.target, context: context, input: lmInput, drafter: drafter))
            }
        }

        // --- summary table ---
        print("\n=== Gemma 4 spec-decode tok/s (8-bit, temp=0, \(measureTokens) tok) ===")
        print(
            String(
                format: "%-34@ %-8@ %9@ %9@ %8@", "target" as NSString, "mode" as NSString,
                "tok/s" as NSString, "accept%" as NSString, "speedup" as NSString))
        // Group by target so speedup is vs that target's default.
        var defaultTps: [String: Double] = [:]
        for r in rows where r.mode == "default" { defaultTps[r.target] = r.tps }
        for r in rows {
            let shortTarget = r.target.replacingOccurrences(
                of: "mlx-community/gemma-4-", with: "")
            let base = defaultTps[r.target]
            let speedup = (base ?? 0) > 0 ? r.tps / base! : 0
            let acc = r.acceptRate.map { String(format: "%.0f%%", $0 * 100) } ?? "-"
            let spd = r.mode == "default" ? "-" : String(format: "%.2fx", speedup)
            let pt = r.passthrough != nil ? "  PASSTHROUGH(\(r.passthrough!))" : ""
            print(
                String(
                    format: "%-34@ %-8@ %9.2f %9@ %8@%@",
                    shortTarget as NSString, r.mode as NSString, r.tps,
                    acc as NSString, spd as NSString, pt as NSString))
        }
        print("====================================================================\n")

        #expect(!rows.isEmpty, "no benchmark rows produced — no checkpoints in cache?")
    }

    // MARK: - per-mode measurement (warm-up excluded)

    private func runStream(
        _ input: LMInput, _ context: ModelContext, maxTokens: Int,
        build: (LMInput, GenerateParameters, ModelContext) throws -> AsyncStream<Generation>
    ) async rethrows -> GenerateCompletionInfo? {
        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let stream = try build(input, params, context)
        var info: GenerateCompletionInfo?
        for await event in stream { if case .info(let i) = event { info = i } }
        return info
    }

    private func measurePlain(
        target: String, context: ModelContext, input: LMInput
    ) async throws -> BenchRow? {
        _ = try await runStream(
            input, context, maxTokens: warmupTokens,
            build: { i, p, c in try generate(input: i, parameters: p, context: c) })
        let info = try await runStream(
            input, context, maxTokens: measureTokens,
            build: { i, p, c in try generate(input: i, parameters: p, context: c) })
        guard let info else { return nil }
        return BenchRow(
            target: target, mode: "default", tps: tokPerSec(info),
            genTokens: info.generationTokenCount, acceptRate: nil, passthrough: nil)
    }

    private func measureMTP(
        target: String, context: ModelContext, input: LMInput,
        drafter: Gemma4AssistantDraftModel
    ) async throws -> BenchRow {
        let run: (LMInput, GenerateParameters, ModelContext) throws -> AsyncStream<Generation> = {
            i, p, c in
            try generate(input: i, parameters: p, context: c, mtpDrafter: drafter, blockSize: 4)
        }
        _ = try await runStream(input, context, maxTokens: warmupTokens, build: run)
        let info = try await runStream(input, context, maxTokens: measureTokens, build: run)!
        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        return BenchRow(
            target: target, mode: "MTP", tps: tokPerSec(info),
            genTokens: info.generationTokenCount,
            acceptRate: proposed > 0 ? Double(accepted) / Double(proposed) : 0,
            passthrough: info.passthroughReason)
    }

    private func measureDSpark(
        target: String, context: ModelContext, input: LMInput, drafter: Gemma4DSparkModel
    ) async throws -> BenchRow {
        let run: (LMInput, GenerateParameters, ModelContext) throws -> AsyncStream<Generation> = {
            i, p, c in
            try generate(input: i, parameters: p, context: c, dsparkDrafter: drafter)
        }
        _ = try await runStream(input, context, maxTokens: warmupTokens, build: run)
        let info = try await runStream(input, context, maxTokens: measureTokens, build: run)!
        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        return BenchRow(
            target: target, mode: "DSpark", tps: tokPerSec(info),
            genTokens: info.generationTokenCount,
            acceptRate: proposed > 0 ? Double(accepted) / Double(proposed) : 0,
            passthrough: info.passthroughReason)
    }
}
