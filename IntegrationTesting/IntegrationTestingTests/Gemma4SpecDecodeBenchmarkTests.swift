// Copyright © 2026 Apple Inc.

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
                // Unquantized target — the favorable DSpark regime (slow baseline).
                // MTP skipped: the 12B is Gemma4Unified and the assistant rejects it.
                "mlx-community/gemma-4-12B-it-bf16",
                nil,
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

            // --- DSpark (bf16 drafter, then the same drafter quantized to 8-bit) ---
            if let dsId = entry.dspark, let drafterDir = hfSnapshotDir(modelId: dsId) {
                let drafter = try Gemma4DSparkDrafter.load(directory: drafterDir)
                rows.append(
                    try await measureDSpark(
                        target: entry.target, mode: "DSpark/bf16", context: context,
                        input: lmInput, drafter: drafter))
                // Quantize the drafter's Linears to 8-bit in place and re-measure.
                // The released drafter ships bf16 only; an 8-bit drafter is the
                // realistic deployed config against an 8-bit target and cuts the
                // draft-forward overhead.
                quantize(model: drafter, groupSize: 64, bits: 8, filter: { _, m in m is Linear })
                eval(drafter)
                rows.append(
                    try await measureDSpark(
                        target: entry.target, mode: "DSpark/8b", context: context,
                        input: lmInput, drafter: drafter))
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

    /// DSpark's advantage is domain-shaped: the paper's accepted length τ for
    /// Gemma4-12B rises from ~2.9–3.5 (chat) to ~4.5–6.0 (math/code). This sweeps
    /// chat / math / code / multi-step reasoning on the **bf16** 12B target with
    /// the **8-bit drafter** (the best deployable config) and reports per-domain
    /// acceptance, τ, and wall-clock speedup vs plain decode.
    ///
    /// Prompts elicit inline step-by-step reasoning (non-thinking mode) to match
    /// the released drafter's training. (DeepSpec notes thinking-mode use wants a
    /// re-tuned drafter.)
    @Test
    func testGemma4DSparkAcceptanceByDomain12BBF16() async throws {
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
        quantize(model: drafter, groupSize: 64, bits: 8, filter: { _, m in m is Linear })
        eval(drafter)

        let prompts: [(String, String)] = [
            ("chat", "Why is the sky blue? Explain in one paragraph."),
            (
                "math",
                "Natalia sold clips to 48 friends in April, then half as many in May. "
                    + "How many clips did she sell altogether? Think step by step."
            ),
            (
                "code",
                "Write a Swift function that returns the nth Fibonacci number iteratively, "
                    + "with a brief explanation."
            ),
            (
                "reason",
                "Three friends — Ann, Bob, and Cid — each have a different pet (cat, dog, fish). "
                    + "Ann doesn't have the cat. Bob has the dog. Who has the fish? "
                    + "Reason step by step."
            ),
        ]

        var lines: [String] = []
        let blk = drafter.blockSize  // proposed per round = blk - 1
        for (domain, prompt) in prompts {
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)]))
            let base = try await measurePlain(target: domain, context: context, input: input)
            let ds = try await measureDSpark(
                target: domain, mode: "DSpark", context: context, input: input, drafter: drafter)
            let acc = ds.acceptRate ?? 0
            let tau = acc * Double(blk - 1) + 1.0  // accepted drafts + 1 bonus
            let speedup = (base?.tps ?? 0) > 0 ? ds.tps / base!.tps : 0
            lines.append(
                String(
                    format: "%-8@ %7.0f%% %6.2f %9.2f %9.2f %8.2fx",
                    domain as NSString, acc * 100, tau, base?.tps ?? 0, ds.tps, speedup))
        }

        print("\n=== DSpark 12B bf16 + 8-bit drafter, by domain (temp=0, \(measureTokens) tok) ===")
        print(
            String(
                format: "%-8@ %8@ %6@ %9@ %9@ %8@", "domain" as NSString, "accept%" as NSString,
                "tau" as NSString, "base t/s" as NSString, "DSpark" as NSString,
                "speedup" as NSString))
        for l in lines { print(l) }
        print("=================================================================\n")
        #expect(!lines.isEmpty)
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
        target: String, mode: String, context: ModelContext, input: LMInput,
        drafter: Gemma4DSparkModel
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
            target: target, mode: mode, tps: tokPerSec(info),
            genTokens: info.generationTokenCount,
            acceptRate: proposed > 0 ? Double(accepted) / Double(proposed) : 0,
            passthrough: info.passthroughReason)
    }
}
