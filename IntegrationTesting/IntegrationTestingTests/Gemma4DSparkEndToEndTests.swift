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

// MARK: - DSpark Gemma 4 12B end-to-end
//
// Sibling of DSparkEndToEndTests (Qwen3). Exercises the full DSpark pipeline
// against the real released checkpoints:
//   target  = mlx-community/gemma-4-12B-it-bf16   (gemma4_unified; bf16 matches
//             the precision the drafter conditions on)
//   drafter = deepseek-ai/dspark_gemma4_12b_block7
// Validates the per-target wiring the unit/parity tests can't: that Gemma 4's
// DSpark multi-layer capture seam fires (mtpCaptureLayersKey → the post-block
// hiddens at target_layer_ids [5,17,29,41,46]), the drafter proposes, and the
// accept loop accepts. Gated on both checkpoints being in the HF cache.

private struct Gemma4DSparkPair {
    let context: ModelContext
    let drafter: Gemma4DSparkModel
}

private func loadGemma4DSparkPair(
    targetModelId: String = "mlx-community/gemma-4-12B-it-bf16",
    drafterModelId: String = "deepseek-ai/dspark_gemma4_12b_block7"
) async throws -> Gemma4DSparkPair? {
    guard let targetDir = hfSnapshotDir(modelId: targetModelId) else { return nil }
    guard let drafterDir = hfSnapshotDir(modelId: drafterModelId) else { return nil }
    let context = try await VLMModelFactory.shared.load(
        from: targetDir, using: #huggingFaceTokenizerLoader())
    let drafter = try Gemma4DSparkDrafter.load(directory: drafterDir)
    return Gemma4DSparkPair(context: context, drafter: drafter)
}

@Suite(.serialized)
struct Gemma4DSparkEndToEndTests {

    /// Headline: load the real Gemma 4 12B + DSpark pair, run 64 tokens at
    /// temp=0, and assert speculation actually runs end-to-end (no sticky-
    /// passthrough, proposals > 0, at least one accepted draft). Logs the
    /// observed acceptance rate + tok/s and the DSpark-vs-baseline speedup.
    @Test
    func testGemma4DSparkProducesAcceptedDrafts() async throws {
        guard let pair = try await loadGemma4DSparkPair() else {
            Issue.record("Gemma 4 12B target or DSpark drafter not in HF cache; skipping")
            return
        }

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await pair.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: pair.context,
            dsparkDrafter: pair.drafter  // blockSize defaults to the drafter's block_size (7)
        )

        var info: GenerateCompletionInfo?
        var text = ""
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .toolCall: break
            case .info(let i): info = i
            }
        }

        guard let info else {
            Issue.record("DSpark stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? 0
        let accepted = info.acceptedDraftTokens ?? 0
        let rate = proposed > 0 ? Double(accepted) / Double(proposed) : 0
        let tps =
            info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0
        print(
            "[DSpark Gemma4-12B] proposed=\(proposed) accepted=\(accepted) "
                + "rate=\(String(format: "%.1f%%", rate * 100)) "
                + "gen=\(info.generationTokenCount) in \(info.generateTime.formatted())s "
                + "tok/s=\(String(format: "%.2f", tps))")
        print("[DSpark Gemma4-12B] text: \(text)")

        // Baseline (no speculation) at the same config, for the speedup ratio.
        let baseStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: pair.context)
        var baseInfo: GenerateCompletionInfo?
        for await event in baseStream {
            if case .info(let i) = event { baseInfo = i }
        }
        if let baseInfo {
            let baseTps =
                baseInfo.generateTime > 0
                ? Double(baseInfo.generationTokenCount) / baseInfo.generateTime : 0
            let speedup = baseTps > 0 ? tps / baseTps : 0
            print(
                "[DSpark Gemma4-12B] baseline tok/s=\(String(format: "%.2f", baseTps)) "
                    + "(gen=\(baseInfo.generationTokenCount) in \(baseInfo.generateTime.formatted())s) "
                    + "| DSpark speedup=\(String(format: "%.2fx", speedup))")
        }

        #expect(
            info.passthroughReason == nil,
            "DSpark sticky-passthrough (target may not emit layer hiddens / capture seam not wired): \(info.passthroughReason ?? "")"
        )
        #expect(
            proposed > 0,
            "no tokens proposed — speculation did not run (capture seam / iterator wiring)")
        #expect(
            accepted > 0,
            "drafts proposed but none accepted — target/drafter mismatch or drafter bug")
    }

    /// Sanity: at temp=0 DSpark's emitted prefix should track baseline
    /// non-speculative generation (bounded to a short prefix by MLX SDPA
    /// shape-determinism, like the Qwen3 e2e).
    @Test
    func testGemma4DSparkPrefixTracksBaseline() async throws {
        guard let pair = try await loadGemma4DSparkPair() else {
            Issue.record("checkpoints not in HF cache; skipping")
            return
        }

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await pair.context.processor.prepare(input: userInput)

        let dsStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: pair.context, dsparkDrafter: pair.drafter)
        var dsText = ""
        var dsInfo: GenerateCompletionInfo?
        for await e in dsStream {
            switch e {
            case .chunk(let c): dsText += c
            case .info(let i): dsInfo = i
            default: break
            }
        }

        let baseStream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 32, temperature: 0),
            context: pair.context)
        var baseText = ""
        for await e in baseStream {
            if case .chunk(let c) = e { baseText += c }
        }

        print("[DSpark Gemma4 prefix] dspark:   \(dsText)")
        print("[DSpark Gemma4 prefix] baseline: \(baseText)")

        #expect(
            (dsInfo?.proposedDraftTokens ?? 0) > 0, "speculation did not run in the DSpark stream")

        let dsTok = pair.context.tokenizer.encode(text: dsText, addSpecialTokens: false)
        let baseTok = pair.context.tokenizer.encode(text: baseText, addSpecialTokens: false)
        let n = Swift.min(8, dsTok.count, baseTok.count)
        #expect(n > 0, "no tokens generated to compare")
        #expect(
            Array(dsTok.prefix(n)) == Array(baseTok.prefix(n)),
            "DSpark diverged from baseline within the first \(n) tokens at temp=0")
    }
}
