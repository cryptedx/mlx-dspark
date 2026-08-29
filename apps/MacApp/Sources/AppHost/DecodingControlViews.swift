import SwiftUI

struct DecodingModeCapPickers: View {
    let modes: [(id: String, label: String)]
    @Binding var mode: String
    @Binding var cap: String

    var body: some View {
        Picker("Mode", selection: $mode) {
            ForEach(modes, id: \.id) { Text($0.label).tag($0.id) }
        }
        .fixedSize()

        Picker("Cap", selection: $cap) {
            Text("Auto").tag("auto")
            ForEach(1...8, id: \.self) { Text("\($0)").tag("\($0)") }
        }
        .fixedSize()
    }
}

struct DecodingConfidencePicker: View {
    let options: [String]
    @Binding var selection: String

    var body: some View {
        Picker("Confidence", selection: $selection) {
            ForEach(options, id: \.self) { Text($0 == "off" ? "Off" : $0).tag($0) }
        }
        .fixedSize()
        .help("Confidence-head early stop: the drafter truncates its own block when it "
              + "stops believing in it. Pays only where the verify curve still rises inside "
              + "the cap AND the drafter leaves acceptance headroom — e.g. the "
              + "Qwen3.6-35B-A3B MoE. Off is right where the drafter already accepts near "
              + "its ceiling (Qwen3.8-27B) or the curve is flat (8-bit targets).")
    }
}

struct DecodingContextPicker: View {
    let presets: [(tag: String, label: String, tokens: Int?)]
    @Binding var selection: String

    var body: some View {
        Picker("Context", selection: $selection) {
            ForEach(presets, id: \.tag) { Text($0.label).tag($0.tag) }
        }
        .fixedSize()
        .help("Cap the context window below the model's own maximum — a RAM lever: the "
              + "KV cache grows with every token of context (~84 KB/token on the "
              + "Qwen3.8-27B pair), so a long agent session at full context can add "
              + "many GB. Requests past the cap get a clear \"prompt is too long\", "
              + "which agent clients like Claude Code auto-compact on.")
    }
}

struct DecodingKVPicker: View {
    @Binding var selection: String

    var body: some View {
        Picker("KV cache", selection: $selection) {
            Text("Full").tag("default")
            Text("8-bit").tag("8")
            Text("4-bit").tag("4")
        }
        .fixedSize()
        .help("Quantize the KV cache (the per-token memory that grows with context): 8-bit "
              + "halves it, 4-bit quarters it — the long-context RAM lever after the "
              + "context cap. Output stays lossless the same way the rest of the engine "
              + "is (the target verifies every token against its own kv-quantized "
              + "forward); quality at very long contexts is the usual KV-quantization "
              + "trade. Full = the model's own precision.")
    }
}

struct DecodingLookupToggle: View {
    @Binding var isOn: Bool

    var body: some View {
        Toggle("Lookup drafts", isOn: $isOn)
            .fixedSize()
            .help("Hybrid n-gram drafts: a 4-gram match in the context supplies a free "
                  + "draft instead of running the drafter that round. Shipped per-pair at "
                  + "its measured best — OFF where extra verify rows cost more than the "
                  + "free draft saves (every MoE, the 4-bit 27B hybrids), on elsewhere. "
                  + "Flip it to A/B on your own content; it can't affect output, only speed.")
    }
}

struct DecodingCPUPrefillToggle: View {
    @Binding var isOn: Bool

    var body: some View {
        Toggle("CPU prefill (experimental)", isOn: $isOn)
            .fixedSize()
            .help("Speeds up long uncached prompts by using CPU and GPU together. "
                  + "Does not affect decode speed. May crash MLX on some Macs. "
                  + "Stored separately for each model.")
    }
}

struct DecodingAPIThinkingPicker: View {
    @Binding var selection: String

    var body: some View {
        Picker("Thinking (API clients)", selection: $selection) {
            Text("On").tag("on")
            Text("Off").tag("off")
        }
        .fixedSize()
        .help("What requests from other apps get when they don't specify thinking "
              + "(DSH, WorkBuddy, Claude Code, pi — most have no reasoning toggle for local "
              + "models). Off stops the long think-before-answer on every agent turn; a "
              + "request that asks for thinking explicitly still gets it. The app's own chat "
              + "keeps its per-chat Allow thinking setting. Sticks across model changes.")
    }
}

struct DecodingApplyControl: View {
    let applying: Bool
    let enabled: Bool
    let action: () -> Void

    var body: some View {
        if applying {
            ProgressView().controlSize(.small)
        } else {
            Button("Apply", action: action)
                .disabled(!enabled)
        }
    }
}
