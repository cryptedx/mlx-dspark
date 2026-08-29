import AppCore
import SwiftUI

/// The engine-level knobs: decode mode and draft cap.
///
/// Both are speed dials, never behavior dials — the target verifies every token, so output is
/// byte-identical across all of them. Applying reloads the model in place (the CLI's
/// `--mode` / `--max-draft`, via `/admin/load` overrides); the port survives.
struct DecodingCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "Decoding", subtitle: model.decodingLine) {
            DecodingControls()

            Text("Output is byte-identical in every mode — these change speed, not text. "
                 + "Cap Auto calibrates this Mac once and adapts per round; pin a value if "
                 + "you've measured a better fixed cap for this model. Applying reloads the "
                 + "model in place (the server and its port stay up).")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Mode + cap pickers with Apply. Shared between Settings → Decoding and the chat toolbar's
/// settings popover, so the knobs live where the user already is.
struct DecodingControls: View {
    @EnvironmentObject private var model: AppModel

    let compact: Bool

    @State private var mode: String = "auto"
    @State private var cap: String = "auto"
    @State private var confidence: String = "off"
    @State private var contextWindow: String = "default"
    @State private var lookupDrafts: Bool = true
    @State private var kvBits: String = "default"
    @State private var cpuPrefill = false
    /// "on" / "off" — the engine's thinking default for API requests that don't specify it.
    @State private var apiThinking: String = "on"
    @State private var applying = false

    /// Context presets as (tag, label, tokens). "default" = the model's own maximum.
    private static let contextPresets: [(tag: String, label: String, tokens: Int?)] = [
        ("default", "Model max", nil),
        ("8192", "8k", 8192), ("16384", "16k", 16384), ("32768", "32k", 32768),
        ("65536", "64k", 65536), ("131072", "128k", 131072), ("262144", "256k", 262144),
    ]

    private var healthForModel: HealthInfo? {
        guard model.health?.target == model.model else { return nil }
        return model.health
    }

    private var confOptions: [String] {
        var options = ["off", "0.2", "0.3", "0.5"]
        let current = Self.confTag(healthForModel?.confidenceThreshold)
        if !options.contains(current) { options.append(current) }
        return options
    }

    private var modes: [(id: String, label: String)] {
        // availableDecodingModes, NOT availableRaceArms: applying reloads the pair, so the
        // drafter mode stays selectable while Baseline/Lookup is running (it used to vanish).
        var options = [(id: "auto", label: "Auto")]
        for arm in model.availableDecodingModes {
            options.append((id: arm, label: arm == "dspark" ? "DSpark"
                            : arm == "dflash" ? "DFlash" : arm.capitalized))
        }
        return options
    }

    private var isDirty: Bool {
        let saved = Defaults.modelSettings(for: model.model)
        let health = healthForModel
        let baseline = saved ?? health.map(ModelSettings.init) ?? ModelSettings()
        return mode != baseline.mode || cap != baseline.maxDraft
            || confidence != Self.confTag(baseline.confidenceThreshold)
            || contextWindow != Self.contextTag(baseline.contextWindow)
            || lookupDrafts != (baseline.lookupDrafts ?? lookupDrafts)
            || kvBits != Self.kvTag(baseline.kvBits)
            || cpuPrefill != (baseline.cpuPrefill ?? (health?.cpuSplit != nil))
            || apiThinking != (baseline.enableThinking.map { $0 ? "on" : "off" }
                ?? health?.thinkingDefault ?? apiThinking)
    }

    init(compact: Bool = false) {
        self.compact = compact
    }

    var body: some View {
        Group {
            if compact {
                // Three pickers no longer fit one 340pt popover row (the clipped-Apply
                    // lesson, relearned the day Confidence landed): mode+cap on one row,
                    // confidence on its own, Apply last.
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 14) {
                        DecodingModeCapPickers(modes: modes, mode: $mode, cap: $cap)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        DecodingConfidencePicker(options: confOptions, selection: $confidence)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        DecodingContextPicker(presets: Self.contextPresets,
                                              selection: $contextWindow)
                        Spacer(minLength: 0)
                    }
                    if healthForModel?.kvBits != nil {
                        HStack(spacing: 14) {
                            DecodingKVPicker(selection: $kvBits)
                            Spacer(minLength: 0)
                        }
                    }
                    if healthForModel?.lookupDrafts != nil {
                        HStack(spacing: 14) {
                            DecodingLookupToggle(isOn: $lookupDrafts)
                            Spacer(minLength: 0)
                        }
                    }
                    if healthForModel?.thinkingDefault != nil {
                        HStack(spacing: 14) {
                            DecodingAPIThinkingPicker(selection: $apiThinking)
                            Spacer(minLength: 0)
                        }
                    }
                    HStack(spacing: 14) {
                        DecodingCPUPrefillToggle(isOn: $cpuPrefill)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 8) {
                        DecodingApplyControl(
                            applying: applying,
                            enabled: model.isServerReady && isDirty,
                            action: apply)
                        if !applying, isDirty {
                            Text("Reloads the model in place.")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                        Spacer(minLength: 0)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 14) {
                        DecodingModeCapPickers(modes: modes, mode: $mode, cap: $cap)
                        DecodingConfidencePicker(options: confOptions, selection: $confidence)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        DecodingContextPicker(presets: Self.contextPresets,
                                              selection: $contextWindow)
                        if healthForModel?.kvBits != nil {
                            DecodingKVPicker(selection: $kvBits)
                        }
                        if healthForModel?.lookupDrafts != nil {
                            DecodingLookupToggle(isOn: $lookupDrafts)
                        }
                        if healthForModel?.thinkingDefault != nil {
                            DecodingAPIThinkingPicker(selection: $apiThinking)
                        }
                        DecodingApplyControl(
                            applying: applying,
                            enabled: model.isServerReady && isDirty,
                            action: apply)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 14) {
                        DecodingCPUPrefillToggle(isOn: $cpuPrefill)
                        Spacer(minLength: 0)
                    }
                }
            }
        }
        // The pickers show what the engine is actually running, not a stale default — the
        // server reports both (`/health` mode + max_draft), so a reopened popover agrees
        // with what was applied.
        .onAppear(perform: syncFromModel)
        .onChange(of: model.model) { _, _ in syncFromModel() }
    }

    private func syncFromModel() {
        let saved = Defaults.modelSettings(for: model.model)
        let health = healthForModel
        mode = saved?.mode ?? health?.mode ?? "auto"
        cap = saved?.maxDraft ?? health?.maxDraft ?? "auto"
        confidence = Self.confTag(saved?.confidenceThreshold ?? health?.confidenceThreshold)
        contextWindow = Self.contextTag(saved?.contextWindow ?? health?.contextWindow)
        lookupDrafts = saved?.lookupDrafts ?? health?.lookupDrafts ?? true
        kvBits = Self.kvTag(saved?.kvBits ?? health?.kvBits)
        cpuPrefill = saved?.cpuPrefill ?? (health?.cpuSplit != nil)
        apiThinking = saved?.enableThinking.map { $0 ? "on" : "off" }
            ?? health?.thinkingDefault ?? "on"
    }

    private func apply() {
        applying = true
        Task {
            await model.applyEngineSettings(
                mode: mode, cap: cap,
                confidence: confidence == "off" ? 0.0 : Double(confidence),
                contextWindow: contextWindow == "default" ? 0 : Int(contextWindow),
                // Sent only when the user actually flipped it, so an untouched Apply
                // keeps riding the pair's measured default instead of pinning it.
                lookupDrafts: lookupDrafts == healthForModel?.lookupDrafts ? nil : lookupDrafts,
                // Same: unchanged -> nil (keep the server's setting); "default" -> explicit
                // 0 (full precision). Gated on health reporting the field at all.
                kvBits: kvBits == Self.kvTag(healthForModel?.kvBits) ? nil
                        : (kvBits == "default" ? 0 : Int(kvBits)),
                cpuPrefill: cpuPrefill == (healthForModel?.cpuSplit != nil) ? nil : cpuPrefill,
                // Unchanged -> nil (keep); "on" -> true = the model's own default.
                enableThinking: apiThinking == healthForModel?.thinkingDefault ? nil
                        : (apiThinking == "on"))
            applying = false
        }
    }

    private static func contextTag(_ value: Int?) -> String {
        guard let value, contextPresets.contains(where: { $0.tokens == value })
        else { return "default" }
        return String(value)
    }

    /// Health's kv_bits (0 = full precision) as a picker tag.
    private static func kvTag(_ value: Int?) -> String {
        guard let value, value == 4 || value == 8 else { return "default" }
        return String(value)
    }

    /// Health's 0.0/0.2/0.3/0.5 as picker tags ("off"/"0.2"/…). Values outside the preset
    /// list (a server started with an unusual --confidence-threshold) round to one decimal
    /// and appear as their own tag so the picker never lies about the running state.
    private static func confTag(_ value: Double?) -> String {
        guard let value, value > 0 else { return "off" }
        return String(format: "%.1f", value)
    }
}
