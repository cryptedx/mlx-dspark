import AppCore
import SwiftUI

/// Everything useful for model management: models already on this disk, measured model
/// recommendations that are not downloaded yet, and any Hugging Face repo the user types.
///
/// The design keeps LM Studio's most-praised idea — answer "will this fit?" *before* someone
/// downloads 15 GB — and adds the thing only this project has to show: the **pair**. A row
/// names the target and the drafter that auto-resolves for it, because a speculative setup is
/// two models or it is nothing. Anything without a pair still runs: `--mode auto` gives every
/// model drafter-free lookup speculation.
struct ModelsScreen: View {
    @EnvironmentObject private var model: AppModel
    @State private var measuredPairsExpanded = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let error = model.modelSwitchError {
                    SwapErrorBanner(message: error) { model.modelSwitchError = nil }
                }
                if let error = model.downloadError {
                    SwapErrorBanner(title: "The model couldn't be downloaded.", message: error) {
                        model.downloadError = nil
                    }
                }

                if model.isModelLoading || model.isModelDownloading {
                    // The screen most downloads start from is also where they should be
                    // stoppable — the same progress + Cancel the empty chat shows.
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            let name = model.model.components(separatedBy: "/").last ?? model.model
                            Text(model.loadingDetail ?? "Loading \(name)…")
                                .font(.callout.weight(.medium))
                        }
                        if let dl = model.downloadProgress {
                            DownloadProgressRow(progress: dl)
                        }
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.5)))
                }

                header

                if !model.poolModelStatuses.isEmpty {
                    sectionTitle("Loaded now", note: "These models use memory right now. "
                                 + "Unload one to free it; keep one in memory to prevent idle eviction.")
                    ForEach(model.poolModelStatuses) { status in
                        PoolStatusRow(status: status) {
                            Task { await model.setKeepLoaded(status.model, !status.pinned) }
                        } unload: {
                            Task { await model.unloadModel(status.model) }
                        }
                    }
                }

                let drafters = model.installedModels.filter(\.isDrafter)
                if !drafters.isEmpty {
                    DisclosureGroup {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(drafters) { drafter in
                                HStack {
                                    Text(drafter.shortRepo).font(.caption.monospaced())
                                    Spacer()
                                    Text(drafter.size).font(.caption).foregroundStyle(.secondary)
                                    RevealButton(path: drafter.path)
                                }
                            }
                        }
                        .padding(.top, 6)
                    } label: {
                        Text("Drafters on disk (\(drafters.count)) — resolved automatically")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(.top, 2)
                }

                let onDisk = model.installedModels.filter { !$0.isDrafter }
                if !onDisk.isEmpty {
                    sectionTitle("On this Mac",
                                 note: "Already downloaded. Anything here loads instantly; "
                                     + "unpaired models run with lookup speculation.")
                    ForEach(onDisk) { installed in
                        InstalledRowView(installed: installed,
                                         isLoaded: model.poolStatus(for: installed.repo)?.ready
                                             ?? (model.isServerReady && installed.repo == model.model))
                    }
                }

                sectionTitle("Download any model",
                             note: "Download any MLX-compatible model directly from Hugging Face.")
                AnyModelField()

                let measuredModelsNotOnDisk = model.models.filter { !$0.targetInstalled }
                if !measuredModelsNotOnDisk.isEmpty {
                    DisclosureGroup(isExpanded: $measuredPairsExpanded) {
                        // These are real target models, not standalone drafter entries. They
                        // confirm first, then download only; loading happens from On this Mac.
                        ForEach(measuredModelsNotOnDisk) { row in
                            ModelRowView(row: row,
                                         isLoaded: model.poolStatus(for: row.target)?.ready
                                             ?? (model.isServerReady && row.target == model.model),
                                         canLoad: model.poolStatus(for: row.target)?.ready != true
                                             && !model.isModelLoading && !model.isModelDownloading,
                                         canUnload: !model.isModelLoading && !model.isModelDownloading) {
                                Task { await model.downloadModel(row.target) }
                            } onUnload: {
                                Task { await model.unloadModel(row.target) }
                            }
                        }
                    } label: {
                        sectionTitle("Measured pairs",
                                     note: "Real models with a benchmarked drafter pair — not downloaded yet.")
                    }
                }
                if model.models.isEmpty {
                    ContentUnavailableView("No models listed", systemImage: "shippingbox")
                        .frame(height: 160)
                }
            }
            .padding(16)
        }
        .task { await model.refreshDiagnostics() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                if let ram = model.doctorReport?.environment.ramGB {
                    Text("\(model.doctorReport?.environment.device ?? "This Mac") · \(ram, specifier: "%.0f") GB memory")
                }
                if let disk = model.diskUsage {
                    Text("· \(disk.total) of models on disk")
                }
            }
            .font(.caption).foregroundStyle(.secondary)

            Text("Load brings a model into memory. Unload frees it again; Keep in memory "
                 + "prevents automatic idle eviction.")
                .font(.caption).foregroundStyle(.secondary)

            // The badges are M4 Pro measurements. Say how this Mac compares instead of
            // letting an M5 Max owner read them as a ceiling (or an M1 owner as a promise).
            if let bw = model.bandwidth, let scale = bw.scale, abs(scale - 1.0) > 0.05 {
                Text(String(format: "Speedup badges were measured on an M4 Pro (273 GB/s). "
                            + "This Mac's memory bandwidth is %.1f× that, so absolute tok/s "
                            + "scale roughly with it; the ratios carry over.", scale))
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func sectionTitle(_ title: String, note: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.headline)
            Text(note).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.top, 6)
    }
}

private struct PoolStatusRow: View {
    @EnvironmentObject private var model: AppModel
    let status: PoolModelStatus
    let togglePin: () -> Void
    let unload: () -> Void

    private var shortName: String {
        status.model.components(separatedBy: "/").last ?? status.model
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text(shortName).font(.callout.weight(.medium))
                Text(status.state.capitalized)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(status.ready ? Theme.verified : Theme.warning)
                if status.pinned {
                    Label("Pinned", systemImage: "pin.fill")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                if status.ready {
                    HStack(spacing: 6) {
                        Button("Unload", action: unload)
                            .buttonStyle(.bordered).controlSize(.small)
                            .disabled(model.isModelLoading || model.isModelDownloading)
                        Button(status.pinned ? "Unpin" : "Keep in memory", action: togglePin)
                            .buttonStyle(.bordered).controlSize(.small)
                            .disabled(model.isModelLoading || model.isModelDownloading)
                    }
                }
            }
            if let reason = status.evictionReason {
                Text("Last transition: \(reason)").font(.caption).foregroundStyle(.secondary)
            }
            if let warning = status.warning ?? status.restoreError ?? status.error {
                Text(warning).font(.caption).foregroundStyle(Theme.warning)
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.cardStroke, lineWidth: 1))
    }
}

/// Download any Hugging Face repo — the engine serves anything MLX-compatible.
struct AnyModelField: View {
    @EnvironmentObject private var model: AppModel
    @State private var repo: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                TextField("org/model — any MLX model on Hugging Face",
                          text: $repo)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout.monospaced())
                    .onSubmit(download)
                Button("Download") { download() }
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmed.isEmpty || model.isModelLoading || model.isModelDownloading)
            }
            Text("Downloads the model to this Mac without loading it. Start it later from "
                 + "the installed-model row.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(12)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.cardStroke, lineWidth: 1))
    }

    private var trimmed: String {
        repo.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func download() {
        let target = trimmed
        guard !target.isEmpty else { return }
        repo = ""
        Task { await model.downloadModel(target) }
    }
}

struct SwapErrorBanner: View {
    let title: String
    let message: String
    let dismiss: () -> Void

    init(title: String = "That model didn't load — the previous one was restored.",
         message: String, dismiss: @escaping () -> Void) {
        self.title = title
        self.message = message
        self.dismiss = dismiss
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.warning)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.callout.weight(.medium))
                Text(message).font(.caption).foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Spacer()
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark").imageScale(.small)
            }
            .buttonStyle(.borderless)
        }
        .padding(12)
        .background(Theme.warning.opacity(0.10), in: RoundedRectangle(cornerRadius: 9))
    }
}

struct ModelRowView: View {
    let row: ModelRow
    let isLoaded: Bool
    var canLoad: Bool = false
    var canUnload: Bool = true
    var onLoad: () -> Void = {}
    var onUnload: () -> Void = {}
    @State private var confirmingDownload = false

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Button(action: load) {
                details
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!canLoad)

            Spacer(minLength: 4)
            if let speedup = row.speedup {
                Text(speedup)
                    .font(.system(.callout, design: .rounded).monospacedDigit())
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.spark)
                    .help("Measured on an M4 Pro; yours will differ")
            }
            if isLoaded {
                Button("Unload", action: onUnload)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(!canUnload)
            } else if canLoad {
                Button(row.targetInstalled ? "Load" : "Download", action: load)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
            isLoaded ? AnyShapeStyle(Theme.spark.opacity(0.45)) : Theme.cardStroke, lineWidth: 1))
        .confirmationDialog(
            "Download \(row.shortTarget)\(row.ram.map { " (\($0))" } ?? "")?",
            isPresented: $confirmingDownload, titleVisibility: .visible
        ) {
            Button("Download") { onLoad() }
        } message: {
            Text(row.fits == false
                 ? "This model looks too large for this Mac — it may not run once downloaded."
                 : "Downloads the missing model files. You can load it afterwards from On this Mac.")
        }
    }

    private var details: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text(row.shortTarget).font(.headline)
                if isLoaded { LoadedBadge() }
            }

            // The pairing — this app's domain language.
            if let drafter = row.shortDrafter {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.triangle.merge").imageScale(.small)
                    Text(drafter).font(.caption.monospaced())
                }
                .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                badge(fitsLabel, systemImage: fitsSymbol, tint: fitsTint)
                badge(stateLabel, systemImage: stateSymbol,
                      tint: row.ready ? Theme.verified : .secondary)
                if let ram = row.ram {
                    Text(ram).font(.caption).foregroundStyle(.secondary)
                }
                // Physics, not a promise: bandwidth ÷ weight bytes is the most a plain
                // decode of these weights can do on this Mac. Speculation multiplies it.
                if let ceiling = row.ceilingTps {
                    Text(String(format: "~%.0f tok/s plain ceiling here", ceiling))
                        .font(.caption).foregroundStyle(.secondary)
                        .help("This Mac's memory bandwidth divided by the model's weight "
                              + "bytes — the single-stream roofline. Speculative decoding "
                              + "is what beats it.")
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func load() {
        // A local target loads immediately; a missing target confirms first — a mis-click must
        // not start a multi-gigabyte fetch.
        if row.targetInstalled { onLoad() } else { confirmingDownload = true }
    }

    private var fitsLabel: String {
        switch row.fits {
        case true?:  return "fits this Mac"
        case false?: return "too large"
        default:     return "unknown"
        }
    }

    private var fitsSymbol: String {
        switch row.fits {
        case true?:  return "checkmark.circle.fill"
        case false?: return "exclamationmark.triangle.fill"
        default:     return "questionmark.circle"
        }
    }

    private var fitsTint: Color {
        switch row.fits {
        case true?:  return Theme.verified
        case false?: return Theme.warning
        default:     return .secondary
        }
    }

    /// Deliberately says which *half* is missing: with speculative decoding a model can be
    /// half-downloaded in a way that matters, and "not downloaded" would hide that.
    private var stateLabel: String {
        if row.ready { return "ready" }
        if row.targetInstalled { return "drafter not downloaded" }
        if row.drafterInstalled { return "target not downloaded" }
        return "not downloaded"
    }

    private var stateSymbol: String {
        row.ready ? "internaldrive.fill" : "arrow.down.circle"
    }

    private func badge(_ text: String, systemImage: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage).imageScale(.small).foregroundStyle(tint)
            Text(text)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

/// A model that's already on disk — load it, reveal it, or reclaim the space.
struct InstalledRowView: View {
    @EnvironmentObject private var model: AppModel
    let installed: InstalledModel
    let isLoaded: Bool
    @State private var confirmingDelete = false

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(installed.shortRepo).font(.body.weight(.medium))
                    if isLoaded { LoadedBadge() }
                }
                HStack(spacing: 8) {
                    Text(installed.size).font(.caption).foregroundStyle(.secondary)
                    if installed.registryID != nil {
                        HStack(spacing: 3) {
                            Image(systemName: "arrow.triangle.merge").imageScale(.small)
                            Text("measured pair available")
                        }
                        .font(.caption)
                        .foregroundStyle(Theme.spark)
                    } else {
                        Text("lookup speculation").font(.caption).foregroundStyle(.secondary)
                    }
                    if let label = installed.sourceLabel {
                        Text(label).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            if !isLoaded {
                Button("Load") { Task { await model.switchModel(to: installed.repo) } }
                    .controlSize(.small)
                    .disabled(model.isModelLoading || model.isModelDownloading)
            } else {
                Button("Unload") { Task { await model.unloadModel(installed.repo) } }
                    .controlSize(.small)
                    .disabled(model.isModelLoading || model.isModelDownloading)
            }
            RevealButton(path: installed.path)
            // LM Studio's downloads and the user's own model folders are not our files —
            // we read them, we never offer to delete them.
            if !installed.isExternal {
                Button {
                    confirmingDelete = true
                } label: {
                    Image(systemName: "trash").imageScale(.small)
                }
                .buttonStyle(.borderless)
                .disabled(isLoaded || model.isModelLoading || model.isModelDownloading)
                .help(isLoaded ? "Can't delete the loaded model" : "Move to Trash")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(Theme.cardFill, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
            isLoaded ? AnyShapeStyle(Theme.spark.opacity(0.45)) : Theme.cardStroke, lineWidth: 1))
        .confirmationDialog("Move \(installed.shortRepo) to the Trash?",
                            isPresented: $confirmingDelete, titleVisibility: .visible) {
            Button("Move to Trash (\(installed.size))", role: .destructive) {
                delete()
            }
        } message: {
            Text("Recoverable from the Trash. Loading it again later re-downloads it.")
        }
    }

    private func delete() {
        let url = URL(fileURLWithPath: installed.path)
        do {
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
            Task { await model.refreshDiagnostics() }
        } catch {
            model.modelSwitchError = "Couldn't move \(installed.shortRepo) to the Trash: "
                + error.localizedDescription
        }
    }
}

struct LoadedBadge: View {
    var body: some View {
        Text("loaded")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(.tint.opacity(0.18), in: Capsule())
            .foregroundStyle(.tint)
    }
}

struct RevealButton: View {
    let path: String

    var body: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
        } label: {
            Image(systemName: "magnifyingglass").imageScale(.small)
        }
        .buttonStyle(.borderless)
        .help("Reveal in Finder")
    }
}
