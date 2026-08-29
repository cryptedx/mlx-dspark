import AppCore
import SwiftUI

/// Versions and updates. Two version numbers on purpose: the app and the engine release
/// independently — the engine keeps itself on the latest release automatically, the app
/// updates through Homebrew (or a fresh DMG) and only *tells* you when one exists.
struct AboutCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "About") {
            VStack(alignment: .leading, spacing: 7) {
                SettingsInfoRow(label: "App", value: AppIdentity.appVersion)
                SettingsInfoRow(label: "Engine", value: model.doctorReport?.environment.version ?? "—")
                if let update = model.appUpdate {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("App v\(update.version) is available.",
                              systemImage: "arrow.down.circle.fill")
                            .font(.callout).foregroundStyle(Theme.spark)
                        HStack(spacing: 8) {
                            Text("brew upgrade --cask mlx-dspark")
                                .font(.caption.monospaced()).textSelection(.enabled)
                            CopyButton(text: "brew upgrade --cask mlx-dspark")
                            Button("Release notes", action: openReleaseNotes)
                                .buttonStyle(.link).font(.caption)
                        }
                    }
                    .padding(.top, 4)
                } else {
                    Text("The engine stays on the latest release automatically; updates are "
                         + "checked at launch and every few hours while the app runs.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let engineUpdate = model.engineUpdateAvailable {
                    HStack(spacing: 8) {
                        Label("Engine \(engineUpdate) is available.",
                              systemImage: "arrow.triangle.2.circlepath")
                            .font(.caption).foregroundStyle(.secondary)
                        Button(model.engineUpdating ? "Updating…" : "Update now",
                               action: updateEngine)
                            .buttonStyle(.link).font(.caption)
                            .disabled(model.engineUpdating)
                        Text("or it installs on the next launch")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                HStack(spacing: 8) {
                    if let when = model.lastUpdateCheck {
                        Text("Last checked \(when.formatted(date: .omitted, time: .shortened))")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                    Spacer()
                    CheckForUpdatesButton()
                }
                .padding(.top, 2)
            }
        }
    }

    private func openReleaseNotes() {
        guard let update = model.appUpdate, let url = URL(string: update.url) else { return }
        NSWorkspace.shared.open(url)
    }

    private func updateEngine() {
        Task { await model.applyEngineUpdateNow() }
    }
}

/// Progressive disclosure — LM Studio's most-copied idea, and the thing that decides whether
/// this app is usable by anyone who didn't write it.
struct DetailLevelCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "How much to show") {
            Picker("", selection: $model.detail) {
                ForEach(Detail.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(model.detail.blurb).font(.callout).foregroundStyle(.secondary)
        }
    }
}

struct MachineCard: View {
    let report: DoctorReport

    private var runtimeDescription: String {
        ["mlx", "mlx_lm", "mlx_vlm"].compactMap { name -> String? in
            guard let version = report.environment.packages[name] ?? nil else { return nil }
            return "\(name) \(version)"
        }
        .joined(separator: " · ")
    }

    var body: some View {
        Card(title: "This Mac",
             subtitle: report.ok ? "Everything checks out." : "Some things need attention.") {
            VStack(alignment: .leading, spacing: 7) {
                SettingsInfoRow(label: "Chip", value: report.environment.device ?? report.environment.machine)
                if let ram = report.environment.ramGB {
                    SettingsInfoRow(label: "Memory", value: String(format: "%.0f GB", ram))
                }
                if let chip = report.environment.chip, let spec = chip.bandwidthGBs {
                    // The number that governs decode speed on a Mac — spec sheet next to
                    // what a microbench actually achieves here (~80–90% of spec is normal).
                    let measured = chip.bandwidthMeasuredGBs
                        .map { String(format: " · %.0f GB/s measured", $0) } ?? ""
                    SettingsInfoRow(label: "Bandwidth",
                                    value: String(format: "%.0f GB/s spec", spec) + measured)
                }
                if let pressure = report.environment.memory?.pressure, pressure != "unknown" {
                    SettingsInfoRow(label: "Pressure",
                                    value: pressure == "normal" ? "normal"
                                        : pressure.uppercased()
                                            + " — generation runs slower until it clears")
                }
                SettingsInfoRow(label: "macOS", value: report.environment.osVersion ?? "—")
                SettingsInfoRow(label: "Engine", value: report.environment.version)
                SettingsInfoRow(label: "Runtime", value: runtimeDescription)

                ForEach(report.problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout).foregroundStyle(.orange)
                }

                // Letting macOS page weights out mid-generation is the classic silent
                // slowdown, so the fix is offered as a copyable command rather than advice.
                if let hint = report.environment.wiredLimitHint {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Large models run faster if the GPU can keep them resident:")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Text(hint).font(.caption.monospaced()).textSelection(.enabled)
                            Button("Copy", action: { copy(hint) })
                                .buttonStyle(.link).font(.caption)
                        }
                    }
                    .padding(.top, 4)
                }
            }
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// Extra folders the engine searches for MLX checkpoints before downloading anything — an
/// external drive, ~/models, a NAS. Reaches the engine as `MLX_DSPARK_MODEL_DIRS`, which a GUI
/// app is the only practical way to set. Needs engine ≥ 0.17.1 (older engines ignore it).
struct ModelFoldersCard: View {
    @EnvironmentObject private var model: AppModel
    @State private var restarting = false

    var body: some View {
        Card(title: "Model folders",
             subtitle: "Where you already keep MLX models. Searched before anything is "
                     + "downloaded; they appear under Models → On this Mac.") {
            VStack(alignment: .leading, spacing: 8) {
                if model.modelDirs.isEmpty {
                    Text("None yet — models load from the app's cache, the Hugging Face cache "
                         + "and LM Studio's folder.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(model.modelDirs, id: \.self) { dir in
                    HStack(spacing: 8) {
                        Image(systemName: "folder").foregroundStyle(.secondary)
                        Text(dir).font(.callout.monospaced()).lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button(action: { removeFolder(dir) }) {
                            Image(systemName: "minus.circle").imageScale(.small)
                        }
                        .buttonStyle(.borderless)
                        .help("Stop searching this folder")
                    }
                }
                HStack(spacing: 10) {
                    Button("Add folder…", action: addFolder)
                        .controlSize(.small)
                    Text("Layouts: publisher/model, publisher_model or a bare model folder "
                         + "(config.json + .safetensors).")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                HStack(spacing: 8) {
                    Text("Changes apply when the engine restarts.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(restarting ? "Restarting…" : "Restart engine now",
                           action: restartEngine)
                        .buttonStyle(.link).font(.caption)
                        .disabled(restarting || model.isModelLoading || model.isModelDownloading)
                }
            }
        }
    }

    private func removeFolder(_ directory: String) {
        model.modelDirs.removeAll { $0 == directory }
    }

    private func restartEngine() {
        restarting = true
        Task {
            await model.restartEngine()
            restarting = false
        }
    }

    private func addFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.message = "Choose a folder that holds MLX model directories"
        panel.prompt = "Add"
        guard panel.runModal() == .OK else { return }
        for url in panel.urls {
            let path = url.path
            if !model.modelDirs.contains(path) { model.modelDirs.append(path) }
        }
    }
}

private struct SettingsInfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            Text(value).font(.callout).textSelection(.enabled)
            Spacer()
        }
    }
}
