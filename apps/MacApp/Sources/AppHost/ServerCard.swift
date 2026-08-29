import AppCore
import SwiftUI

struct ServerCard: View {
    @EnvironmentObject private var model: AppModel

    // String mirror, committed on submit/apply — a TextField(value:format:) with a live
    // clamp rewrites the text under the cursor (see the Race max-tokens field).
    @State private var portText: String =
        Defaults.enginePort == 0 ? "" : String(Defaults.enginePort)
    @State private var restarting = false
    @State private var serveOnLAN = Defaults.serveOnLAN
    @State private var apiKeyEnabled = Defaults.apiKeyEnabled
    @State private var apiKey = Defaults.apiKey
    @State private var idleTTLSeconds = Defaults.modelIdleTTLSeconds
    @State private var addresses = LocalNetwork.ipv4Addresses()

    /// Settings differ from what the running engine was started with.
    private var needsRestart: Bool {
        serveOnLAN != model.runningServeOnLAN
            || (apiKeyEnabled && !apiKey.trimmingCharacters(in: .whitespaces).isEmpty
                ? apiKey.trimmingCharacters(in: .whitespaces) : nil) != model.runningAPIKey
            || idleTTLSeconds != model.runningModelIdleTTLSeconds
    }

    var body: some View {
        Card(title: "Local server",
             subtitle: model.runningServeOnLAN
                 ? "OpenAI- and Anthropic-compatible, reachable from other devices on your network."
                 : "OpenAI- and Anthropic-compatible, on this machine only.") {
            VStack(alignment: .leading, spacing: 8) {
                if case .ready(let port, let health) = model.serverState {
                    endpoint("OpenAI", "http://127.0.0.1:\(port)/v1")
                    endpoint("Anthropic", "http://127.0.0.1:\(port)")
                    if model.runningServeOnLAN {
                        if addresses.isEmpty {
                            Text("On the network: no IPv4 address found (not connected?)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        ForEach(addresses, id: \.ip) { address in
                            endpoint("\(address.interface)", "http://\(address.ip):\(port)/v1")
                        }
                        if model.runningAPIKey == nil {
                            Label("Serving to the network without an API key — anyone on it can "
                                  + "use your Mac's GPU and read the server's admin endpoints.",
                                  systemImage: "exclamationmark.triangle.fill")
                                .font(.caption).foregroundStyle(Theme.warning)
                        }
                    }
                    if let window = health.contextWindow {
                        Text("Context window \(window) tokens")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Point Claude Code at it with:  mlx-dspark claude")
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                } else {
                    Text(model.statusLine).foregroundStyle(.secondary)
                }
                Divider()
                portRow
                Divider()
                poolRow
                Divider()
                networkRows
            }
        }
        .onAppear(perform: refreshAddresses)
    }

    /// LAN serving + API key (the app side of what `serve --host 0.0.0.0 --api-key` does).
    /// Both take effect on the next engine start; "Apply & restart engine" does it now.
    @ViewBuilder private var networkRows: some View {
        Toggle("Serve on the local network", isOn: $serveOnLAN)
            .onChange(of: serveOnLAN) { _, on in updateServeOnLAN(on) }
        Text("Binds to every interface (0.0.0.0) so phones, other Macs and agents on your "
             + "network can use this engine. The app itself keeps talking over loopback.")
            .font(.caption).foregroundStyle(.secondary)
        Toggle("Require an API key", isOn: $apiKeyEnabled)
            .onChange(of: apiKeyEnabled) { _, on in updateAPIKeyEnabled(on) }
        if apiKeyEnabled {
            HStack(spacing: 8) {
                TextField("API key", text: $apiKey)
                    .textFieldStyle(.roundedBorder).font(.callout.monospaced())
                    .onSubmit { saveAPIKey(apiKey) }
                    .onChange(of: apiKey) { _, value in saveAPIKey(value) }
                CopyButton(text: apiKey)
                Button("Generate", action: generateAPIKey)
                    .controlSize(.small)
            }
            Text("Clients send it as  Authorization: Bearer <key>  (or x-api-key). The app, "
                 + "the Race tab and the Agents launcher use it automatically.")
                .font(.caption).foregroundStyle(.secondary)
        }
        if serveOnLAN && !apiKeyEnabled {
            Label("Without a key, anyone on your network can use the engine.",
                  systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(Theme.warning)
        }
        if needsRestart {
            HStack(spacing: 8) {
                Text("Server settings apply when the engine restarts.")
                    .font(.caption).foregroundStyle(.secondary)
                Button(restarting ? "Restarting…" : "Apply & restart engine",
                       action: restartEngine)
                    .font(.caption).disabled(restarting || model.isModelLoading
                                              || model.isModelDownloading)
            }
        }
    }

    /// Fixed engine port (issue #16): external clients (Claude Code, Open WebUI, …) need a
    /// base URL that survives an app restart; the automatic port changes on every launch.
    @ViewBuilder private var portRow: some View {
        HStack(spacing: 8) {
            Text("Port").font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            TextField("automatic", text: $portText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 92)
                .onSubmit { commitPort() }
            Button(restarting ? "Restarting…" : "Apply & restart engine",
                   action: restartEngine)
                .font(.caption)
                .disabled(restarting || model.isModelLoading || model.isModelDownloading)
            Spacer()
        }
        Text("A fixed port keeps external clients' base URL stable across launches "
             + "(default \(Defaults.defaultEnginePort); if something else holds it, the "
             + "app falls back to an automatic port). Blank = always automatic. "
             + "Ports 1024–65535.")
            .font(.caption).foregroundStyle(.secondary)
    }

    @ViewBuilder private var poolRow: some View {
        Picker("Idle model unload", selection: $idleTTLSeconds) {
            Text("Off").tag(0)
            Text("15 minutes").tag(900)
            Text("60 minutes").tag(3600)
        }
        .onChange(of: idleTTLSeconds) { _, value in updateIdleTTL(value) }
        Text("Unpinned, idle models leave the two-slot pool after this delay. Pins keep their "
             + "weights resident; prefix caches may still be cleared under memory pressure.")
            .font(.caption).foregroundStyle(.secondary)
    }

    private func refreshAddresses() {
        addresses = LocalNetwork.ipv4Addresses()
    }

    private func updateServeOnLAN(_ on: Bool) {
        Defaults.serveOnLAN = on
        if on { refreshAddresses() }
    }

    private func updateAPIKeyEnabled(_ on: Bool) {
        Defaults.apiKeyEnabled = on
        if on, apiKey.trimmingCharacters(in: .whitespaces).isEmpty {
            generateAPIKey()
        }
    }

    private func saveAPIKey(_ value: String) {
        Defaults.apiKey = value
    }

    private func generateAPIKey() {
        apiKey = LocalNetwork.generateAPIKey()
        Defaults.apiKey = apiKey
    }

    private func updateIdleTTL(_ value: Int) {
        Defaults.modelIdleTTLSeconds = value
    }

    private func restartEngine() {
        commitPort()
        restarting = true
        Task {
            await model.restartEngine()
            restarting = false
        }
    }

    private func commitPort() {
        let value = Int(portText.trimmingCharacters(in: .whitespaces)) ?? 0
        Defaults.enginePort = (1024...65535).contains(value) ? value : 0
        portText = Defaults.enginePort == 0 ? "" : String(Defaults.enginePort)
    }

    private func endpoint(_ label: String, _ url: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.callout).foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            Text(url).font(.callout.monospaced()).textSelection(.enabled)
            Button("Copy", action: { copy(url) })
                .buttonStyle(.link).font(.caption)
            Spacer()
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
