import Foundation

// MARK: - Wire types
//
// Only the fields the app actually reads are decoded. The engine adds an `x_mlx_dspark` block
// to every response — that non-standard block is where the whole point of this project lives
// (accept length, cap, lookup rounds), so it is a first-class type here, not an afterthought.

/// One primary model in the on-demand resident pool. Pins and residency are session-only; the
/// Mac app remains the durable source for each model's loading profile.
public struct PoolModelStatus: Decodable, Sendable, Equatable, Identifiable {
    public var id: String { model }
    public let model: String
    public let state: String
    public let ready: Bool
    public let pinned: Bool
    public let leases: Int
    public let error: String?
    public let restoreError: String?
    public let profilePendingReload: Bool
    public let evictionReason: String?
    public let warning: String?
    public let mode: String?
    public let target: String?
    public let drafter: String?
    public let maxDraft: String?
    public let contextWindow: Int?
    public let maxOutputTokens: Int?
    public let supportsReasoningEffort: Bool?
    public let confidenceThreshold: Double?
    public let raceArmConfidence: Bool?
    public let lookupDrafts: Bool?
    public let kvBits: Int?
    public let cpuSplit: HealthInfo.CPUSplitInfo?
    public let thinkingDefault: String?

    enum CodingKeys: String, CodingKey {
        case model, state, ready, pinned, leases, error, warning, mode, target, drafter
        case restoreError = "restore_error"
        case profilePendingReload = "profile_pending_reload"
        case evictionReason = "eviction_reason"
        case maxDraft = "max_draft"
        case contextWindow = "context_window"
        case maxOutputTokens = "max_output_tokens"
        case supportsReasoningEffort = "supports_reasoning_effort"
        case confidenceThreshold = "confidence_threshold"
        case raceArmConfidence = "race_arm_confidence"
        case lookupDrafts = "lookup_drafts"
        case kvBits = "kv_bits"
        case cpuSplit = "cpu_split"
        case thinkingDefault = "thinking_default"
    }
}

public struct ModelPoolInfo: Decodable, Sendable, Equatable {
    public let ready: Bool
    public let loading: Bool
    public let model: String?
    public let error: String?
    public let maxResidentModels: Int
    public let idleTTLSeconds: Double
    public let models: [PoolModelStatus]

    enum CodingKeys: String, CodingKey {
        case ready, loading, model, error, models
        case maxResidentModels = "max_resident_models"
        case idleTTLSeconds = "idle_ttl_seconds"
    }
}

public struct HealthInfo: Decodable, Sendable, Equatable {
    public struct CPUSplitInfo: Decodable, Sendable, Equatable {
        public let minRows: Int

        enum CodingKeys: String, CodingKey { case minRows = "min_rows" }
    }

    /// `ok` (model loaded) · `loading` (swap in flight) · `no_model` (server up, nothing
    /// loaded — the fast-launch/unloaded state). Older engines only ever report `ok` here;
    /// their loading state was undecodable and surfaced as a failed request instead.
    public let status: String
    /// Short display id, e.g. `Qwen3-4B-8bit`. Also the id `/v1/models` lists.
    /// `nil` while loading or with no model loaded.
    public let model: String?
    /// Resolved mode — `auto` is decided server-side, so this may differ from what was asked.
    public let mode: String?
    /// The full target repo and the drafter that auto-resolved for it. Showing the *pair* is
    /// this app's domain language; no other local-LLM app has a second model to name.
    public let target: String?
    public let drafter: String?
    /// Configured draft cap — `"auto"` or the pinned/derived value as a string.
    public let maxDraft: String?
    public let contextWindow: Int?
    public let maxOutputTokens: Int?
    /// Whether the loaded model's chat template reads `reasoning_effort` (Qwen3.8-class).
    /// Optional so the app keeps decoding health from older engines that don't report it.
    public let supportsReasoningEffort: Bool?
    /// Confidence-head early-stop threshold (0 = off). Part of a pair's measured-best
    /// bundle where the verify curve still rises inside the cap window AND the drafter
    /// leaves acceptance headroom (e.g. the Qwen3.6-35B-A3B MoE). Optional: older engines
    /// don't report it.
    public let confidenceThreshold: Double?
    /// Whether `/admin/race` arms accept a per-arm `confidence` — the capability gate for
    /// the Lab's cap+conf bundle chip. An engine without it would silently drop the field
    /// and the lane label would lie, so the chip only shows when this is true.
    public let raceArmConfidence: Bool?
    /// While `status == "loading"`: which stage the load is in — `"loading"` (fetching /
    /// loading weights) or `"warming_up"` (a short throwaway generation after the weights are
    /// resident, so the first real request is warm). Lets the loading screen say "Warming up…"
    /// instead of a bar that looks stuck. Nil when not loading / on older engines.
    public let phase: String?
    /// While `status == "loading"` and the engine is fetching weights: live download
    /// progress (`/health.download`). Nil once the fetch is done, on hot swaps of cached
    /// models, and on older engines.
    public let download: DownloadProgress?
    /// Whether hybrid n-gram lookup drafts are on for this load — the per-pair measured
    /// default the registry stamps (off for every MoE and the 4-bit 27B hybrids), settable
    /// per swap. Optional: older engines don't report it, and the toggle hides then.
    public let lookupDrafts: Bool?
    /// KV-cache quantization for the loaded target: 0 = full precision, 4/8 = quantized
    /// (issue #17). Optional: engines below 0.13.1 don't report it — and they also lack the
    /// `/admin/load` override, so the picker hides then (a control that silently does
    /// nothing is worse than none).
    public let kvBits: Int?
    /// Non-nil when experimental CPU co-prefill is active for this load.
    public let cpuSplit: CPUSplitInfo?
    /// Banner-worthy conditions: live macOS memory pressure and the engine's load-time notes
    /// (e.g. "this context window's KV cache can't fit alongside the weights"). Optional:
    /// older engines don't report it.
    public let warnings: [EngineWarning]?
    /// What API requests that don't specify thinking get: `"on"` (the model's default) or
    /// `"off"` (`--no-thinking` / the `/admin/load` override). Issue #19: clients like DSH and
    /// WorkBuddy have no reasoning toggle for local models, so this server default is what
    /// they live with. Optional: presence = the override exists on this engine.
    public let thinkingDefault: String?
    /// Present only for the opt-in on-demand server. It makes every resident model's state
    /// visible without pretending a two-model pool has an implicit current model.
    public let pool: ModelPoolInfo?

    enum CodingKeys: String, CodingKey {
        case status, model, mode, target, drafter, download, phase, warnings, pool
        case thinkingDefault = "thinking_default"
        case maxDraft = "max_draft"
        case contextWindow = "context_window"
        case maxOutputTokens = "max_output_tokens"
        case supportsReasoningEffort = "supports_reasoning_effort"
        case confidenceThreshold = "confidence_threshold"
        case raceArmConfidence = "race_arm_confidence"
        case lookupDrafts = "lookup_drafts"
        case kvBits = "kv_bits"
        case cpuSplit = "cpu_split"
    }

    /// True when a model is loaded and serving (`status == "ok"`).
    public var isLoaded: Bool { status == "ok" }
}

public extension HealthInfo {
    /// Rehydrate the selected model's health from its pool slot. The aggregate pool health has
    /// no implicit current model when several slots are resident.
    init(poolStatus: PoolModelStatus) {
        status = poolStatus.ready ? "ok" : poolStatus.state
        model = poolStatus.model
        mode = poolStatus.mode
        target = poolStatus.target
        drafter = poolStatus.drafter
        maxDraft = poolStatus.maxDraft
        contextWindow = poolStatus.contextWindow
        maxOutputTokens = poolStatus.maxOutputTokens
        supportsReasoningEffort = poolStatus.supportsReasoningEffort
        confidenceThreshold = poolStatus.confidenceThreshold
        raceArmConfidence = poolStatus.raceArmConfidence
        phase = nil
        download = nil
        lookupDrafts = poolStatus.lookupDrafts
        kvBits = poolStatus.kvBits
        cpuSplit = poolStatus.cpuSplit
        warnings = nil
        thinkingDefault = poolStatus.thinkingDefault
        pool = nil
    }
}

public extension HealthInfo {
    /// A pool deliberately has no implicit current model once two slots are resident. Prefer
    /// the app's selected target, then fall back to any serving slot for status UI.
    func readyPoolModel(preferred target: String) -> PoolModelStatus? {
        guard let pool else { return nil }
        return pool.models.first {
            $0.ready && ($0.model == target || $0.target == target)
        } ?? pool.models.first(where: \.ready)
    }

    /// Unlike the top-level `model`, this remains true when more than one pool slot is ready.
    var hasLoadedModel: Bool {
        pool?.models.contains(where: \.ready) ?? isLoaded
    }
}

/// A first-time model download in flight, as `/health` reports it while loading.
/// `bytesTotal` is best-effort (hub metadata) and may be nil for the first seconds.
public struct DownloadProgress: Decodable, Sendable, Equatable {
    public let repo: String
    public let bytesDone: Int64
    public let bytesTotal: Int64?

    enum CodingKeys: String, CodingKey {
        case repo
        case bytesDone = "bytes_done"
        case bytesTotal = "bytes_total"
    }

    public var fraction: Double? {
        guard let total = bytesTotal, total > 0 else { return nil }
        return min(1.0, Double(bytesDone) / Double(total))
    }
}

public struct SpecInfo: Codable, Sendable, Equatable {
    public let mode: String
    public let acceptLen: Double
    public let tokensPerSec: Double                  // end-to-end (prefill + decode)
    public let decodeTokensPerSec: Double?           // decode-only; nil on older engines that omit it
    public let targetForwards: Int
    public let cap: Int?
    public let lookupRounds: Int?
    // Per-request timing tiles (engines ≥ the roofline batch; nil before, and in sessions
    // saved by older builds — all optional so nothing stored ever fails to decode).
    public let promptTokens: Int?
    /// Prompt tokens the prefix cache served — why turn 2 is fast and turn 1 isn't.
    public let cachedTokens: Int?
    public let completionTokens: Int?
    public let contextTokens: Int?
    public let prefillSeconds: Double?
    public let decodeSeconds: Double?
    /// Time to the first streamed token, at the engine.
    public let ttftSeconds: Double?
    public let prefillTokensPerSec: Double?
    /// Late/early decode rate within the request (< 0.85 = it slowed as it went).
    public let decayRatio: Double?
    public let swapDeltaBytes: Int?
    public let cold: Bool?
    /// This Mac's single-stream roofline at this request's context depth …
    public let ceilingTokensPerSec: Double?
    /// … and decode ÷ that. > 1 is speculative decoding beating physics' one-token-per-
    /// weight-read limit; the number the whole project is about.
    public let rooflineRatio: Double?

    /// The rate to show the user: decode-only when the engine reports it (the honest,
    /// comparable number other local runtimes show), else the end-to-end value from
    /// older engines. Always >= `tokensPerSec`.
    public var displayTokensPerSec: Double { decodeTokensPerSec ?? tokensPerSec }

    enum CodingKeys: String, CodingKey {
        case mode, cap, cold
        case acceptLen = "accept_len"
        case tokensPerSec = "tokens_per_sec"
        case decodeTokensPerSec = "decode_tokens_per_sec"
        case targetForwards = "target_forwards"
        case lookupRounds = "lookup_rounds"
        case promptTokens = "prompt_tokens"
        case cachedTokens = "cached_tokens"
        case completionTokens = "completion_tokens"
        case contextTokens = "context_tokens"
        case prefillSeconds = "prefill_seconds"
        case decodeSeconds = "decode_seconds"
        case ttftSeconds = "ttft_seconds"
        case prefillTokensPerSec = "prefill_tokens_per_sec"
        case decayRatio = "decay_ratio"
        case swapDeltaBytes = "swap_delta_bytes"
        case ceilingTokensPerSec = "ceiling_tokens_per_sec"
        case rooflineRatio = "roofline_ratio"
    }
}

public struct ModelEntry: Decodable, Sendable, Identifiable {
    public let id: String
}

struct ModelList: Decodable { let data: [ModelEntry] }

/// Result of a hot model swap (`/admin/load`, `/admin/status`).
public struct LoadStatus: Decodable, Sendable {
    public let ready: Bool
    public let loading: Bool
    public let model: String?
    public let error: String?
    public let models: [PoolModelStatus]?
}

/// One event from a streaming completion.
public enum ChatEvent: Sendable {
    case delta(String)
    /// Reasoning streamed as a separate channel (`reasoning_content` — Muse-class models,
    /// whose thinking never appears inline in the content).
    case reasoning(String)
    /// Arrives on the last chunk, carrying the speculative-decoding stats for the turn.
    case finished(SpecInfo?)
}

public enum APIError: LocalizedError {
    case badStatus(Int, String)
    case notReady

    public var errorDescription: String? {
        switch self {
        case .badStatus(let code, let body):
            return "Server returned \(code): \(body)"
        case .notReady:
            return "The engine is not running."
        }
    }
}

// MARK: - APIClient

/// Talks to the local engine over its OpenAI-compatible API.
public struct APIClient: Sendable {
    public let baseURL: URL
    public let apiKey: String?
    let session: URLSession   // internal: shared with APIClient extensions in other files

    public init(baseURL: URL, apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        let config = URLSessionConfiguration.ephemeral
        // Generation can legitimately go quiet for a while on a big prompt; the default 60 s
        // resource timeout would kill a long agent-style request mid-stream.
        config.timeoutIntervalForRequest = 1800
        config.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: config)
    }

    func request(_ path: String, method: String = "GET", body: Data? = nil) -> URLRequest {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        if let apiKey, !apiKey.isEmpty {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return req
    }

    func modelRequest(_ path: String, model: String?) -> URLRequest {
        var req = request(path)
        guard let model, !model.isEmpty,
              var parts = URLComponents(url: req.url!, resolvingAgainstBaseURL: false)
        else { return req }
        parts.queryItems = [URLQueryItem(name: "model", value: model)]
        req.url = parts.url
        return req
    }

    public func health() async throws -> HealthInfo {
        var req = request("health")
        req.timeoutInterval = 2                      // used as a readiness poll; fail fast
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
        return try JSONDecoder().decode(HealthInfo.self, from: data)
    }

    public func models() async throws -> [ModelEntry] {
        let (data, response) = try await session.data(for: request("v1/models"))
        try Self.check(response, data)
        return try JSONDecoder().decode(ModelList.self, from: data).data
    }

    /// Raw `/metrics` JSON. Left untyped for now — the shape grows with the engine, and the
    /// Lab screens will decode the parts they need.
    public func metrics(model: String? = nil) async throws -> [String: Any] {
        let (data, response) = try await session.data(for: modelRequest("metrics", model: model))
        try Self.check(response, data)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    /// What the loaded model holds resident right now — the MLX allocator state that
    /// `/metrics` reports alongside the throughput stats.
    public func engineMemory(model: String? = nil) async throws -> EngineMemory? {
        struct Envelope: Decodable { let memory: EngineMemory? }
        let (data, response) = try await session.data(for: modelRequest("metrics", model: model))
        try Self.check(response, data)
        return try JSONDecoder().decode(Envelope.self, from: data).memory
    }

    /// Hot-swap the loaded model, keeping the server and its port. Returns once the new model
    /// is loaded (or throws with the reason). Far preferable to a process restart: the port
    /// survives, so nothing pointed at the server has to be reconfigured.
    ///
    /// `mode` (auto/dspark/dflash/lookup/baseline) and `maxDraft` ("auto" or a cap) override
    /// the server's startup flags for this load only — how the app turns speculation on/off
    /// or pins a measured cap without touching the model.
    public func loadModel(_ target: String, mode: String? = nil,
                          maxDraft: String? = nil,
                          confidence: Double? = nil,
                          contextWindow: Int? = nil,
                          lookupDrafts: Bool? = nil,
                          kvBits: Int? = nil,
                          cpuPrefill: Bool? = nil,
                          enableThinking: Bool? = nil,
                          keepLoaded: Bool = true,
                          reload: Bool = false) async throws -> LoadStatus {
        var payload: [String: Any] = ["model": target, "keep_loaded": keepLoaded,
                                      "reload": reload]
        if let mode { payload["mode"] = mode }
        if let maxDraft {
            payload["max_draft"] = Int(maxDraft).map { $0 as Any } ?? maxDraft
        }
        // 0 = explicitly off; nil = keep the server default. Older engines ignore the key.
        if let confidence { payload["confidence_threshold"] = confidence }
        // A limit below the model's own max — the KV-cache RAM lever. nil = model max.
        if let contextWindow { payload["context_window"] = contextWindow }
        // nil = keep this pair's measured default; a bool overrides it for this load.
        if let lookupDrafts { payload["lookup_drafts"] = lookupDrafts }
        // KV-cache quantization (issue #17): 0 = explicitly full precision, 4/8 = quantized;
        // nil = keep the server's setting. The caller gates on /health.kv_bits presence.
        if let kvBits { payload["kv_bits"] = kvBits }
        // App choice is deliberately only off/automatic; fixed fractions remain a CLI knob.
        if let cpuPrefill {
            payload["cpu_split"] = cpuPrefill ? "auto" : 0
        }
        // Thinking default for API clients (issue #19): false = off unless a request asks,
        // true = the model's own default; nil = keep. Sticky across later swaps server-side.
        if let enableThinking { payload["enable_thinking"] = enableThinking }
        let body = try JSONSerialization.data(withJSONObject: payload)
        var req = request("admin/load", method: "POST", body: body)
        req.timeoutInterval = 1800        // a first-time load downloads weights
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
        return try JSONDecoder().decode(LoadStatus.self, from: data)
    }

    /// Download a target and its resolved drafter without loading either into memory.
    public func downloadModel(_ target: String) async throws {
        let body = try JSONSerialization.data(withJSONObject: ["model": target])
        var req = request("admin/download", method: "POST", body: body)
        req.timeoutInterval = 1800
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
    }

    /// Register the app's persistent settings as a session-only JIT load profile. This never
    /// loads weights: absent context/mode extras keep the server's safe defaults instead of
    /// leaking another resident model's options into this one.
    public func registerModelProfile(_ target: String, mode: String,
                                     maxDraft: String, confidence: Double,
                                     contextWindow: Int?, lookupDrafts: Bool?, kvBits: Int?,
                                     cpuPrefill: Bool?, enableThinking: Bool?) async throws {
        var profile: [String: Any] = ["mode": mode, "max_draft": Int(maxDraft) ?? maxDraft,
                                      "confidence_threshold": confidence]
        if let contextWindow { profile["context_window"] = contextWindow }
        if let lookupDrafts { profile["lookup_drafts"] = lookupDrafts }
        if let kvBits { profile["kv_bits"] = kvBits }
        if let cpuPrefill { profile["cpu_split"] = cpuPrefill ? "auto" : 0 }
        if let enableThinking { profile["enable_thinking"] = enableThinking }
        let payload: [String: Any] = ["profiles": [["model": target, "profile": profile]]]
        let body = try JSONSerialization.data(withJSONObject: payload)
        let req = request("admin/model-profiles", method: "PUT", body: body)
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
    }

    /// Cancel an in-flight first-time download (`/admin/load/cancel`). The blocked
    /// `/admin/load` then fails cleanly and the server stays up, model-less. `cleanup`
    /// also removes the partial files; the default keeps them so loading the same model
    /// again resumes where it stopped instead of restarting a multi-gigabyte fetch.
    @discardableResult
    public func cancelLoad(cleanup: Bool = false) async throws -> Bool {
        let body = try JSONSerialization.data(withJSONObject: ["cleanup": cleanup])
        let req = request("admin/load/cancel", method: "POST", body: body)
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return obj?["cancelled"] as? Bool ?? false
    }

    public func loadStatus() async throws -> LoadStatus {
        let (data, response) = try await session.data(for: request("admin/status"))
        try Self.check(response, data)
        return try JSONDecoder().decode(LoadStatus.self, from: data)
    }

    /// Release the loaded model without loading another (`/admin/unload`) — frees its memory;
    /// the server and its port stay up and `/admin/load` brings a model back.
    @discardableResult
    public func unloadModel(model: String? = nil, all: Bool = false) async throws -> LoadStatus {
        var payload: [String: Any] = ["all": all]
        if let model { payload["model"] = model }
        let body = try JSONSerialization.data(withJSONObject: payload)
        let req = request("admin/unload", method: "POST", body: body)
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
        return try JSONDecoder().decode(LoadStatus.self, from: data)
    }

    /// This machine's measured verify/drafter cost curves (Lab → Curves).
    public func calibration(model: String? = nil) async throws -> Calibration {
        let (data, response) = try await session.data(for: modelRequest("calibration", model: model))
        try Self.check(response, data)
        return try JSONDecoder().decode(Calibration.self, from: data)
    }

    /// Environment + model inventory (onboarding, Models screen).
    public func doctor() async throws -> DoctorReport {
        let (data, response) = try await session.data(for: request("doctor"))
        try Self.check(response, data)
        return try JSONDecoder().decode(DoctorReport.self, from: data)
    }

    public func modelInventory() async throws -> ModelInventory {
        let (data, response) = try await session.data(for: request("admin/models"))
        try Self.check(response, data)
        return try JSONDecoder().decode(ModelInventory.self, from: data)
    }

    /// Recent rounds plus aggregates, without holding a stream open.
    public func rounds(limit: Int = 128, model: String? = nil) async throws -> (rounds: [RoundEvent], stats: RoundStats) {
        var req = modelRequest("rounds", model: model)
        var parts = URLComponents(url: req.url!, resolvingAgainstBaseURL: false)!
        parts.queryItems = (parts.queryItems ?? []) + [URLQueryItem(name: "limit", value: String(limit))]
        req.url = parts.url
        let (data, response) = try await session.data(for: req)
        try Self.check(response, data)
        struct Payload: Decodable { let rounds: [RoundEvent]; let stats: RoundStats }
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        return (payload.rounds, payload.stats)
    }

    /// Live per-round telemetry.
    ///
    /// Not scoped to a request: this reports every round the engine runs, whoever asked for
    /// it. That is what lets the app show a live accept ribbon while a *different* client —
    /// Claude Code, say — is the one generating.
    public func streamRounds(model: String? = nil) -> AsyncThrowingStream<TelemetryEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: modelRequest("events", model: model))
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        throw APIError.badStatus(http.statusCode, "")
                    }
                    let decoder = JSONDecoder()
                    var eventName = "round"
                    for try await line in bytes.lines {
                        if line.hasPrefix(":") { continue }            // keep-alive comment
                        if line.hasPrefix("event: ") {
                            eventName = String(line.dropFirst(7))
                            continue
                        }
                        guard line.hasPrefix("data: "),
                              let data = String(line.dropFirst(6)).data(using: .utf8)
                        else { continue }
                        switch eventName {
                        case "stats":
                            if let stats = try? decoder.decode(RoundStats.self, from: data) {
                                continuation.yield(.stats(stats))
                            }
                        case "prefill":
                            if let progress = try? decoder.decode(PrefillEvent.self, from: data) {
                                continuation.yield(.prefill(progress))
                            }
                        default:
                            if let round = try? decoder.decode(RoundEvent.self, from: data) {
                                continuation.yield(.round(round))
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Stream a chat completion, yielding text deltas and finally the spec-decode stats.
    public func streamChat(
        model: String,
        messages: [[String: String]],
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        topP: Double? = nil,
        seed: Int? = nil,
        stop: [String]? = nil
    ) -> AsyncThrowingStream<ChatEvent, Error> {
        var payload: [String: Any] = [
            "model": model,
            "messages": messages,
            "stream": true,
        ]
        if let temperature { payload["temperature"] = temperature }
        if let maxTokens { payload["max_tokens"] = maxTokens }
        if let enableThinking { payload["enable_thinking"] = enableThinking }
        if let reasoningEffort { payload["reasoning_effort"] = reasoningEffort }
        if let topP { payload["top_p"] = topP }
        if let seed { payload["seed"] = seed }
        if let stop, !stop.isEmpty { payload["stop"] = stop }

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let body = try JSONSerialization.data(withJSONObject: payload)
                    var req = request("v1/chat/completions", method: "POST", body: body)
                    req.timeoutInterval = 1800       // includes a first on-demand cold load
                    let (bytes, response) = try await session.bytes(for: req)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var detail = ""
                        for try await line in bytes.lines { detail += line }
                        throw APIError.badStatus(http.statusCode, detail)
                    }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))
                        if payload == "[DONE]" { break }
                        guard let data = payload.data(using: .utf8),
                              let obj = try? JSONSerialization.jsonObject(with: data)
                                  as? [String: Any] else { continue }

                        if let choices = obj["choices"] as? [[String: Any]],
                           let delta = choices.first?["delta"] as? [String: Any] {
                            if let text = delta["content"] as? String, !text.isEmpty {
                                continuation.yield(.delta(text))
                            }
                            if let thought = delta["reasoning_content"] as? String,
                               !thought.isEmpty {
                                continuation.yield(.reasoning(thought))
                            }
                        }
                        // The engine attaches its stats to the final chunk only.
                        if let specDict = obj["x_mlx_dspark"] {
                            let specData = try? JSONSerialization.data(withJSONObject: specDict)
                            let info = specData.flatMap {
                                try? JSONDecoder().decode(SpecInfo.self, from: $0)
                            }
                            continuation.yield(.finished(info))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard http.statusCode == 200 else {
            throw APIError.badStatus(http.statusCode, Self.errorMessage(from: data))
        }
    }

    /// The engine wraps errors as `{"error": {"message": …}}`; surface that message rather
    /// than a JSON blob — it's already written for humans (e.g. the drafter-registry hint).
    static func errorMessage(from data: Data) -> String {
        struct Wire: Decodable {
            struct Inner: Decodable { let message: String? }
            let error: Inner?
        }
        if let wire = try? JSONDecoder().decode(Wire.self, from: data),
           let message = wire.error?.message, !message.isEmpty {
            return message
        }
        return String(data: data, encoding: .utf8) ?? ""
    }
}

/// MLX allocator state from `/metrics` — what the loaded model actually holds resident.
public struct EngineMemory: Decodable, Sendable, Equatable {
    public let available: Bool
    public let activeBytes: Int?
    public let peakBytes: Int?
    public let cacheBytes: Int?

    enum CodingKeys: String, CodingKey {
        case available
        case activeBytes = "active_bytes"
        case peakBytes = "peak_bytes"
        case cacheBytes = "cache_bytes"
    }

    public var activeGB: Double? { activeBytes.map { Double($0) / 1_073_741_824 } }
    public var peakGB: Double? { peakBytes.map { Double($0) / 1_073_741_824 } }
}
