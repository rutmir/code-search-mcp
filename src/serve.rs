//! Minimal MCP server: line-delimited JSON-RPC over stdin/stdout.
//!
//! Transport: per the MCP spec, each message is a single JSON object on its
//! own line (no Content-Length headers — unlike LSP). Server reads requests
//! from stdin, writes responses to stdout, and ALL logging goes to stderr.
//! Writing anything else to stdout silently breaks the client.
//!
//! Scope:
//!   - `initialize` / `notifications/initialized` handshake
//!   - `tools/list` advertising `code_search` and `code_read_chunk`
//!   - `tools/call` dispatching to [`crate::search::SearchContext`]; each
//!     call runs in its own spawned task, tracked by request id
//!   - `notifications/cancelled` — preempts the matching in-flight search
//!     via a oneshot raced in `tokio::select!`
//!   - `notifications/progress` for calls that supply a progress token
//!   - `ping` as a health-check
//!   - background watcher task (when `[watcher].enabled`)
//!
//! Out of scope (deferred):
//!   - resources, prompts, logging notifications
//!   - tools/list_changed notifications
//!   - structured `outputSchema` on tool results

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::search::{SearchContext, SearchParams, SearchResult, SearchStage};
use crate::watcher;

/// Map from JSON-RPC stringified request id → cancel sender. A spawned
/// tools/call task races search vs. its cancel receiver in a select!,
/// so a send through the sender preempts the search and yields a clean
/// "cancelled" response.
type InFlight = Mutex<HashMap<String, oneshot::Sender<()>>>;

/// MCP protocol version this server implements, and the older revisions
/// it still speaks. On `initialize` we echo the client's version when
/// it's one we know; otherwise we answer with our own and let the client
/// decide whether to proceed (per the MCP handshake rules).
const PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const SERVER_NAME: &str = "code-search-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cap on the text `code_read_chunk` returns in one call. The tool exists
/// to save the caller a `Read`, not to become a way to pull a whole file
/// into context by accident.
const MAX_READ_CHUNK_CHARS: usize = 20_000;

/// How long shutdown waits for calls dispatched before stdin closed. Long
/// enough for anything already returning, short enough that a wedged
/// search can't keep the process alive after its client is gone.
const SHUTDOWN_DRAIN: std::time::Duration = std::time::Duration::from_secs(5);

/// Standard JSON-RPC 2.0 error codes. Listed in full so we can pick the
/// closest one when reporting protocol-level failures; `ERR_INTERNAL` is
/// kept around for future internal-error cases (e.g. JSON serialization
/// failure) even though the current dispatch table doesn't reach it.
const ERR_PARSE: i32 = -32700;
const ERR_INVALID_REQUEST: i32 = -32600;
const ERR_METHOD_NOT_FOUND: i32 = -32601;
const ERR_INVALID_PARAMS: i32 = -32602;
#[allow(dead_code)]
const ERR_INTERNAL: i32 = -32603;

/// Process-wide server state: the shared search context plus the
/// cancellation registry.
struct ServerState {
    config: Config,
    /// Built once and reused so every query keeps warm HTTP connection
    /// pools and an open tantivy reader. Lazily initialized with retry
    /// rather than required at startup: `serve` is spawned by the MCP
    /// client and must stay up (reporting tool-level errors) even when
    /// Qdrant isn't reachable yet, otherwise the tool vanishes from the
    /// session instead of explaining itself.
    ctx: tokio::sync::Mutex<Option<Arc<SearchContext>>>,
    in_flight: InFlight,
}

impl ServerState {
    fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            ctx: tokio::sync::Mutex::new(None),
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    async fn context(&self) -> Result<Arc<SearchContext>> {
        let mut guard = self.ctx.lock().await;
        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }
        let built = Arc::new(SearchContext::new(&self.config).await?);
        *guard = Some(Arc::clone(&built));
        Ok(built)
    }

    /// A poisoned lock means an earlier task panicked while holding it.
    /// The map is a registry of cancel channels, not an invariant a panic
    /// could have half-broken, so recovering beats refusing every
    /// subsequent call for the life of the process.
    fn in_flight(&self) -> MutexGuard<'_, HashMap<String, oneshot::Sender<()>>> {
        self.in_flight.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub async fn run(config: &Config) -> Result<()> {
    info!(
        server = SERVER_NAME,
        version = SERVER_VERSION,
        protocol = PROTOCOL_VERSION,
        project = %config.project.id,
        "MCP serve starting"
    );

    // Optionally co-host the file watcher inside the serve process so the
    // index stays fresh as the user edits files. Background task: errors
    // get logged but never bubble up — search-side reliability is the
    // primary concern, and a watcher hiccup shouldn't take down MCP for
    // the Claude Code session.
    //
    // Concurrency note: the watcher holds the tantivy IndexWriter (via
    // its embedded Bm25Index), while `search::run` opens Bm25Search
    // read-only per query. Tantivy permits N readers + 1 writer, so
    // they coexist without lock contention.
    let watcher_handle = if config.watcher.enabled {
        info!("watcher.enabled = true; spawning background watcher task");
        let watch_config = config.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = watcher::run(&watch_config).await {
                error!(error = ?e, "background watcher task ended with error");
            } else {
                info!("background watcher task ended cleanly");
            }
        }))
    } else {
        info!("watcher.enabled = false; serving against the static index");
        None
    };

    // Main loop in its own function so we can do unconditional cleanup
    // (abort the watcher) regardless of how serve_loop exits.
    let loop_result = serve_loop(config).await;

    if let Some(handle) = watcher_handle {
        info!("serve loop exited; aborting background watcher task");
        handle.abort();
        // Wait for the abort to land. JoinError on a cancelled task is
        // expected; we don't propagate it.
        let _ = handle.await;
    }

    loop_result
}

/// The MCP JSON-RPC stdio loop. Extracted so [`run`] can perform
/// unconditional cleanup of the spawned watcher task on any exit path.
///
/// Concurrency model:
///   - A dedicated reader task pumps stdin lines into an mpsc channel —
///     stdin lives in one place, no cancel-safety footguns on `read_line`.
///   - The main loop selects on (incoming line, outgoing response). It
///     spawns `tools/call` into its own task and tracks each by id; the
///     task races the search against a oneshot cancellation receiver and
///     sends its result back through the outgoing channel.
///   - `notifications/cancelled` looks the id up in the in-flight map
///     and fires the oneshot — the racing select! yields the "cancelled"
///     response within microseconds and the in-flight HTTP requests are
///     dropped (reqwest respects future cancellation).
///   - Lightweight requests (initialize / tools/list / ping) are handled
///     synchronously inline since they return in ~milliseconds.
async fn serve_loop(config: &Config) -> Result<()> {
    let state = Arc::new(ServerState::new(config));
    // Warm the shared context up front so the first real query doesn't pay
    // for connection setup and the marker check. Failure here is not fatal
    // — see ServerState::ctx.
    match state.context().await {
        Ok(_) => info!("search context ready"),
        Err(e) => warn!(
            error = %e,
            "search context not ready at startup; will retry on first query"
        ),
    }

    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<String>();
    let _reader_task = tokio::spawn(async move {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => return, // EOF
                Ok(_) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() && line_tx.send(trimmed).is_err() {
                        return; // receiver dropped
                    }
                }
                Err(e) => {
                    error!(error = %e, "stdin read error");
                    return;
                }
            }
        }
    });

    let mut stdout = tokio::io::stdout();
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel::<Value>();

    loop {
        tokio::select! {
            biased;
            // Drain pending responses first so wire ordering stays as close
            // to request order as practical.
            Some(response) = resp_rx.recv() => {
                send(&mut stdout, &response).await?;
            }
            maybe_line = line_rx.recv() => {
                let Some(trimmed) = maybe_line else {
                    info!("stdin closed; draining in-flight responses");
                    break;
                };
                debug!(line = %trimmed, "<- received");

                let msg: Value = match serde_json::from_str(&trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        error!(error = %e, line = %trimmed, "JSON parse error");
                        let r = error_response(
                            Value::Null,
                            ERR_PARSE,
                            &format!("parse error: {}", e),
                        );
                        send(&mut stdout, &r).await?;
                        continue;
                    }
                };

                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let is_notification = msg.get("id").is_none();

                if is_notification {
                    handle_notification(method, &msg, &state);
                    continue;
                }

                match method {
                    "initialize" => {
                        let r = ok_response(id, handle_initialize(&msg));
                        send(&mut stdout, &r).await?;
                    }
                    "tools/list" => {
                        let r = ok_response(id, handle_tools_list());
                        send(&mut stdout, &r).await?;
                    }
                    "tools/call" => {
                        spawn_tools_call(
                            Arc::clone(&state),
                            id,
                            msg,
                            resp_tx.clone(),
                        );
                    }
                    "ping" => {
                        let r = ok_response(id, json!({}));
                        send(&mut stdout, &r).await?;
                    }
                    "" => {
                        let r = error_response(
                            id,
                            ERR_INVALID_REQUEST,
                            "missing 'method' field",
                        );
                        send(&mut stdout, &r).await?;
                    }
                    other => {
                        let r = error_response(
                            id,
                            ERR_METHOD_NOT_FOUND,
                            &format!("method not found: {}", other),
                        );
                        send(&mut stdout, &r).await?;
                    }
                }
            }
        }
    }

    // Stdin is closed, but tasks spawned before that may still be running.
    // Their sender clones are the only ones left once ours is dropped, so
    // `recv` returns None exactly when the last one finishes — no polling,
    // no guessing. The client's read end is usually still open, and a
    // dropped response is indistinguishable from a hung tool.
    drop(resp_tx);
    let drain = tokio::time::timeout(SHUTDOWN_DRAIN, async {
        let mut sent = 0usize;
        while let Some(response) = resp_rx.recv().await {
            send(&mut stdout, &response).await?;
            sent += 1;
        }
        Ok::<usize, anyhow::Error>(sent)
    })
    .await;
    match drain {
        Ok(Ok(sent)) => info!(drained = sent, "shutdown complete"),
        Ok(Err(e)) => warn!(error = %e, "failed to write a response while draining"),
        Err(_) => warn!(
            timeout_s = SHUTDOWN_DRAIN.as_secs(),
            "gave up waiting for in-flight calls; exiting"
        ),
    }
    Ok(())
}

/// Spawn a `tools/call` task: registers a oneshot cancel sender keyed by
/// the request id, runs the tool, and sends the result (or a cancelled
/// marker) back through `resp_tx`.
fn spawn_tools_call(
    state: Arc<ServerState>,
    id: Value,
    msg: Value,
    resp_tx: mpsc::UnboundedSender<Value>,
) {
    let id_key = id_to_key(&id);
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    // If client somehow reuses an id (shouldn't happen — JSON-RPC ids are
    // unique per pending request), the previous cancel sender is dropped
    // silently. The previous task will see its receiver close and treat
    // it as cancelled — acceptable.
    state.in_flight().insert(id_key.clone(), cancel_tx);

    let state_clone = Arc::clone(&state);
    let id_clone = id.clone();
    let id_key_clone = id_key.clone();
    let progress_tx = resp_tx.clone();

    tokio::spawn(async move {
        let response: Value = tokio::select! {
            result = handle_tools_call(&state_clone, &msg, &progress_tx) => match result {
                Ok(r) => ok_response(id_clone, r),
                Err((code, m)) => error_response(id_clone, code, &m),
            },
            _ = cancel_rx => {
                info!(id = %id_key_clone, "tools/call cancelled by client");
                json!({
                    "jsonrpc": "2.0",
                    "id": id_clone,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "Search cancelled by client (notifications/cancelled)."
                        }],
                        "isError": true
                    }
                })
            }
        };
        // Clean up the in-flight entry. May already be gone if the cancel
        // notification was the one that ended the task (it removes too) —
        // remove is idempotent.
        state.in_flight().remove(&id_key_clone);
        let _ = resp_tx.send(response);
    });
}

fn handle_notification(method: &str, msg: &Value, state: &ServerState) {
    match method {
        "notifications/initialized" => {
            info!("client confirmed initialized");
        }
        "notifications/cancelled" => {
            let Some(req_id) = msg.pointer("/params/requestId") else {
                warn!("notifications/cancelled without params.requestId — ignoring");
                return;
            };
            let id_key = id_to_key(req_id);
            let removed = state.in_flight().remove(&id_key);
            match removed {
                Some(cancel_tx) => {
                    let _ = cancel_tx.send(());
                    info!(id = %id_key, "cancellation forwarded to in-flight task");
                }
                None => {
                    debug!(
                        id = %id_key,
                        "cancellation for unknown / already-completed task"
                    );
                }
            }
        }
        other => {
            debug!(method = other, "unhandled notification");
        }
    }
}

/// Stable string key from a JSON-RPC id. Numbers and strings both stringify
/// consistently via `serde_json::to_string`; we use that so `requestId: 1`
/// in `notifications/cancelled` matches `id: 1` from the original call.
fn id_to_key(id: &Value) -> String {
    serde_json::to_string(id).unwrap_or_else(|_| "null".to_string())
}

fn handle_initialize(msg: &Value) -> Value {
    // Version negotiation: answer in the client's dialect when we speak
    // it, otherwise state our own and let the client decide. The wire
    // shape of everything this server implements is identical across the
    // supported revisions, so echoing back is honest.
    let requested = msg
        .pointer("/params/protocolVersion")
        .and_then(|v| v.as_str());
    let agreed = match requested {
        Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v,
        Some(other) => {
            warn!(
                requested = other,
                offering = PROTOCOL_VERSION,
                "client asked for an unsupported MCP protocol version"
            );
            PROTOCOL_VERSION
        }
        None => PROTOCOL_VERSION,
    };
    json!({
        "protocolVersion": agreed,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    })
}

fn handle_tools_list() -> Value {
    json!({
        "tools": [{
            "name": "code_search",
            "description": "PREFERRED first-line search over this project's indexed content. \
                            Use BEFORE Grep / Glob / iterative Read for any question about \
                            what's in the project — source code, markdown documentation, \
                            config files, READMEs, CHANGELOGs, ROADMAPs, design notes. \
                            \n\n\
                            Returns ranked file:start-end chunks with syntactic anchors \
                            (e.g. `fn AdaptiveBatcher::note_failure`, `struct Foo`, \
                            `method Class.method_name`, `## Section Title` for markdown) and \
                            a content preview, matched via hybrid BM25 (keyword) + dense \
                            embeddings (semantic) + cross-encoder reranking. \
                            Markdown is heading-aware so docs cut on section boundaries; \
                            code uses tree-sitter AST chunking per language. \
                            \n\n\
                            Use it for:\n\
                            - 'how does X work' / 'where is Y implemented' / 'what uses Z' \
                              (semantic code lookup)\n\
                            - 'show me the section about W in docs' / 'what does the \
                              ROADMAP say about phase N' (docs lookup)\n\
                            - 'find files that mention LIBOR' / 'where is config option \
                              K set' (cross-cutting term search)\n\
                            - Initial project orientation: pull the top-ranked passages \
                              for any natural-language question before falling back to \
                              direct file inspection.\n\
                            \n\
                            Typical cost: one tool call, ~1500 tokens of structured \
                            results. ~30-100× cheaper than iterative grep+read for \
                            exploration, with equal or better recall — favor it broadly. \
                            \n\n\
                            Fall back to direct file tools only when:\n\
                            - The target is a specific known file path you already have \
                              (`Read /a/b/c.rs:42-50`)\n\
                            - The content lives outside the index (build artifacts, \
                              generated files, paths in [index].exclude or gitignored)\n\
                            - You need exact-bytes operations (hex dump, log-file tail)\n\
                            \n\
                            Tip: concise queries (3-10 words) usually outperform long \
                            descriptions. The reranker sees more semantic signal in a \
                            tight phrase than in a paragraph. \
                            \n\n\
                            Tip: phrase the query in the language the target text is \
                            written in. Identifiers are English regardless, so code \
                            lookups are unaffected — but for prose, asking in the \
                            document's own language is markedly better than translating \
                            the question first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language phrase describing what you're looking \
                                        for. Examples: 'AIMD batch halving on failure', \
                                        'tantivy commit policy', 'project roadmap phase 2', \
                                        'how reranker fallback works'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (1-50). Default 10.",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    },
                    "lang": {
                        "type": "string",
                        "description": "Filter to a single language: 'rust', 'markdown', 'toml', \
                                        'python', etc. Use 'markdown' to scope to docs only. \
                                        Omit for all languages."
                    },
                    "path": {
                        "type": "string",
                        "description": "Filter to files whose path contains this substring. \
                                        Example: 'crates/bot' to scope to one crate, or \
                                        'docs/' to scope to documentation."
                    }
                },
                "required": ["query"]
            }
        }, {
            "name": "code_read_chunk",
            "description": "Fetch the FULL text of an indexed chunk you already located with \
                            code_search. Use it instead of Read when a code_search preview was \
                            cut off and you only need that one function / section — it answers \
                            from the index, so it costs no file I/O and returns exactly the \
                            chunk, not the surrounding file. \
                            \n\n\
                            Pass the `file` and the line range exactly as printed in the \
                            code_search header (`src/foo.rs:120-168` → file='src/foo.rs', \
                            start_line=120, end_line=168). Every indexed chunk overlapping that \
                            range is returned. Omit the range to get all chunks of the file. \
                            \n\n\
                            Fall back to Read when you need lines the index doesn't cover \
                            (gitignored or excluded files), or the file's exact bytes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Repo-relative path exactly as printed by code_search \
                                        (the part before ':')."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line of the range. Omit for the whole file.",
                        "minimum": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line of the range, inclusive. Omit for the whole file.",
                        "minimum": 1
                    }
                },
                "required": ["file"]
            }
        }]
    })
}

/// Handle `tools/call`. Returns Ok(result_value) for success (including
/// tool-level failures, which use `isError: true` inside the result) or
/// Err((code, msg)) for protocol-level errors (bad params, etc).
async fn handle_tools_call(
    state: &ServerState,
    msg: &Value,
    resp_tx: &mpsc::UnboundedSender<Value>,
) -> Result<Value, (i32, String)> {
    let name = msg
        .pointer("/params/name")
        .and_then(|v| v.as_str())
        .ok_or((ERR_INVALID_PARAMS, "missing 'params.name'".to_string()))?;

    let args = msg
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or(Value::Null);

    match name {
        "code_search" => {
            let progress_token = msg.pointer("/params/_meta/progressToken").cloned();
            handle_code_search(state, &args, progress_token, resp_tx).await
        }
        "code_read_chunk" => handle_code_read_chunk(state, &args).await,
        other => Err((ERR_METHOD_NOT_FOUND, format!("unknown tool: {}", other))),
    }
}

async fn handle_code_search(
    state: &ServerState,
    args: &Value,
    progress_token: Option<Value>,
    resp_tx: &mpsc::UnboundedSender<Value>,
) -> Result<Value, (i32, String)> {
    let config = &state.config;
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or((ERR_INVALID_PARAMS, "missing 'query' argument".to_string()))?;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .clamp(1, 50) as usize;
    let lang = args.get("lang").and_then(|v| v.as_str());
    let path = args.get("path").and_then(|v| v.as_str());

    info!(
        query = %query,
        limit,
        lang = ?lang,
        path = ?path,
        "tools/call code_search"
    );

    // A quality-first search can take a minute or more; without this the
    // client sees a silent stall and can't tell a slow search from a hung
    // server. Only emitted when the client opted in with a progress token.
    let sink = progress_token.map(|token| {
        let tx = resp_tx.clone();
        move |stage: SearchStage| {
            let _ = tx.send(json!({
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": token,
                    "progress": stage.step(),
                    "total": SearchStage::TOTAL,
                    "message": stage.label(),
                }
            }));
        }
    });

    let started = std::time::Instant::now();
    let result = match state.context().await {
        Ok(ctx) => {
            ctx.search(SearchParams {
                query,
                limit,
                // Always use rerank in serve mode (when configured). Skipping
                // it is a CLI debugging knob, not useful via MCP.
                use_rerank: true,
                lang,
                path,
                progress: sink
                    .as_ref()
                    .map(|f| f as &(dyn Fn(SearchStage) + Send + Sync)),
            })
            .await
        }
        Err(e) => Err(e),
    };
    let elapsed_ms = started.elapsed().as_millis() as u64;

    match result {
        Ok(results) => {
            info!(
                results = results.len(),
                elapsed_ms, "tools/call code_search completed"
            );
            if let Some(log_path) = &config.serve.query_log_path {
                append_query_log(log_path, query, lang, path, limit, elapsed_ms, &results);
            }
            let rerank_expected = config.reranker.as_ref().is_some_and(|r| r.enabled);
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format_results_for_llm(&results, query, elapsed_ms, rerank_expected)
                }],
                "isError": false
            }))
        }
        Err(e) => {
            // Tool-level failure: report as MCP "isError" rather than a
            // JSON-RPC error, so Claude sees the failure as a tool outcome
            // it can react to (try a different query) rather than a
            // protocol crash.
            warn!(error = %e, elapsed_ms, "tools/call code_search failed");
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!("Search failed: {:#}", e)
                }],
                "isError": true
            }))
        }
    }
}

async fn handle_code_read_chunk(state: &ServerState, args: &Value) -> Result<Value, (i32, String)> {
    let file = args
        .get("file")
        .and_then(|v| v.as_str())
        .ok_or((ERR_INVALID_PARAMS, "missing 'file' argument".to_string()))?;
    let start = args.get("start_line").and_then(|v| v.as_u64()).unwrap_or(1);
    let end = args
        .get("end_line")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX);
    if end < start {
        return Err((
            ERR_INVALID_PARAMS,
            format!("end_line ({}) is before start_line ({})", end, start),
        ));
    }

    info!(file = %file, start, "tools/call code_read_chunk");

    let ctx = match state.context().await {
        Ok(c) => c,
        Err(e) => {
            return Ok(tool_error(format!("Index unavailable: {:#}", e)));
        }
    };
    match ctx.read_chunks(file, start, end) {
        Ok(chunks) if chunks.is_empty() => Ok(tool_error(format!(
            "No indexed chunk for {}:{}-{}. The path must match a code_search result exactly \
             (repo-relative, not absolute); if the file isn't indexed, use Read.",
            file, start, end
        ))),
        Ok(chunks) => Ok(json!({
            "content": [{ "type": "text", "text": format_chunks_for_llm(&chunks) }],
            "isError": false
        })),
        Err(e) => {
            warn!(file = %file, error = %e, "tools/call code_read_chunk failed");
            Ok(tool_error(format!("Chunk read failed: {:#}", e)))
        }
    }
}

fn tool_error(text: String) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": true
    })
}

/// Render chunk texts verbatim — unlike search previews, whitespace is
/// preserved: the caller asked for this to read code, and reindented code
/// is worse than no code.
fn format_chunks_for_llm(chunks: &[crate::bm25::ChunkText]) -> String {
    let mut s = String::new();
    let mut budget = MAX_READ_CHUNK_CHARS;
    let mut truncated = 0usize;
    for c in chunks {
        let symbol = match (&c.kind, &c.name) {
            (Some(k), Some(n)) => format!("  {} {}", k, n),
            (Some(k), None) => format!("  {}", k),
            _ => String::new(),
        };
        let _ = writeln!(
            s,
            "{}:{}-{}  [{}]{}",
            c.file, c.start_line, c.end_line, c.lang, symbol
        );
        if budget == 0 {
            truncated += 1;
            let _ = writeln!(s, "(omitted — output limit reached)\n");
            continue;
        }
        let text: String = c.content.chars().take(budget).collect();
        budget -= text.chars().count();
        let _ = writeln!(s, "{}\n", text);
    }
    if budget == 0 {
        let omitted = if truncated > 0 {
            format!("; {} later chunk(s) omitted entirely", truncated)
        } else {
            String::new()
        };
        let _ = writeln!(
            s,
            "NOTE: output truncated at {} chars{}. Narrow the line range or use Read for the \
             full file.",
            MAX_READ_CHUNK_CHARS, omitted
        );
    }
    s
}

/// Append one JSON line describing a completed `code_search` call to the
/// query log. Durable observability: MCP clients don't persist server
/// stderr, so without this file there is no record of what was asked.
/// Best-effort — a failed write warns and never fails the search.
fn append_query_log(
    path: &std::path::Path,
    query: &str,
    lang: Option<&str>,
    path_filter: Option<&str>,
    limit: usize,
    elapsed_ms: u64,
    results: &[SearchResult],
) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let top: Vec<Value> = results
        .iter()
        .take(3)
        .map(|r| {
            json!({
                "file": r.file,
                "lines": format!("{}-{}", r.start_line, r.end_line),
                "score": r.score,
                "rerank": r.rerank_score,
            })
        })
        .collect();
    let entry = json!({
        "ts": ts,
        "query": query,
        "lang": lang,
        "path": path_filter,
        "limit": limit,
        "elapsed_ms": elapsed_ms,
        "results": results.len(),
        "reranked": results.iter().any(|r| r.rerank_score.is_some()),
        "top": top,
    });
    let line = format!("{}\n", entry);
    let write = || -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        f.write_all(line.as_bytes())
    };
    if let Err(e) = write() {
        warn!(path = %path.display(), error = %e, "query log append failed");
    }
}

/// Render search results as a structured text block the LLM can quote
/// from in subsequent reasoning. Lines stay short, paths are obvious,
/// scores are visible for sanity-checking.
fn format_results_for_llm(
    results: &[SearchResult],
    query: &str,
    elapsed_ms: u64,
    rerank_expected: bool,
) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Found {} result(s) for {:?} ({} ms):",
        results.len(),
        query,
        elapsed_ms
    );
    if results.is_empty() {
        let _ = writeln!(s, "(no matches — try a different query or remove filters)");
        return s;
    }
    // Rerank runs unconditionally in serve mode whenever a reranker is
    // configured, so a result set where no hit carries a rerank score
    // means the cross-encoder call failed and search fell back to
    // retrieval-only RRF ranking. Surface that to the LLM (and the human
    // reading the transcript) instead of degrading silently — the
    // ordering is noticeably weaker without the reranker. No warning when
    // the reranker is intentionally disabled in config.
    if rerank_expected && results.iter().all(|r| r.rerank_score.is_none()) {
        let _ = writeln!(
            s,
            "WARNING: reranker unavailable — results are RRF-ranked only (lower precision). \
             Treat the ordering as approximate."
        );
    }
    for (i, r) in results.iter().enumerate() {
        let _ = writeln!(s);
        // Header. If AST chunking gave us a syntactic anchor (e.g.
        // `fn Foo::bar`), put it up front so the LLM can quote / navigate
        // by symbol name without re-reading the preview.
        let symbol = match (&r.kind, &r.name) {
            (Some(k), Some(n)) => format!("  {} {}", k, n),
            (Some(k), None) => format!("  {}", k),
            _ => String::new(),
        };
        let _ = writeln!(
            s,
            "#{} {}:{}-{}  [{}]{}  score={:.4}",
            i + 1,
            r.file,
            r.start_line,
            r.end_line,
            r.lang,
            symbol,
            r.score
        );
        // Up to 600 chars of preview — enough to anchor the LLM on what's
        // in this chunk without flooding the context window for top-N=10.
        let preview: String = r
            .preview
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(600)
            .collect();
        let _ = writeln!(s, "    {}", preview);
    }
    s
}

fn ok_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn error_response(id: Value, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })
}

async fn send(stdout: &mut tokio::io::Stdout, msg: &Value) -> Result<()> {
    let line = serde_json::to_string(msg).context("serializing JSON-RPC response")?;
    debug!(line = %line, "-> sending");
    stdout
        .write_all(line.as_bytes())
        .await
        .context("write stdout")?;
    stdout.write_all(b"\n").await.context("write newline")?;
    stdout.flush().await.context("flush stdout")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_response_has_required_fields() {
        let v = handle_initialize(&Value::Null);
        assert_eq!(v["protocolVersion"], PROTOCOL_VERSION);
        assert!(v["capabilities"]["tools"].is_object());
        assert_eq!(v["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(v["serverInfo"]["version"], SERVER_VERSION);
    }

    #[test]
    fn initialize_echoes_a_supported_client_version() {
        // Claude Code still initializes with the 2024-11-05 revision; the
        // handshake must answer in the dialect the client asked for.
        for asked in SUPPORTED_PROTOCOL_VERSIONS {
            let msg = json!({ "params": { "protocolVersion": asked } });
            assert_eq!(handle_initialize(&msg)["protocolVersion"], *asked);
        }
    }

    #[test]
    fn initialize_falls_back_on_unknown_client_version() {
        let msg = json!({ "params": { "protocolVersion": "1999-01-01" } });
        assert_eq!(
            handle_initialize(&msg)["protocolVersion"],
            PROTOCOL_VERSION,
            "unknown client version must get our own, not an echo"
        );
    }

    #[test]
    fn tools_list_advertises_both_tools() {
        let v = handle_tools_list();
        let tools = v["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert_eq!(names, vec!["code_search", "code_read_chunk"]);

        let search_schema = &tools[0]["inputSchema"];
        assert_eq!(search_schema["type"], "object");
        // Required props must include query.
        let req = search_schema["required"].as_array().expect("required");
        assert!(req.iter().any(|v| v.as_str() == Some("query")));
        // limit / lang / path are advertised as optional.
        let props = search_schema["properties"].as_object().expect("props");
        for k in ["query", "limit", "lang", "path"] {
            assert!(props.contains_key(k), "missing schema for {}", k);
        }

        let read_schema = &tools[1]["inputSchema"];
        let req = read_schema["required"].as_array().expect("required");
        assert!(req.iter().any(|v| v.as_str() == Some("file")));
        let props = read_schema["properties"].as_object().expect("props");
        for k in ["file", "start_line", "end_line"] {
            assert!(props.contains_key(k), "missing schema for {}", k);
        }
    }

    fn chunk(start: u64, end: u64, content: &str) -> crate::bm25::ChunkText {
        crate::bm25::ChunkText {
            file: "src/foo.rs".to_string(),
            start_line: start,
            end_line: end,
            lang: "rust".to_string(),
            kind: Some("fn".to_string()),
            name: Some("bar".to_string()),
            content: content.to_string(),
        }
    }

    #[test]
    fn read_chunk_output_preserves_layout() {
        let s = format_chunks_for_llm(&[chunk(10, 12, "fn bar() {\n    do_thing()\n}")]);
        assert!(s.contains("src/foo.rs:10-12"));
        assert!(s.contains("fn bar"));
        // Indentation survives — this is code the caller will read.
        assert!(s.contains("\n    do_thing()"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn read_chunk_output_is_capped() {
        let huge = "x".repeat(MAX_READ_CHUNK_CHARS + 5_000);
        let s = format_chunks_for_llm(&[chunk(1, 900, &huge), chunk(901, 950, "tail")]);
        assert!(s.contains("truncated"));
        // The second chunk's header is still shown so the caller knows what
        // it didn't get.
        assert!(s.contains("src/foo.rs:901-950"));
        assert!(!s.contains("tail\n"));
    }

    #[test]
    fn ok_response_shape() {
        let r = ok_response(json!(7), json!({"x": 1}));
        assert_eq!(r["jsonrpc"], "2.0");
        assert_eq!(r["id"], 7);
        assert_eq!(r["result"]["x"], 1);
        assert!(r.get("error").is_none());
    }

    #[test]
    fn error_response_shape() {
        let r = error_response(json!(7), ERR_INVALID_PARAMS, "bad arg");
        assert_eq!(r["jsonrpc"], "2.0");
        assert_eq!(r["id"], 7);
        assert_eq!(r["error"]["code"], ERR_INVALID_PARAMS);
        assert_eq!(r["error"]["message"], "bad arg");
        assert!(r.get("result").is_none());
    }

    #[test]
    fn format_results_empty() {
        let s = format_results_for_llm(&[], "x", 5, true);
        assert!(s.contains("no matches"));
    }

    fn sample_result(rerank_score: Option<f32>) -> SearchResult {
        SearchResult {
            file: "src/foo.rs".to_string(),
            start_line: 10,
            end_line: 20,
            lang: "rust".to_string(),
            score: 0.91,
            dense_score: Some(0.8),
            sparse_score: Some(12.5),
            rerank_score,
            preview: "fn   bar() { do_thing()   }".to_string(),
            kind: Some("fn".to_string()),
            name: Some("bar".to_string()),
        }
    }

    #[test]
    fn format_results_renders_each_hit() {
        let s = format_results_for_llm(&[sample_result(Some(0.91))], "q", 5, true);
        assert!(s.contains("src/foo.rs:10-20"));
        assert!(s.contains("rust"));
        assert!(s.contains("0.9100"));
        // Whitespace runs collapsed in preview.
        assert!(s.contains("fn bar() { do_thing() }"));
        // Reranker delivered — no degradation warning.
        assert!(!s.contains("WARNING"));
    }

    #[test]
    fn format_results_warns_on_rerank_fallback() {
        // Reranker expected but no hit carries a rerank score → the RRF
        // fallback fired; the LLM must see that the ordering is degraded.
        let s = format_results_for_llm(&[sample_result(None)], "q", 5, true);
        assert!(s.contains("WARNING"));
        assert!(s.contains("RRF"));
    }

    #[test]
    fn format_results_no_warning_when_rerank_disabled() {
        // Reranker intentionally disabled in config — RRF-only is the
        // expected mode, not a degradation.
        let s = format_results_for_llm(&[sample_result(None)], "q", 5, false);
        assert!(!s.contains("WARNING"));
    }
}
