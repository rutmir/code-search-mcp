//! Minimal MCP server: line-delimited JSON-RPC over stdin/stdout.
//!
//! Transport: per the MCP spec, each message is a single JSON object on its
//! own line (no Content-Length headers — unlike LSP). Server reads requests
//! from stdin, writes responses to stdout, and ALL logging goes to stderr.
//! Writing anything else to stdout silently breaks the client.
//!
//! Scope:
//!   - `initialize` / `notifications/initialized` handshake
//!   - `tools/list` advertising the `code_search` tool
//!   - `tools/call` dispatching to [`crate::search::run`]; each call runs
//!     in its own spawned task, tracked by request id
//!   - `notifications/cancelled` — preempts the matching in-flight search
//!     via a oneshot raced in `tokio::select!`
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
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::search::{self, SearchParams};
use crate::watcher;

/// Map from JSON-RPC stringified request id → cancel sender. A spawned
/// tools/call task races search vs. its cancel receiver in a select!,
/// so a send through the sender preempts the search and yields a clean
/// "cancelled" response.
type InFlight = Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<()>>>>;

/// Latest MCP protocol version this server implements.
const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "code-search-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    let in_flight: InFlight = Arc::new(std::sync::Mutex::new(HashMap::new()));

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
                    info!("stdin closed; shutting down");
                    return Ok(());
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
                    handle_notification(method, &msg, &in_flight);
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
                            config,
                            id,
                            msg,
                            Arc::clone(&in_flight),
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
}

/// Spawn a `tools/call` task: registers a oneshot cancel sender keyed by
/// the request id, runs the tool, and sends the result (or a cancelled
/// marker) back through `resp_tx`.
fn spawn_tools_call(
    config: &Config,
    id: Value,
    msg: Value,
    in_flight: InFlight,
    resp_tx: mpsc::UnboundedSender<Value>,
) {
    let id_key = id_to_key(&id);
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    {
        let mut guard = in_flight.lock().unwrap();
        // If client somehow reuses an id (shouldn't happen — JSON-RPC ids are
        // unique per pending request), the previous cancel sender is dropped
        // silently. The previous task will see its receiver close and treat
        // it as cancelled — acceptable.
        guard.insert(id_key.clone(), cancel_tx);
    }

    let config_clone = config.clone();
    let in_flight_clone = Arc::clone(&in_flight);
    let id_clone = id.clone();
    let id_key_clone = id_key.clone();

    tokio::spawn(async move {
        let response: Value = tokio::select! {
            result = handle_tools_call(&config_clone, &msg) => match result {
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
        in_flight_clone.lock().unwrap().remove(&id_key_clone);
        let _ = resp_tx.send(response);
    });
}

fn handle_notification(method: &str, msg: &Value, in_flight: &InFlight) {
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
            let removed = in_flight.lock().unwrap().remove(&id_key);
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

fn handle_initialize(_msg: &Value) -> Value {
    // We don't actually do anything with the client's `protocolVersion` /
    // `capabilities` / `clientInfo` yet — just advertise our own.
    json!({
        "protocolVersion": PROTOCOL_VERSION,
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
                            tight phrase than in a paragraph.",
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
        }]
    })
}

/// Handle `tools/call`. Returns Ok(result_value) for success (including
/// tool-level failures, which use `isError: true` inside the result) or
/// Err((code, msg)) for protocol-level errors (bad params, etc).
async fn handle_tools_call(config: &Config, msg: &Value) -> Result<Value, (i32, String)> {
    let name = msg
        .pointer("/params/name")
        .and_then(|v| v.as_str())
        .ok_or((ERR_INVALID_PARAMS, "missing 'params.name'".to_string()))?;

    if name != "code_search" {
        return Err((ERR_METHOD_NOT_FOUND, format!("unknown tool: {}", name)));
    }

    let args = msg
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or(Value::Null);

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

    let started = std::time::Instant::now();
    let result = search::run(
        config,
        SearchParams {
            query,
            limit,
            // Always use rerank in serve mode (when configured). Skipping
            // it is a CLI debugging knob, not useful via MCP.
            use_rerank: true,
            lang,
            path,
        },
    )
    .await;
    let elapsed_ms = started.elapsed().as_millis() as u64;

    match result {
        Ok(results) => {
            info!(
                results = results.len(),
                elapsed_ms, "tools/call code_search completed"
            );
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

/// Render search results as a structured text block the LLM can quote
/// from in subsequent reasoning. Lines stay short, paths are obvious,
/// scores are visible for sanity-checking.
fn format_results_for_llm(
    results: &[search::SearchResult],
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
    fn tools_list_advertises_code_search() {
        let v = handle_tools_list();
        let tools = v["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "code_search");
        let schema = &tools[0]["inputSchema"];
        assert_eq!(schema["type"], "object");
        // Required props must include query.
        let req = schema["required"].as_array().expect("required");
        assert!(req.iter().any(|v| v.as_str() == Some("query")));
        // limit / lang / path are advertised as optional.
        let props = schema["properties"].as_object().expect("props");
        for k in ["query", "limit", "lang", "path"] {
            assert!(props.contains_key(k), "missing schema for {}", k);
        }
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

    fn sample_result(rerank_score: Option<f32>) -> search::SearchResult {
        search::SearchResult {
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
