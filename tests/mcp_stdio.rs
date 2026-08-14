//! End-to-end test of the MCP stdio loop against the real binary.
//!
//! The unit tests in `serve.rs` cover response *shapes*; this covers the
//! thing they structurally cannot: that the process as a whole speaks
//! clean line-delimited JSON-RPC on stdout. A stray `println!` anywhere
//! reachable from `serve` silently breaks every MCP client, and nothing
//! else in the suite would catch it.
//!
//! No external services are needed. The config points Qdrant and the
//! embedding server at a closed port, so the search context fails fast;
//! the handshake, tool listing, error codes and framing are all
//! service-independent.

use std::io::Write;
use std::process::{Command, Stdio};

/// Unique scratch dir — mirrors the approach in `bm25.rs`'s e2e test so
/// the suite stays dependency-free.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "code-search-mcp-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &std::path::Path) -> std::path::PathBuf {
    // Port 1 is reserved and closed: connections are refused immediately
    // rather than hanging until a timeout.
    let toml = format!(
        r#"
[project]
id = "mcp-stdio-test"
root = "{root}"

[index]

[embedding]
provider = "openai-compatible"
url = "http://127.0.0.1:1"
model = "test-model"
dimensions = 4
startup_wait_secs = 0

[vector_store]
provider = "qdrant"
url = "http://127.0.0.1:1"

[bm25]
provider = "tantivy"
index_path = "{index_path}"

[watcher]
enabled = false

[chunking]
strategy = "lines"
"#,
        root = dir.display(),
        index_path = dir.join("tantivy").display(),
    );
    let path = dir.join("code-search.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

/// Drive one `serve` session: write every line, close stdin, collect stdout.
fn serve_session(
    config: &std::path::Path,
    requests: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_code-search-mcp"))
        .arg("-c")
        .arg(config)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning serve");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for req in requests {
            writeln!(stdin, "{}", req).unwrap();
        }
        // Deliberately malformed line: the server must answer with a parse
        // error and keep serving, not die or emit non-JSON.
        writeln!(stdin, "{{not json at all").unwrap();
        // Blank lines are skipped by the reader task.
        writeln!(stdin).unwrap();
    }
    // Dropping stdin closes it; the serve loop exits on EOF.
    drop(child.stdin.take());

    let out = child.wait_with_output().expect("waiting for serve");
    let stdout = String::from_utf8(out.stdout).expect("stdout must be UTF-8");

    stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("stdout carried a non-JSON line ({e}): {line:?}\nstdout was:\n{stdout}")
            })
        })
        .collect()
}

fn by_id(responses: &[serde_json::Value], id: i64) -> &serde_json::Value {
    responses
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("no response with id {id} in {responses:#?}"))
}

#[test]
fn serve_speaks_clean_json_rpc() {
    let dir = scratch_dir("stdio");
    let config = write_config(&dir);

    let responses = serve_session(
        &config,
        &[
            serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "2024-11-05", "capabilities": {} }
            }),
            // A notification: no id, so it must produce no response at all.
            serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 3, "method": "ping" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 4, "method": "no/such/method" }),
            serde_json::json!({ "jsonrpc": "2.0", "id": 5 }),
        ],
    );

    // Framing: every line is a JSON-RPC envelope (parsing already happened
    // in serve_session; this asserts the envelope itself).
    for r in &responses {
        assert_eq!(r["jsonrpc"], "2.0", "bad envelope: {r}");
    }

    // Handshake, answered in the dialect the client asked for.
    let init = by_id(&responses, 1);
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init["result"]["serverInfo"]["name"], "code-search-mcp");
    assert!(init["result"]["capabilities"]["tools"].is_object());

    // Both tools are advertised with a usable schema.
    let tools = by_id(&responses, 2)["result"]["tools"]
        .as_array()
        .expect("tools array")
        .clone();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(names, vec!["code_search", "code_read_chunk"]);
    for tool in &tools {
        assert_eq!(tool["inputSchema"]["type"], "object");
        assert!(
            tool["description"].as_str().is_some_and(|d| !d.is_empty()),
            "tool {} has no description",
            tool["name"]
        );
    }

    assert!(by_id(&responses, 3)["result"].is_object(), "ping");
    assert_eq!(by_id(&responses, 4)["error"]["code"], -32601);
    assert_eq!(by_id(&responses, 5)["error"]["code"], -32600);

    // The malformed line got a parse error with a null id.
    assert!(
        responses
            .iter()
            .any(|r| r["error"]["code"] == -32700 && r["id"].is_null()),
        "no parse error for the malformed line: {responses:#?}"
    );

    // Notifications produce nothing: 5 requests answered, plus the parse
    // error, and not one line more.
    assert_eq!(responses.len(), 6, "unexpected traffic: {responses:#?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tool_call_without_services_is_a_tool_error_not_a_crash() {
    let dir = scratch_dir("stdio-err");
    let config = write_config(&dir);

    // Qdrant is unreachable. The call must come back as an MCP tool-level
    // error the model can react to — not a JSON-RPC error, and certainly
    // not a dead server.
    let responses = serve_session(
        &config,
        &[
            serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "code_read_chunk", "arguments": { "file": "src/main.rs" } }
            }),
            serde_json::json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": { "name": "nope", "arguments": {} }
            }),
        ],
    );

    let call = by_id(&responses, 1);
    assert!(
        call.get("error").is_none(),
        "should not be a protocol error"
    );
    assert_eq!(call["result"]["isError"], true);
    assert!(call["result"]["content"][0]["text"].is_string());

    assert_eq!(by_id(&responses, 2)["error"]["code"], -32601);

    std::fs::remove_dir_all(&dir).ok();
}
