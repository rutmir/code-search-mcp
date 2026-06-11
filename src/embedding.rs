use anyhow::{Context, Result};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::EmbeddingConfig;

/// HTTP client to an OpenAI-compatible embeddings endpoint
/// (`/v1/embeddings`, as exposed by llama.cpp server, vLLM, Ollama, etc.).
///
/// This client does *no* internal retry — the adaptive batcher in `indexer`
/// owns the retry/halve/skip policy, since it has the context to classify
/// errors (workload-too-big vs. server-down) by probing.
pub struct Client {
    http: HttpClient,
    url: String,
    model: String,
}

/// How an error from `embed_with_timeout` should be handled.
///
/// `Ambiguous` means the error string alone can't tell us whether the server
/// is dying or the request is just too big — the caller should issue a quick
/// probe (`is_alive_quick`) to disambiguate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Server unreachable or unhealthy: connection refused, "loading model".
    /// Caller waits for recovery, then retries the *same* batch.
    ServerDown,
    /// Server explicitly rejected the input as too large, or the request
    /// timed out with a healthy server (workload too big for current capacity).
    /// Caller halves the budget and retries the first half of the batch.
    WorkloadTooBig,
    /// Server returned 4xx or otherwise indicated the input is malformed.
    /// Caller skips the offending input (no retry).
    PermanentBad,
    /// Need a probe to disambiguate ServerDown vs WorkloadTooBig.
    /// Used for raw timeouts and connection-closed errors which can be either.
    Ambiguous,
}

impl Client {
    pub fn new(config: &EmbeddingConfig) -> Self {
        let http = HttpClient::builder()
            // This is the *outer* HTTP timeout cap; the adaptive batcher sets
            // a per-request timeout via tokio::time::timeout that's usually
            // much shorter (proportional to batch size).
            .timeout(Duration::from_secs(config.timeout_secs))
            // Aggressively recycle idle connections to dodge server-side keep-alive
            // timeouts that surface as "connection closed before message completed".
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self {
            http,
            url: config.url.clone(),
            model: config.model.clone(),
        }
    }

    /// Single-attempt embed using the underlying client's default timeout.
    /// Used by `probe_dimensions` and any caller that doesn't need adaptive
    /// control. No retry — failures propagate.
    pub async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let body = EmbedRequest {
            model: &self.model,
            input: &inputs,
        };
        let parsed = self.try_post(&body).await?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Single-attempt embed with a caller-specified timeout. Used by the
    /// adaptive batcher, which scales the timeout by batch size.
    pub async fn embed_with_timeout(
        &self,
        inputs: Vec<String>,
        timeout: Duration,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let body = EmbedRequest {
            model: &self.model,
            input: &inputs,
        };
        match tokio::time::timeout(timeout, self.try_post(&body)).await {
            Ok(Ok(parsed)) => Ok(parsed.data.into_iter().map(|d| d.embedding).collect()),
            Ok(Err(e)) => Err(e),
            Err(_) => anyhow::bail!("embed request timed out after {}s", timeout.as_secs()),
        }
    }

    async fn try_post(&self, body: &EmbedRequest<'_>) -> Result<EmbedResponse> {
        let resp = self
            .http
            .post(&self.url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        if !status.is_success() {
            // Capture the response body — llama.cpp returns useful diagnostics
            // here (OOM, input too long, tokenizer error). error_for_status()
            // throws this away.
            let body_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "<unable to read response body>".to_string());
            anyhow::bail!(
                "embedding endpoint returned HTTP {}: {}",
                status.as_u16(),
                body_text.trim()
            );
        }
        let parsed: EmbedResponse = resp.json().await.context("parsing embedding response")?;
        Ok(parsed)
    }

    /// Probe the endpoint with a single tiny input; return the embedding
    /// dimension. Used by `check` to validate that the embedder is alive
    /// and that the declared dimensions match.
    pub async fn probe_dimensions(&self) -> Result<usize> {
        let vecs = self.embed(vec!["probe".to_string()]).await?;
        let v = vecs
            .first()
            .context("embedding endpoint returned no vectors for probe")?;
        Ok(v.len())
    }

    /// Liveness check (≤30s) used by the adaptive batcher to disambiguate
    /// Ambiguous errors. If the server answers a tiny probe within 30s,
    /// the previous failure was workload, not server health.
    ///
    /// 30s (rather than 5s) is required on single-slot llama.cpp servers
    /// (`--parallel 1`): when the previous request was cancelled on timeout,
    /// the slot still has to drain the cancelled task before accepting the
    /// probe. A 5s budget routinely falls inside that drain window and
    /// produces false "server down" classifications.
    pub async fn is_alive_quick(&self) -> bool {
        matches!(
            tokio::time::timeout(Duration::from_secs(30), self.try_post_probe()).await,
            Ok(Ok(()))
        )
    }

    /// Poll the endpoint until it accepts an embed request or `max_wait`
    /// elapses. Handles the case where the llama.cpp server is up but still
    /// loading the model (returns 503 "Loading model") — typical for
    /// jina-code-embeddings: 1-3 minutes. Also covers the case where the
    /// TCP listener isn't up yet (connect timeout).
    pub async fn wait_until_ready(&self, max_wait: Duration) -> Result<()> {
        let start = std::time::Instant::now();
        let poll_interval = Duration::from_secs(5);
        // Only emit progress logs roughly every 30s instead of on every poll.
        // Otherwise a 5-minute wait produces 60 INFO lines × N consecutive
        // failing files = 180 lines drowning out the actual signal.
        let log_throttle = Duration::from_secs(30);
        let mut probe_attempt = 0u32;
        let mut last_log_at = std::time::Instant::now() - log_throttle; // log first attempt
        loop {
            probe_attempt += 1;
            // Short-timeout probe so a hung TCP connect doesn't eat the
            // whole budget on one attempt.
            match tokio::time::timeout(Duration::from_secs(10), self.try_post_probe()).await {
                Ok(Ok(())) => {
                    tracing::info!(
                        attempt = probe_attempt,
                        elapsed_s = start.elapsed().as_secs(),
                        "embedding server ready"
                    );
                    return Ok(());
                }
                Ok(Err(e)) => {
                    if start.elapsed() >= max_wait {
                        return Err(e).context(format!(
                            "embedding server not ready after {}s",
                            max_wait.as_secs()
                        ));
                    }
                    if last_log_at.elapsed() >= log_throttle {
                        tracing::info!(
                            attempt = probe_attempt,
                            elapsed_s = start.elapsed().as_secs(),
                            error = %short_err(&e),
                            "embedding server not ready yet (will retry)"
                        );
                        last_log_at = std::time::Instant::now();
                    }
                }
                Err(_timeout) => {
                    if start.elapsed() >= max_wait {
                        anyhow::bail!(
                            "embedding server not ready after {}s (probe kept timing out)",
                            max_wait.as_secs()
                        );
                    }
                    if last_log_at.elapsed() >= log_throttle {
                        tracing::info!(
                            attempt = probe_attempt,
                            elapsed_s = start.elapsed().as_secs(),
                            "probe timed out, server likely still starting"
                        );
                        last_log_at = std::time::Instant::now();
                    }
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn try_post_probe(&self) -> Result<()> {
        let body = EmbedRequest {
            model: &self.model,
            input: &[String::from("probe")],
        };
        self.try_post(&body).await.map(|_| ())
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

/// Classify an embedding error so the adaptive batcher can decide what to do.
/// Pure function on the error chain; if the answer is `Ambiguous`, caller
/// should probe the server to disambiguate.
pub fn classify(err: &anyhow::Error) -> ErrorClass {
    let chain = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
        .to_lowercase();

    // Explicit "input is too big" signals from the server. Halve and retry.
    if chain.contains("too large to process")
        || chain.contains("exceeds context")
        || chain.contains("exceeds the context")
        || chain.contains("input is too large")
        || chain.contains("input is too long")
        || chain.contains("input length")
        || chain.contains("token limit")
    {
        return ErrorClass::WorkloadTooBig;
    }

    // Server unhealthy and explicit about it.
    if chain.contains("loading model") || chain.contains("connection refused") {
        return ErrorClass::ServerDown;
    }

    // 4xx: bad request from us; retrying won't change anything.
    if chain.contains("http 400")
        || chain.contains("http 401")
        || chain.contains("http 403")
        || chain.contains("http 404")
        || chain.contains("http 415")
        || chain.contains("http 422")
    {
        return ErrorClass::PermanentBad;
    }

    // Ambiguous: either the server is dying or the workload is too big.
    // A raw timeout could be a frozen server OR a healthy-but-busy one.
    // A "connection closed" could be a server crash on big input OR a
    // keep-alive race on a healthy connection. Caller probes to decide.
    if chain.contains("operation timed out")
        || chain.contains("timed out")
        || chain.contains("timeout")
        || chain.contains("connection closed")
        || chain.contains("connection reset")
        || chain.contains("broken pipe")
        || chain.contains("http 500")
        || chain.contains("http 502")
        || chain.contains("http 503")
        || chain.contains("http 504")
    {
        return ErrorClass::Ambiguous;
    }

    // Unknown: default to PermanentBad so we don't retry blindly.
    ErrorClass::PermanentBad
}

/// Used by the indexer's `consecutive_server_down` fail-fast counter to
/// decide whether to abort the whole run after N file failures in a row.
/// Looser than `classify(...) == ServerDown` because by the time the
/// adaptive batcher gives up on a file and returns Err, the wrapping
/// context strings make exact classification noisy.
pub fn is_server_down(err: &anyhow::Error) -> bool {
    let chain = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
        .to_lowercase();
    chain.contains("connection refused")
        || chain.contains("connection timed out")
        || chain.contains("operation timed out")
        || chain.contains("loading model")
        || chain.contains("server unavailable after recovery")
}

pub(crate) fn short_err(e: &anyhow::Error) -> String {
    e.chain()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(160)
        .collect()
}
