use anyhow::{Context, Result};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Duration;

use crate::config::RerankerConfig;

/// HTTP client to a Jina-style `/v1/rerank` endpoint (e.g.
/// bge-reranker-v2-m3 as served by llama.cpp with `--reranking`).
///
/// Per-request timeout is throughput-aware: the client tracks an EWMA of
/// observed `chars / sec` across successful rerank calls and derives the
/// next call's timeout from that — no static `timeout_secs` to guess at
/// every hardware change.
pub struct Client {
    http: HttpClient,
    url: String,
    model: String,
    max_document_chars: usize,
    /// EWMA of chars/sec across successful rerank calls; `None` until first
    /// observation. Mutex because rerank() takes `&self` but updates state;
    /// std::sync::Mutex is fine — we never hold it across an await.
    chars_per_sec: Mutex<Option<f64>>,
}

/// Conservative throughput estimate before any observations.
/// 200 chars/s ≈ 70 tok/s — sized for CPU bge-v2-m3 with `--parallel 1`
/// (under-estimate on 4-slot setups; the EWMA quickly corrects up).
const BOOTSTRAP_CHARS_PER_SEC: f64 = 200.0;
/// EWMA smoothing factor. 0.3 ≈ converges in ~5-10 calls.
const THROUGHPUT_EWMA_ALPHA: f64 = 0.3;
/// Fixed overhead per request: TCP, JSON parse, server queue.
const TIMEOUT_BASE_SECS: f64 = 30.0;
/// Multiplier on the estimated processing time. Absorbs jitter and the
/// difference between first-batch (cold cache) and steady-state throughput.
const TIMEOUT_SAFETY: f64 = 3.0;
/// Per-request timeout floor — even a tiny request needs server scheduling.
const TIMEOUT_FLOOR_SECS: f64 = 30.0;
/// Per-request timeout ceiling. Beyond this the server is almost certainly
/// stuck (real hang, crash); the fallback-to-RRF path is the right response.
const TIMEOUT_CEILING_SECS: f64 = 1200.0;

/// How many "batch too large" rejections trigger a halve-and-retry before
/// giving up. Two halvings take the default 8000-char budget down to 2000 —
/// under any physical batch limit a working server could plausibly have.
const MAX_TOO_LARGE_RETRIES: u32 = 2;
/// Never truncate below this. Documents this short are barely rankable;
/// if the server still rejects them, it's misconfigured and the caller's
/// RRF fallback is the right answer.
const TRUNCATION_FLOOR_CHARS: usize = 512;

/// Does this error look like the server rejecting the request for size
/// (physical batch / context overflow) rather than being down or slow?
/// Matched against llama.cpp's diagnostics ("input (N tokens) is too large
/// to process...", "...exceeds the available context size") — retrying
/// with harder truncation only makes sense for these.
fn is_too_large_error(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}").to_lowercase();
    msg.contains("too large") || msg.contains("context size") || msg.contains("exceeds")
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

impl Client {
    pub fn new(config: &RerankerConfig) -> Self {
        // Reqwest's own timeout is the *outer* cap; the throughput-aware
        // tokio::time::timeout inside `rerank()` is the binding constraint
        // for normal operation. Cap is set well above TIMEOUT_CEILING_SECS
        // so it never fires before the dynamic one.
        let outer_cap_secs = config.timeout_secs.max(TIMEOUT_CEILING_SECS as u64 + 30);
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(outer_cap_secs))
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self {
            http,
            url: config.url.clone(),
            model: config.model.clone(),
            max_document_chars: config.max_document_chars,
            chars_per_sec: Mutex::new(None),
        }
    }

    /// Estimate per-request timeout from observed throughput (or bootstrap
    /// if no observations yet). Clamped to [`TIMEOUT_FLOOR_SECS`,
    /// `TIMEOUT_CEILING_SECS`].
    fn estimate_timeout(&self, total_chars: usize) -> Duration {
        let cps = self
            .chars_per_sec
            .lock()
            .unwrap()
            .unwrap_or(BOOTSTRAP_CHARS_PER_SEC);
        let secs = TIMEOUT_BASE_SECS + TIMEOUT_SAFETY * (total_chars as f64 / cps);
        Duration::from_secs_f64(secs.clamp(TIMEOUT_FLOOR_SECS, TIMEOUT_CEILING_SECS))
    }

    /// Update the throughput EWMA after a successful rerank. Pulls the
    /// estimate ~30% toward the new observation each call.
    fn note_success(&self, total_chars: usize, elapsed: Duration) {
        let secs = elapsed.as_secs_f64().max(0.001);
        let observed = total_chars as f64 / secs;
        let mut guard = self.chars_per_sec.lock().unwrap();
        *guard = Some(match *guard {
            None => observed,
            Some(prev) => (1.0 - THROUGHPUT_EWMA_ALPHA) * prev + THROUGHPUT_EWMA_ALPHA * observed,
        });
    }

    /// Observed throughput (chars/sec), if at least one rerank has succeeded.
    /// Mainly for introspection / tests.
    pub fn chars_per_sec(&self) -> Option<f64> {
        *self.chars_per_sec.lock().unwrap()
    }

    /// Score each document for relevance to the query. Returns scores in
    /// the same order as the input `documents` (the server returns them
    /// sorted by score with `index` references; we re-arrange here so the
    /// caller can pair scores with metadata it kept alongside the docs).
    ///
    /// Documents are truncated to `max_document_chars` before being sent —
    /// cross-encoder rerankers commonly have small (1-8K token) context
    /// windows, and a single oversized doc causes the whole rerank request
    /// to fail with "input is too large". If the server still rejects the
    /// batch as too large (chars→tokens ratio varies by language; Cyrillic
    /// markdown packs ~2× more tokens per char than ASCII code), the
    /// truncation limit is halved and the call retried — a degraded rerank
    /// over shortened documents beats losing the cross-encoder entirely.
    pub async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut limit = self.max_document_chars;
        let mut retries = 0u32;
        loop {
            match self.rerank_attempt(query, &documents, limit).await {
                Ok(scores) => return Ok(scores),
                Err(e)
                    if retries < MAX_TOO_LARGE_RETRIES
                        && limit / 2 >= TRUNCATION_FLOOR_CHARS
                        && is_too_large_error(&e) =>
                {
                    limit /= 2;
                    retries += 1;
                    tracing::warn!(
                        new_limit = limit,
                        retry = retries,
                        error = %e,
                        "reranker rejected batch as too large; halving truncation and retrying"
                    );
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// One rerank round-trip with documents truncated to `limit` chars.
    async fn rerank_attempt(
        &self,
        query: &str,
        documents: &[String],
        limit: usize,
    ) -> Result<Vec<f32>> {
        let n = documents.len();
        let mut truncated_count = 0usize;
        let prepared: Vec<String> = documents
            .iter()
            .map(|d| {
                if d.len() > limit {
                    truncated_count += 1;
                    // chars().take() avoids cutting in the middle of a
                    // multi-byte codepoint; cheap relative to network RTT.
                    d.chars().take(limit).collect()
                } else {
                    d.clone()
                }
            })
            .collect();
        if truncated_count > 0 {
            tracing::debug!(
                count = truncated_count,
                total = n,
                limit,
                "truncated documents to fit reranker context"
            );
        }

        let total_chars: usize = query.len() + prepared.iter().map(|d| d.len()).sum::<usize>();
        let timeout = self.estimate_timeout(total_chars);
        tracing::debug!(
            n,
            total_chars,
            timeout_s = timeout.as_secs(),
            chars_per_sec = self.chars_per_sec().unwrap_or(0.0) as u64,
            "rerank starting"
        );

        let body = RerankRequest {
            model: &self.model,
            query,
            documents: &prepared,
        };
        let started = std::time::Instant::now();
        let send_fut = self.post_and_parse(&body);
        let parsed = match tokio::time::timeout(timeout, send_fut).await {
            Ok(Ok(parsed)) => parsed,
            Ok(Err(e)) => return Err(e),
            Err(_) => anyhow::bail!(
                "rerank timed out after {}s (estimated for {} chars at ~{} chars/sec)",
                timeout.as_secs(),
                total_chars,
                self.chars_per_sec().unwrap_or(BOOTSTRAP_CHARS_PER_SEC) as u64
            ),
        };

        // Record throughput observation for the next call's estimate.
        let elapsed = started.elapsed();
        self.note_success(total_chars, elapsed);
        tracing::debug!(
            n,
            total_chars,
            elapsed_ms = elapsed.as_millis() as u64,
            new_chars_per_sec = self.chars_per_sec().unwrap_or(0.0) as u64,
            "rerank done"
        );

        // Server returns scores sorted by relevance with `index` pointing
        // back into our input array. Re-arrange to input order so the
        // caller can zip with kept-aside metadata.
        let mut scores = vec![0.0f32; n];
        for item in parsed.results {
            if item.index >= n {
                anyhow::bail!(
                    "reranker returned out-of-range index {} for {} documents",
                    item.index,
                    n
                );
            }
            scores[item.index] = item.relevance_score;
        }
        Ok(scores)
    }

    /// POST + body capture + JSON parse. Extracted so the timeout wrapper
    /// in `rerank()` can box it via `tokio::time::timeout`.
    async fn post_and_parse(&self, body: &RerankRequest<'_>) -> Result<RerankResponse> {
        let resp = self
            .http
            .post(&self.url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        if !status.is_success() {
            // Capture body — llama.cpp returns useful diagnostics on rerank
            // failures (context overflow, model error, etc).
            let body_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "<unable to read response body>".to_string());
            anyhow::bail!(
                "reranker endpoint returned HTTP {}: {}",
                status.as_u16(),
                body_text.trim()
            );
        }
        let parsed: RerankResponse = resp.json().await.context("parsing rerank response")?;
        Ok(parsed)
    }
}
