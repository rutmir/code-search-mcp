use anyhow::{Context, Result};
use ignore::{gitignore::Gitignore, overrides::OverrideBuilder, WalkBuilder};
use std::path::{Path, PathBuf};

use crate::config::{IndexConfig, ProjectConfig};

pub struct FileEntry {
    pub absolute: PathBuf,
    pub relative: PathBuf,
    pub language: String,
}

pub fn walk(
    project: &ProjectConfig,
    index: &IndexConfig,
) -> Result<impl Iterator<Item = Result<FileEntry>>> {
    let root = project.root.clone();
    let mut builder = WalkBuilder::new(&root);

    if index.respect_gitignore {
        builder.standard_filters(true);
    } else {
        builder
            .git_ignore(false)
            .git_exclude(false)
            .git_global(false)
            .ignore(false);
    }
    builder.hidden(true);

    if !index.exclude.is_empty() {
        let mut overrides = OverrideBuilder::new(&root);
        for pattern in &index.exclude {
            // The `!` prefix marks the pattern as something to ignore.
            overrides
                .add(&format!("!{}", pattern))
                .with_context(|| format!("invalid exclude pattern: {}", pattern))?;
        }
        builder.overrides(overrides.build()?);
    }

    let extensions = allowed_extensions(&index.languages);
    let root_for_filter = root.clone();

    let iter = builder.build().filter_map(move |result| {
        let entry = match result {
            Ok(e) => e,
            Err(e) => return Some(Err(anyhow::anyhow!("walk error: {}", e))),
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            return None;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if !extensions.iter().any(|e| e == &ext) {
            return None;
        }
        let relative = path
            .strip_prefix(&root_for_filter)
            .unwrap_or(path)
            .to_path_buf();
        Some(Ok(FileEntry {
            absolute: path.to_path_buf(),
            relative,
            language: language_from_ext(&ext),
        }))
    });

    Ok(iter)
}

fn allowed_extensions(languages: &[String]) -> Vec<String> {
    languages
        .iter()
        .flat_map(|l| extensions_for_language(l).iter().map(|s| s.to_string()))
        .collect()
}

/// Stateless "should this path be indexed" check used by the watcher to
/// decide whether to process a filesystem-event-arrived path. Built once
/// at startup so per-event cost is just pattern matching, not a full walk.
///
/// Filters applied (in order):
///   1. Path must currently exist as a regular file
///   2. Extension must match one of the configured languages
///   3. If `respect_gitignore`, project-root `.gitignore` must not match
///   4. Excludes (`index.exclude`) must not match
///
/// Limitations: only the project-root `.gitignore` is considered; the
/// recursive walker honors nested `.gitignore` files too. Good enough for
/// the typical layout where root .gitignore covers `target/`, `build/`,
/// `node_modules/`, etc.
pub struct PathFilter {
    extensions: Vec<String>,
    gitignore: Option<Gitignore>,
    excludes: Option<ignore::overrides::Override>,
    root: PathBuf,
}

impl PathFilter {
    pub fn new(project: &ProjectConfig, index: &IndexConfig) -> Result<Self> {
        let extensions = allowed_extensions(&index.languages);

        let gitignore = if index.respect_gitignore {
            // Collect every `.gitignore` in the project tree so nested ones
            // (e.g. `crates/foo/.gitignore`) are honored alongside the root
            // one. Uses `WalkBuilder` which already understands the gitignore
            // hierarchy and skips dirs that are themselves gitignored, so
            // we don't recurse into target/, node_modules/, etc.
            let mut builder = ignore::gitignore::GitignoreBuilder::new(&project.root);
            let mut found = 0usize;
            for result in ignore::WalkBuilder::new(&project.root)
                .standard_filters(true)
                .hidden(false) // include .gitignore itself (hidden dir entries are filtered separately)
                .build()
            {
                let Ok(entry) = result else { continue };
                if entry.file_name() != ".gitignore" {
                    continue;
                }
                if let Some(err) = builder.add(entry.path()) {
                    tracing::warn!(
                        path = %entry.path().display(),
                        error = %err,
                        ".gitignore parse warning"
                    );
                } else {
                    found += 1;
                }
            }
            tracing::debug!(found, "loaded .gitignore files");
            Some(builder.build().context("building gitignore matcher")?)
        } else {
            None
        };

        let excludes = if !index.exclude.is_empty() {
            let mut builder = OverrideBuilder::new(&project.root);
            for pattern in &index.exclude {
                builder
                    .add(&format!("!{}", pattern))
                    .with_context(|| format!("invalid exclude pattern: {}", pattern))?;
            }
            Some(builder.build()?)
        } else {
            None
        };

        Ok(Self {
            extensions,
            gitignore,
            excludes,
            root: project.root.clone(),
        })
    }

    /// Returns a [`FileEntry`] if the path is indexable, `None` otherwise.
    /// Cheap — no I/O beyond the existence/file-type check.
    pub fn check(&self, path: &Path) -> Option<FileEntry> {
        // 1. Must currently exist as a regular file. Symlinks resolved.
        let meta = std::fs::metadata(path).ok()?;
        if !meta.is_file() {
            return None;
        }

        // 2. Extension must be in the allowed set.
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !self.extensions.iter().any(|e| e == ext) {
            return None;
        }

        // 3. Gitignore check.
        if let Some(gi) = &self.gitignore {
            if gi.matched(path, false).is_ignore() {
                return None;
            }
        }

        // 4. Excludes check.
        if let Some(ov) = &self.excludes {
            if ov.matched(path, false).is_ignore() {
                return None;
            }
        }

        let relative = path.strip_prefix(&self.root).unwrap_or(path).to_path_buf();
        Some(FileEntry {
            absolute: path.to_path_buf(),
            relative,
            language: language_from_ext(ext),
        })
    }
}

fn extensions_for_language(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &["rs"],
        "python" => &["py", "pyi"],
        "cpp" => &["cpp", "cc", "cxx", "hpp", "hxx", "h", "ipp"],
        "typescript" => &["ts", "tsx", "mts", "cts"],
        "javascript" => &["js", "jsx", "mjs", "cjs"],
        "go" => &["go"],
        "csharp" => &["cs"],
        "java" => &["java"],
        "toml" => &["toml"],
        "markdown" => &["md", "markdown"],
        "dart" => &["dart"],
        "yaml" => &["yaml", "yml"],
        "json" => &["json"],
        // Plain-text-ish formats. No tree-sitter for these — the line
        // chunker (the default fallback) handles them well because they
        // tend to be small and have flat structure.
        "shell" => &["sh", "bash", "zsh"],
        "systemd" => &["service", "socket", "timer", "mount", "target", "path"],
        "env" => &["env"],
        // `text` is the generic "this file is plain text I want indexed"
        // bucket. Common config-file suffixes (`*.example`, `*.local`,
        // `*.ini`, `*.cfg`, `*.conf`) and `.txt` notes go here. Pick this
        // up by adding "text" to `[index].languages`.
        "text" => &["txt", "local", "example", "ini", "cfg", "conf"],
        _ => &[],
    }
}

fn language_from_ext(ext: &str) -> String {
    match ext {
        "rs" => "rust",
        "py" | "pyi" => "python",
        // NOTE: `.h` is ambiguous between C and C++. Treated as C++ here —
        // tree-sitter-cpp parses pure C correctly because C is largely a
        // subset of C++. For a C-only project, point indexing at a C
        // grammar instead.
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "h" | "ipp" => "cpp",
        "ts" | "tsx" | "mts" | "cts" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "go" => "go",
        "cs" => "csharp",
        "java" => "java",
        "toml" => "toml",
        "md" | "markdown" => "markdown",
        "dart" => "dart",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "sh" | "bash" | "zsh" => "shell",
        "service" | "socket" | "timer" | "mount" | "target" | "path" => "systemd",
        "env" => "env",
        "txt" | "local" | "example" | "ini" | "cfg" | "conf" => "text",
        _ => "text",
    }
    .to_string()
}
