use anyhow::{Context, Result};
use ignore::{gitignore::Gitignore, overrides::OverrideBuilder, WalkBuilder};
use std::collections::HashSet;
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

    let policy = ExtPolicy::from_languages(&index.languages);
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
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !policy.admits(ext) {
            return None;
        }
        // Size cap (all-but-binary default only) — skip oversize files without
        // reading them. Metadata is cheap and only fetched once ext passes.
        if let Ok(meta) = entry.metadata() {
            if !policy.admits_size(meta.len()) {
                return None;
            }
        }
        let language = language_from_ext(ext);
        let relative = path
            .strip_prefix(&root_for_filter)
            .unwrap_or(path)
            .to_path_buf();
        Some(Ok(FileEntry {
            absolute: path.to_path_buf(),
            relative,
            language,
        }))
    });

    Ok(iter)
}

/// Files larger than this are skipped by the all-but-binary default without
/// being read — they're almost always generated (minified bundles, lockfiles,
/// embedded data) and would flood the index with low-value chunks. An explicit
/// `[index].languages` whitelist bypasses the cap: you named those types on
/// purpose.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Extensions the all-but-binary default never tries to read as text. Fast
/// pre-filter so large binaries aren't read + UTF-8-decoded just to be
/// rejected; the indexer's NUL/UTF-8 sniff is the backstop for binaries not
/// listed here (extensionless executables, exotic types). Matched
/// case-insensitively.
#[rustfmt::skip]
const BINARY_EXTENSIONS: &[&str] = &[
    // images
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "tif", "svg", "psd", "heic",
    // audio / video
    "mp3", "wav", "flac", "ogg", "aac", "m4a", "mp4", "m4v", "mov", "avi", "mkv", "webm",
    // archives / compressed
    "zip", "jar", "war", "aar", "ear", "tar", "gz", "tgz", "bz2", "xz", "zst", "7z", "rar", "lz4",
    // native binaries / objects
    "so", "dylib", "dll", "a", "lib", "o", "obj", "class", "dex", "exe", "bin", "wasm", "node",
    "pyc", "pyo", "pdb", "elf", "ko",
    // mobile / gpu artifacts
    "apk", "aab", "ipa", "spv", "nib", "car",
    // fonts
    "ttf", "otf", "woff", "woff2", "eot",
    // documents / data blobs
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "sqlite", "db", "dat",
    // ml weights / large arrays
    "gguf", "safetensors", "onnx", "pt", "pth", "ckpt", "npy", "npz", "parquet",
];

fn is_binary_extension(ext: &str) -> bool {
    let lower = ext.to_ascii_lowercase();
    BINARY_EXTENSIONS.contains(&lower.as_str())
}

/// How the walker decides which files to admit, derived from
/// `[index].languages`.
enum ExtPolicy {
    /// `languages` was set — admit only those languages' extensions (the way
    /// to *narrow* a noisy repo). No size cap: you named these types.
    Whitelist(HashSet<String>),
    /// `languages` omitted/empty — admit everything except known-binary
    /// extensions and oversize files. Zero-config default.
    AllButBinary,
}

impl ExtPolicy {
    fn from_languages(languages: &[String]) -> Self {
        if languages.is_empty() {
            ExtPolicy::AllButBinary
        } else {
            ExtPolicy::Whitelist(
                languages
                    .iter()
                    .flat_map(|l| extensions_for_language(l).iter().map(|s| s.to_string()))
                    .collect(),
            )
        }
    }

    fn admits(&self, ext: &str) -> bool {
        match self {
            ExtPolicy::Whitelist(exts) => exts.contains(ext),
            ExtPolicy::AllButBinary => !is_binary_extension(ext),
        }
    }

    /// Only the all-but-binary default caps file size; an explicit whitelist
    /// indexes what it named regardless of size (preserves prior behavior).
    fn admits_size(&self, len: u64) -> bool {
        match self {
            ExtPolicy::Whitelist(_) => true,
            ExtPolicy::AllButBinary => len <= MAX_FILE_BYTES,
        }
    }
}

/// Stateless "should this path be indexed" check used by the watcher to
/// decide whether to process a filesystem-event-arrived path. Built once
/// at startup so per-event cost is just pattern matching, not a full walk.
///
/// Filters applied (in order):
///   1. Path must currently exist as a regular file
///   2. Extension must satisfy the [`ExtPolicy`] (whitelist, or all-but-binary
///      when `languages` is omitted) and the file must be within the size cap
///   3. If `respect_gitignore`, project-root `.gitignore` must not match
///   4. Excludes (`index.exclude`) must not match
///
/// Limitations: only the project-root `.gitignore` is considered; the
/// recursive walker honors nested `.gitignore` files too. Good enough for
/// the typical layout where root .gitignore covers `target/`, `build/`,
/// `node_modules/`, etc.
pub struct PathFilter {
    policy: ExtPolicy,
    gitignore: Option<Gitignore>,
    excludes: Option<ignore::overrides::Override>,
    root: PathBuf,
}

impl PathFilter {
    pub fn new(project: &ProjectConfig, index: &IndexConfig) -> Result<Self> {
        let policy = ExtPolicy::from_languages(&index.languages);

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
            policy,
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

        // 2. Extension policy + size cap.
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !self.policy.admits(ext) || !self.policy.admits_size(meta.len()) {
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
        "xml" => &["xml"],
        // Build / manifest config formats for mobile & JVM projects. No
        // tree-sitter — line-chunked like the other structured-text buckets.
        // `xml` (above) covers AndroidManifest.xml and res/values/*.xml;
        // `gradle` covers build.gradle / settings.gradle (Groovy DSL);
        // `properties` covers gradle.properties / gradle-wrapper.properties.
        "gradle" => &["gradle", "kts"],
        "properties" => &["properties"],
        // Kotlin and Swift have no tree-sitter grammar wired up yet, so
        // they're line-chunked. They still get their own bucket rather
        // than falling into `text`: an Android or iOS project must be able
        // to name its primary language in an `[index].languages`
        // whitelist, and `lang = "kotlin"` must be a usable search filter.
        "kotlin" => &["kt"],
        "swift" => &["swift"],
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
        "xml" => "xml",
        // `.kts` is a Kotlin script, but in practice it is nearly always
        // build.gradle.kts — grouping it with `gradle` keeps a project's
        // build config in one filterable bucket.
        "gradle" | "kts" => "gradle",
        "properties" => "properties",
        "kt" => "kotlin",
        "swift" => "swift",
        "sh" | "bash" | "zsh" => "shell",
        "service" | "socket" | "timer" | "mount" | "target" | "path" => "systemd",
        "env" => "env",
        "txt" | "local" | "example" | "ini" | "cfg" | "conf" => "text",
        _ => "text",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whitelist(langs: &[&str]) -> ExtPolicy {
        ExtPolicy::from_languages(&langs.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn whitelist_admits_only_listed_language_extensions() {
        // Enabling these languages admits the Android manifest / resource /
        // build-config surface (previously dropped entirely) and nothing else.
        let p = whitelist(&["xml", "gradle", "properties"]);
        for ext in ["xml", "gradle", "properties"] {
            assert!(p.admits(ext), "{ext} should be admitted");
        }
        assert!(!p.admits("rs"), "rs not in the list");
        assert!(!p.admits(""), "extensionless not in the list");
        // Explicit whitelist never caps size — you named these types.
        assert!(p.admits_size(MAX_FILE_BYTES + 1));
    }

    #[test]
    fn empty_languages_is_all_but_binary() {
        // Zero-config default: index everything textual, skip known binaries.
        let p = ExtPolicy::from_languages(&[]);
        assert!(matches!(p, ExtPolicy::AllButBinary));
        // Text / code / config admitted, including extensionless (Makefile).
        for ext in [
            "rs", "java", "xml", "gradle", "toml", "md", "kt", "swift", "",
        ] {
            assert!(p.admits(ext), "{ext:?} should be admitted by default");
        }
        // Known binaries rejected without a config change.
        for ext in [
            "png", "jar", "so", "spv", "dex", "class", "obj", "PNG", "gguf",
        ] {
            assert!(!p.admits(ext), "{ext:?} is binary, must be rejected");
        }
        // Oversize files skipped under the default; within-cap admitted.
        assert!(p.admits_size(MAX_FILE_BYTES));
        assert!(!p.admits_size(MAX_FILE_BYTES + 1));
    }

    #[test]
    fn language_maps_back_to_its_bucket() {
        assert_eq!(language_from_ext("xml"), "xml");
        assert_eq!(language_from_ext("gradle"), "gradle");
        assert_eq!(language_from_ext("properties"), "properties");
        assert_eq!(language_from_ext("kt"), "kotlin");
        assert_eq!(language_from_ext("swift"), "swift");
        // Kotlin build scripts belong with the rest of the build config.
        assert_eq!(language_from_ext("kts"), "gradle");
        // Unknown text extension falls through to the line-chunked `text` bucket.
        assert_eq!(language_from_ext("scala"), "text");
    }

    /// Every bucket `extensions_for_language` can name must map back to
    /// itself, or an `[index].languages` whitelist would admit a file and
    /// then tag it with a language the user never asked for — which the
    /// `lang` search filter would then never match.
    #[test]
    fn whitelist_extensions_round_trip_to_their_language() {
        for lang in [
            "rust",
            "python",
            "cpp",
            "typescript",
            "javascript",
            "go",
            "csharp",
            "java",
            "toml",
            "markdown",
            "dart",
            "yaml",
            "json",
            "xml",
            "gradle",
            "properties",
            "kotlin",
            "swift",
            "shell",
            "systemd",
            "env",
            "text",
        ] {
            let exts = extensions_for_language(lang);
            assert!(!exts.is_empty(), "{lang} has no extensions");
            for ext in exts {
                assert_eq!(
                    language_from_ext(ext),
                    lang,
                    "extension {ext:?} is listed under {lang:?} but maps elsewhere"
                );
            }
        }
    }

    #[test]
    fn binary_extension_check_is_case_insensitive() {
        assert!(is_binary_extension("png"));
        assert!(is_binary_extension("PNG"));
        assert!(is_binary_extension("Spv"));
        assert!(!is_binary_extension("rs"));
    }
}
