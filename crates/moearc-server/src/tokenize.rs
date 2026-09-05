//! Text ↔ token ids.
//!
//! Wraps Hugging Face's `tokenizers` — the reference implementation, and the one whose output
//! the models were actually trained against. Writing a BPE from scratch here would be a way to
//! be subtly wrong in a place nobody looks until generations come out mangled.
//!
//! Two sources are supported:
//!
//! - **`tokenizer.json`** ([`Tokenizer::from_tokenizer_json`]) — the serialised HF tokeniser,
//!   loaded verbatim. Nothing is inferred; pre-tokeniser, normaliser, decoder and added tokens
//!   all come from the file. **This is the verified path**, exercised in the tests below
//!   against a real 151k-entry Qwen vocabulary.
//! - **A GGUF's embedded vocabulary** ([`Tokenizer::from_gguf`]) — reconstructed from
//!   `tokenizer.ggml.*`. GGUF stores the vocabulary and merge table but *not* the pre-tokeniser
//!   regex, so that half has to be inferred from `tokenizer.ggml.model`. Byte-level BPE
//!   (`gpt2`, which covers Qwen and Llama-3 style vocabularies) is reconstructed; SentencePiece
//!   is refused by name rather than approximated, because an approximated tokeniser produces
//!   plausible-looking text that silently disagrees with the model.
//!
//! GGUF does not store the pre-tokeniser regex, only the *name* of its family in
//! `tokenizer.ggml.pre`, so [`PRE_TOKENIZER_REGEXES`] carries llama.cpp's table of them. Token
//! ids from the GGUF path were checked against `llama-tokenize` on a real Qwen2.5 GGUF and
//! agree exactly. A family missing from that table falls back to GPT-2's regex **and says so**,
//! because that fallback is precisely where boundaries drift.
//!
//! [`load_for_model`] still prefers a sibling `tokenizer.json`: the JSON is the model's own
//! serialised pre-tokeniser rather than a reconstruction of one.

use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer as HfTokenizer};

use crate::gguf::GgufMetadata;

/// Where a loaded tokeniser came from, for the startup banner and `/health`.
///
/// `docs/ux.md`: startup prints what it decided. "Which tokeniser did it pick" is exactly the
/// sort of thing that is invisible until it is wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    TokenizerJson(PathBuf),
    /// A GGUF, plus the `tokenizer.ggml.pre` family whose split regex was reconstructed.
    /// The family is part of the identity: two GGUFs with the same vocabulary and different
    /// `pre` values tokenise differently.
    Gguf {
        path: PathBuf,
        pre: String,
    },
    /// An in-memory fixture from [`crate::testing`]. Never produced by loading a model.
    Fixture(&'static str),
}

impl std::fmt::Display for TokenizerSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenizerJson(p) => write!(f, "tokenizer.json ({})", p.display()),
            Self::Gguf { path, pre } => {
                write!(f, "GGUF embedded vocabulary ({}, pre-tokeniser `{pre}`)", path.display())
            }
            Self::Fixture(name) => write!(f, "in-memory fixture `{name}` (NOT a real vocabulary)"),
        }
    }
}

/// A loaded tokeniser plus the special ids the serving layer needs.
pub struct Tokenizer {
    inner: HfTokenizer,
    source: TokenizerSource,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl Tokenizer {
    /// Load a serialised HF `tokenizer.json`.
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("reading tokenizer.json at {}", path.display()))?;
        // A tokenizer.json does not carry the ids as such; they are named in
        // tokenizer_config.json. Look the conventional names up in the vocabulary instead,
        // which works for every ChatML-family model and degrades to `None` rather than to a
        // wrong id.
        let bos = first_id(&inner, &["<|begin_of_text|>", "<s>", "<|startoftext|>"]);
        let eos = first_id(
            &inner,
            &["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "<|end_of_text|>"],
        );
        Ok(Self { inner, source: TokenizerSource::TokenizerJson(path.to_path_buf()), bos, eos })
    }

    /// Reconstruct a tokeniser from a GGUF's `tokenizer.ggml.*` metadata.
    pub fn from_gguf(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let md = GgufMetadata::from_path(path)?;
        Self::from_gguf_metadata(&md, path)
    }

    /// The half of [`Self::from_gguf`] that does not touch the filesystem, so it can be tested
    /// against a synthesised header.
    pub fn from_gguf_metadata(md: &GgufMetadata, path: &Path) -> anyhow::Result<Self> {
        let model = md.get_str("tokenizer.ggml.model").unwrap_or_default();
        if model != "gpt2" {
            bail!(
                "this GGUF carries a `{model}` tokeniser, and MoEArc can only rebuild byte-level \
                 BPE (`gpt2`) from GGUF metadata. Point --tokenizer at the model's \
                 tokenizer.json instead; approximating a `{model}` vocabulary would tokenise \
                 differently from the model that was trained on it."
            );
        }

        let tokens = md
            .get("tokenizer.ggml.tokens")
            .and_then(crate::gguf::Value::as_str_array)
            .ok_or_else(|| anyhow!("GGUF has no `tokenizer.ggml.tokens` array"))?;
        let merges = md
            .get("tokenizer.ggml.merges")
            .and_then(crate::gguf::Value::as_str_array)
            .ok_or_else(|| anyhow!("GGUF has no `tokenizer.ggml.merges` array"))?;

        // `Vocab` is `tokenizers`' own alias for an ahash-backed map; building a std `HashMap`
        // and converting would rehash 150k entries for nothing.
        let vocab: Vocab =
            tokens.iter().enumerate().map(|(i, t)| ((*t).to_string(), i as u32)).collect();
        let merges: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| m.split_once(' '))
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(true)
            .build()
            .map_err(|e| anyhow!("rebuilding BPE from GGUF metadata: {e}"))?;

        let pre = md.get_str("tokenizer.ggml.pre").unwrap_or("default").to_string();
        let mut inner = HfTokenizer::new(bpe);
        inner.with_pre_tokenizer(Some(gguf_pre_tokenizer(&pre)?));
        inner.with_decoder(Some(ByteLevelDecoder::default()));

        // Token types: 3 = CONTROL, 4 = USER_DEFINED. Both must survive pre-tokenisation
        // whole, or `<|im_start|>` is split into a dozen pieces and the chat template stops
        // meaning anything.
        if let Some(types) =
            md.get("tokenizer.ggml.token_type").and_then(crate::gguf::Value::as_array)
        {
            let specials: Vec<AddedToken> = types
                .iter()
                .enumerate()
                .filter(|(_, t)| matches!(t.as_u64(), Some(3 | 4)))
                .filter_map(|(i, _)| tokens.get(i))
                .map(|t| AddedToken::from((*t).to_string(), true))
                .collect();
            if !specials.is_empty() {
                inner
                    .add_special_tokens(specials)
                    .map_err(|e| anyhow!("registering the GGUF's control tokens: {e}"))?;
            }
        }

        let bos = md.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos = md.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        Ok(Self {
            inner,
            source: TokenizerSource::Gguf { path: path.to_path_buf(), pre },
            bos,
            eos,
        })
    }

    /// Wrap an already-built `tokenizers` instance. Used by [`crate::testing`] to construct a
    /// fixture without a file on disk; loading paths go through the constructors above.
    pub(crate) fn from_parts(
        inner: HfTokenizer,
        source: TokenizerSource,
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> Self {
        Self { inner, source, bos, eos }
    }

    pub fn source(&self) -> &TokenizerSource {
        &self.source
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn bos_token(&self) -> Option<String> {
        self.bos.and_then(|id| self.inner.id_to_token(id))
    }

    pub fn eos_token(&self) -> Option<String> {
        self.eos.and_then(|id| self.inner.id_to_token(id))
    }

    /// Vocabulary size *including* added tokens — the width of a logits vector.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode text to ids.
    ///
    /// `add_special_tokens` is `false` everywhere in this crate: a rendered chat template
    /// already contains its BOS and its turn markers as *text*, so letting the tokeniser add
    /// another set produces a doubled prefix. This is the single most common way to get
    /// subtly-degraded output from an otherwise correct server.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow!("tokenising: {e}"))
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow!("detokenising: {e}"))
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("source", &self.source)
            .field("vocab_size", &self.vocab_size())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish()
    }
}

/// Split regexes by `tokenizer.ggml.pre` family, transcribed from llama.cpp's
/// `llama-vocab.cpp` (which in turn transcribes them from each model's `tokenizer.json`).
///
/// 🔴 **This table is why GGUF tokenisation is correct rather than approximately correct.**
/// GGUF stores the vocabulary and the merge table but *not* the pre-tokeniser regex, only the
/// name of its family. Reconstructing GPT-2's regex for every byte-level vocabulary is the
/// obvious shortcut and it is wrong: measured against `llama-tokenize` on a real Qwen2.5 GGUF,
/// `"mixture-of-experts"` came out as 6 tokens instead of 4, because GPT-2's regex splits a
/// leading hyphen off a word and Qwen2's `[^\r\n\p{L}\p{N}]?\p{L}+` keeps it attached.
/// Five of six test strings agreed; the one that did not would have silently degraded every
/// generation containing a hyphenated word.
const PRE_TOKENIZER_REGEXES: &[(&[&str], &str)] = &[
    (
        &["qwen2", "stablelm2", "hunyuan", "solar-open"],
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    (
        &["qwen35"],
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    (
        &["llama-bpe", "llama3", "smaug-bpe", "dbrx"],
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    (
        &["gpt-2", "mpt", "olmo", "jais", "default"],
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    ),
];

/// Look up the split regex for a `tokenizer.ggml.pre` value.
fn split_regex_for(pre: &str) -> Option<&'static str> {
    PRE_TOKENIZER_REGEXES.iter().find(|(names, _)| names.contains(&pre)).map(|(_, re)| *re)
}

/// Build the pre-tokeniser for a byte-level GGUF vocabulary: split on the family's regex,
/// then map bytes to the byte-level alphabet. This is the same two-stage `Sequence` a Hugging
/// Face `tokenizer.json` serialises for these models.
fn gguf_pre_tokenizer(pre: &str) -> anyhow::Result<PreTokenizerWrapper> {
    let regex = split_regex_for(pre).unwrap_or_else(|| {
        // Named, not swallowed: an unknown family still tokenises, but the operator is told
        // which one so they can pass a tokenizer.json if the output looks off.
        tracing::warn!(
            pre,
            "unknown GGUF pre-tokeniser family; falling back to the GPT-2 split regex. \
             Token boundaries may differ from llama.cpp — prefer the model's tokenizer.json."
        );
        split_regex_for("gpt-2").expect("the gpt-2 family is in the table")
    });
    let split =
        Split::new(SplitPattern::Regex(regex.to_string()), SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| anyhow!("compiling the `{pre}` pre-tokeniser regex: {e}"))?;
    // `use_regex = false`: the split above already did that job, and letting ByteLevel apply
    // GPT-2's regex on top would re-split what the family regex deliberately kept together.
    let byte_level = ByteLevel::new(false, false, false);
    Ok(PreTokenizerWrapper::Sequence(PreTokSequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ])))
}

fn first_id(t: &HfTokenizer, names: &[&str]) -> Option<u32> {
    names.iter().find_map(|n| t.token_to_id(n))
}

/// Load the best tokeniser available for a model path, preferring `tokenizer.json`.
///
/// `path` may be a GGUF file or a directory. A GGUF sitting beside a `tokenizer.json` uses the
/// JSON, for the reason in the module docs: the JSON carries the pre-tokeniser, the GGUF does
/// not.
pub fn load_for_model(path: &Path) -> anyhow::Result<Tokenizer> {
    if path.is_dir() {
        let json = path.join("tokenizer.json");
        if json.is_file() {
            return Tokenizer::from_tokenizer_json(&json);
        }
        bail!("{} contains no tokenizer.json", path.display());
    }
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
        return Tokenizer::from_tokenizer_json(path);
    }
    if let Some(dir) = path.parent() {
        let json = dir.join("tokenizer.json");
        if json.is_file() {
            return Tokenizer::from_tokenizer_json(&json);
        }
    }
    Tokenizer::from_gguf(path)
}

/// Incremental detokenisation for streaming.
///
/// A byte-level BPE token is not a string: one token can be a fragment of a UTF-8 sequence, and
/// `▁`-style prefixes only resolve against their neighbours. Decoding each token alone and
/// concatenating produces replacement characters mid-word — visible to anyone streaming
/// non-ASCII, which is most of the world.
///
/// So the whole sequence is decoded each step and only the new *suffix* is emitted. That is
/// quadratic in the completion length, and deliberately so at this size: a 4k-token completion
/// costs a few milliseconds of string work against seconds of inference. If that ever stops
/// being true the fix is a windowed decode, not per-token decoding.
#[derive(Debug, Default)]
pub struct IncrementalDecoder {
    ids: Vec<u32>,
    emitted: String,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a token; return the text that became newly available, if any.
    pub fn push(&mut self, tk: &Tokenizer, id: u32) -> anyhow::Result<String> {
        self.ids.push(id);
        let full = tk.decode(&self.ids, true)?;
        // An incomplete UTF-8 sequence decodes to U+FFFD, which would be emitted and then
        // never corrected. Hold it back until the next token completes the character.
        if full.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        let delta =
            full.strip_prefix(self.emitted.as_str()).map(str::to_string).unwrap_or_else(|| {
                // The decoded prefix changed rather than grew — a merge rewrote earlier text.
                // Rare, but emitting a wrong delta is worse than emitting the whole tail, and the
                // client concatenates either way.
                full.chars().skip(self.emitted.chars().count()).collect()
            });
        self.emitted = full;
        Ok(delta)
    }

    /// Everything emitted so far.
    pub fn text(&self) -> &str {
        &self.emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real vocabularies are hundreds of megabytes and cannot be committed to a public repo,
    /// so the round-trip tests below run only where one happens to exist. `MOEARC_TEST_TOKENIZER`
    /// points at it; without it these skip rather than fail, and the CLI transcript in the
    /// crate docs is the standing evidence.
    fn test_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var_os("MOEARC_TEST_TOKENIZER")?;
        Some(Tokenizer::from_tokenizer_json(PathBuf::from(path)).expect("loading test tokenizer"))
    }

    const ROUND_TRIP_CASES: &[&str] = &[
        "The quick brown fox jumps over the lazy dog.",
        "MoEArc runs mixture-of-experts models on Intel Arc.",
        "fn main() { println!(\"hello, world\"); }",
        "Ünïcödé, emoji 🚀🔥, and CJK 混合专家模型.",
        "  leading and trailing whitespace  ",
        "line one\nline two\n\tindented",
    ];

    #[test]
    fn tokenizer_json_round_trips() {
        let Some(tk) = test_tokenizer() else { return };
        // Printed so `cargo test -- --nocapture` is usable as evidence: an assertion that
        // passes tells you nothing about *what* was encoded.
        println!("tokeniser: {:?}", tk);
        for case in ROUND_TRIP_CASES {
            let ids = tk.encode(case, false).unwrap();
            assert!(!ids.is_empty(), "{case:?} encoded to nothing");
            let back = tk.decode(&ids, false).unwrap();
            println!(
                "  in  {case:?}\n  ids {ids:?}\n  out {back:?}  -> {}",
                if &back == case { "IDENTICAL" } else { "CHANGED" }
            );
            assert_eq!(&back, case, "round trip changed the text");
        }
    }

    /// The GGUF reconstruction path, against a real model file. Set `MOEARC_TEST_GGUF` to a
    /// `.gguf` with a `gpt2`-style vocabulary; skips otherwise, for the same reason as above.
    #[test]
    fn gguf_vocabulary_round_trips() {
        let Some(path) = std::env::var_os("MOEARC_TEST_GGUF") else { return };
        let tk = Tokenizer::from_gguf(PathBuf::from(path)).expect("rebuilding from GGUF");
        println!("tokeniser: {tk:?}");
        for case in ROUND_TRIP_CASES {
            let ids = tk.encode(case, false).unwrap();
            let back = tk.decode(&ids, false).unwrap();
            println!(
                "  in  {case:?}\n  ids {ids:?}\n  out {back:?}  -> {}",
                if &back == case { "IDENTICAL" } else { "CHANGED" }
            );
            assert_eq!(&back, case, "GGUF round trip changed the text");
        }
    }

    /// Conformance against llama.cpp, on the same file.
    ///
    /// 🔴 These ids are not invented: they are `llama-tokenize --no-bos --ids` output for
    /// `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, the reference implementation reading the same
    /// vocabulary and merge table out of the same bytes. Round-tripping proves the tokeniser
    /// is self-consistent; only this proves it agrees with the ecosystem. Set
    /// `MOEARC_TEST_GGUF_QWEN25` to that file to run it.
    ///
    /// The second case is the one that caught the pre-tokeniser bug: with GPT-2's regex,
    /// `mixture-of-experts` tokenised as `... 12, 1055, 12, 4580 ...` — six tokens where
    /// llama.cpp produces four.
    #[test]
    fn gguf_token_ids_match_llama_cpp() {
        const EXPECTED: &[(&str, &[u32])] = &[
            (
                "The quick brown fox jumps over the lazy dog.",
                &[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13],
            ),
            (
                "MoEArc runs mixture-of-experts models on Intel Arc.",
                &[25612, 19112, 1287, 8473, 20980, 8668, 18376, 15546, 4119, 389, 15611, 19689, 13],
            ),
            (
                "fn main() { println!(\"hello, world\"); }",
                &[8822, 1887, 368, 314, 13751, 17223, 14990, 11, 1879, 5038, 335],
            ),
            (
                "Ünïcödé, emoji \u{1F680}\u{1F525}, and CJK 混合专家模型.",
                &[
                    52491, 77, 37572, 66, 2956, 128505, 11, 42365, 11162, 248, 222, 144670, 11,
                    323, 356, 34070, 6567, 115, 115, 39762, 101057, 104949, 13,
                ],
            ),
            ("  leading and trailing whitespace  ", &[220, 6388, 323, 27748, 36372, 256]),
            ("line one\nline two\n\tindented", &[1056, 825, 198, 1056, 1378, 198, 197, 484, 15864]),
        ];

        let Some(path) = std::env::var_os("MOEARC_TEST_GGUF_QWEN25") else { return };
        let tk = Tokenizer::from_gguf(PathBuf::from(path)).expect("rebuilding from GGUF");
        assert_eq!(tk.vocab_size(), 151_936);
        assert_eq!(tk.eos_id(), Some(151_645));
        for (text, ids) in EXPECTED {
            assert_eq!(
                &tk.encode(text, false).unwrap(),
                ids,
                "ids differ from llama.cpp for {text:?}"
            );
        }
    }

    #[test]
    fn every_pre_tokenizer_family_regex_compiles() {
        // A typo in the table would otherwise surface only when someone loads that family's
        // GGUF — i.e. for a model we do not have on hand.
        for (names, _) in PRE_TOKENIZER_REGEXES {
            for name in *names {
                gguf_pre_tokenizer(name).unwrap_or_else(|e| panic!("family `{name}`: {e:#}"));
            }
        }
    }

    #[test]
    fn an_unknown_pre_tokenizer_family_falls_back_rather_than_failing() {
        assert!(split_regex_for("no-such-family").is_none());
        assert!(gguf_pre_tokenizer("no-such-family").is_ok());
    }

    #[test]
    fn incremental_decode_matches_a_whole_decode() {
        let Some(tk) = test_tokenizer() else { return };
        for case in ROUND_TRIP_CASES {
            let ids = tk.encode(case, false).unwrap();
            let mut dec = IncrementalDecoder::new();
            let mut streamed = String::new();
            for id in &ids {
                streamed.push_str(&dec.push(&tk, *id).unwrap());
            }
            assert_eq!(
                streamed,
                tk.decode(&ids, true).unwrap(),
                "streaming lost or duplicated text"
            );
        }
    }

    #[test]
    fn gguf_refuses_sentencepiece_by_name() {
        use crate::gguf::Value;
        let mut kv = std::collections::HashMap::new();
        kv.insert("tokenizer.ggml.model".to_string(), Value::String("llama".to_string()));
        let md = GgufMetadata { version: 3, tensor_count: 0, kv };
        let err = Tokenizer::from_gguf_metadata(&md, Path::new("x.gguf")).unwrap_err().to_string();
        assert!(err.contains("llama"), "the message must name what it found: {err}");
        assert!(err.contains("tokenizer.json"), "and say what to do instead: {err}");
    }

    #[test]
    fn tokenizer_is_shareable_across_threads() {
        // The router holds one behind an `Arc` for every concurrent request. If this stops
        // holding it fails as a wall of trait errors in the handlers; here it fails in one line.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tokenizer>();
    }
}
