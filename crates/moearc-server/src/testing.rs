//! In-memory fixtures, so the serving layer can be exercised without a model on disk.
//!
//! This mirrors `moearc-cli`'s `Sources::stub`: the shapes are real, the contents are not, and
//! [`crate::tokenize::TokenizerSource::Fixture`] says so wherever they surface. Shipping this in
//! the library rather than behind `#[cfg(test)]` is deliberate — the integration test drives a
//! *real* server over a *real* socket, and it needs the same fixture the unit tests use.

use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

use crate::tokenize::{Tokenizer, TokenizerSource};

/// The ChatML control tokens the fallback template emits, appended after the byte alphabet.
pub const FIXTURE_SPECIALS: [&str; 3] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"];

/// A byte-level BPE over the 256-symbol byte alphabet with **no merges**.
///
/// Every input therefore tokenises to one token per byte, which makes it a genuine tokeniser —
/// real byte-level encoding, real added-token handling, exact round-trip on arbitrary UTF-8 —
/// with a 259-entry vocabulary and no file to download. A `WordLevel` fixture would have been
/// smaller and would not round-trip whitespace, which is exactly the property the streaming
/// decoder has to get right.
pub fn tiny_tokenizer() -> Tokenizer {
    let vocab: Vocab = ByteLevel::alphabet()
        .into_iter()
        .collect::<std::collections::BTreeSet<char>>() // sorted, so ids are stable across runs
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("the fixture vocabulary must build");

    let mut inner = HfTokenizer::new(bpe);
    inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    inner
        .add_special_tokens(
            FIXTURE_SPECIALS
                .iter()
                .map(|t| AddedToken::from((*t).to_string(), true))
                .collect::<Vec<_>>(),
        )
        .expect("the fixture's special tokens must register");

    let eos = inner.token_to_id("<|im_end|>");
    Tokenizer::from_parts(inner, TokenizerSource::Fixture("byte-level, no merges"), None, eos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_round_trips_arbitrary_text() {
        let tk = tiny_tokenizer();
        for case in ["Hello, world!", "  spaces  ", "emoji 🚀", "líne\nbreak\ttab"] {
            let ids = tk.encode(case, false).unwrap();
            assert_eq!(tk.decode(&ids, false).unwrap(), case, "{case:?}");
        }
    }

    #[test]
    fn special_tokens_stay_whole() {
        let tk = tiny_tokenizer();
        let ids = tk.encode("<|im_start|>user", false).unwrap();
        assert_eq!(ids[0], tk.token_to_id("<|im_start|>").unwrap());
        assert_eq!(ids.len(), 1 + "user".len());
    }

    #[test]
    fn it_declares_itself_a_fixture() {
        assert!(tiny_tokenizer().source().to_string().contains("NOT a real vocabulary"));
    }
}
