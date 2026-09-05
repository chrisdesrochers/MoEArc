//! What a running server holds, and the one place the engine is plugged in.

use std::sync::Arc;

use crate::chat::{ChatTemplate, TemplateContext};
use crate::generate::SharedGenerator;
use crate::sampling::SamplingParams;
use crate::tokenize::Tokenizer;

/// Everything a request handler needs, shared behind an `Arc` by axum.
///
/// # 🔴 The integration point
///
/// `generator` is the *only* place the inference engine enters this crate. Today the binary
/// constructs it as `Arc::new(EchoGenerator::new(tokenizer.vocab_size()))`; wiring the real
/// engine means changing that expression and nothing else. No handler, no template, no encoder
/// and no test in this crate names a concrete generator type — see
/// [`crate::generate`] for the contract the replacement has to meet.
pub struct ServerState {
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
    pub generator: SharedGenerator,
    /// The id reported by `GET /v1/models` and echoed in every response.
    pub model_id: String,
    /// Sampling used when a request specifies nothing.
    pub defaults: SamplingParams,
}

impl ServerState {
    /// Assemble a server.
    ///
    /// `stop_tokens` are seeded from the tokeniser's EOS here rather than per request, because
    /// they are a property of the model and a client should never have to know the id.
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        template: Arc<ChatTemplate>,
        generator: SharedGenerator,
        model_id: impl Into<String>,
        defaults: SamplingParams,
    ) -> Self {
        let mut defaults = defaults;
        if let Some(eos) = tokenizer.eos_id()
            && !defaults.stop_tokens.contains(&eos)
        {
            defaults.stop_tokens.push(eos);
        }
        Self { tokenizer, template, generator, model_id: model_id.into(), defaults }
    }

    /// The special tokens the chat template may interpolate.
    pub fn template_context(&self) -> TemplateContext {
        TemplateContext {
            bos_token: self.tokenizer.bos_token(),
            eos_token: self.tokenizer.eos_token(),
        }
    }

    /// A vocabulary mismatch between tokeniser and generator, if there is one.
    ///
    /// Returned rather than logged so the caller decides: the binary prints it as a startup
    /// warning, and it is exactly the class of misconfiguration that otherwise surfaces as
    /// fluent-looking nonsense hours later.
    pub fn vocab_mismatch(&self) -> Option<String> {
        let (tk, logits) = (self.tokenizer.vocab_size(), self.generator.vocab_size());
        (tk != logits).then(|| {
            format!(
                "tokeniser vocabulary is {tk} tokens but the generator produces {logits} logits — \
                 the tokeniser and the weights are from different models, and generations will \
                 be nonsense"
            )
        })
    }

    /// One line for the startup banner, per `docs/ux.md`: startup prints what it decided.
    pub fn banner(&self) -> String {
        let mut s = format!(
            "model      {}\ntokeniser  {}\ntemplate   {}\ngenerator  {}",
            self.model_id,
            self.tokenizer.source(),
            self.template.source(),
            self.generator.name(),
        );
        if self.generator.is_stub() {
            s.push_str(
                "\n\n\u{26a0}  The generator is a STUB. The HTTP path, tokeniser, chat template \
                 and sampler\n   are all real; the text is not model output. No weights are loaded.",
            );
        }
        if let Some(w) = self.vocab_mismatch() {
            s.push_str(&format!("\n\n\u{26a0}  {w}"));
        }
        s
    }
}

/// Shared handle, as axum's handlers see it.
pub type AppState = Arc<ServerState>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::generate::{EchoGenerator, GenerationStats, Generator};

    struct FixedVocab(usize);
    impl Generator for FixedVocab {
        fn generate(
            &self,
            _p: &[u32],
            _s: &SamplingParams,
            _cb: &mut dyn FnMut(u32) -> bool,
        ) -> anyhow::Result<GenerationStats> {
            Ok(GenerationStats::default())
        }
        fn vocab_size(&self) -> usize {
            self.0
        }
        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    fn state_with(generator: SharedGenerator) -> ServerState {
        let tk = Arc::new(crate::testing::tiny_tokenizer());
        ServerState::new(
            tk,
            Arc::new(ChatTemplate::chatml()),
            generator,
            "test-model",
            SamplingParams::default(),
        )
    }

    #[test]
    fn eos_becomes_a_default_stop_token() {
        let tk = Arc::new(crate::testing::tiny_tokenizer());
        let eos = tk.eos_id().expect("the fixture defines an EOS");
        let s = ServerState::new(
            tk,
            Arc::new(ChatTemplate::chatml()),
            Arc::new(EchoGenerator::new(64)),
            "m",
            SamplingParams::default(),
        );
        assert!(s.defaults.stop_tokens.contains(&eos));
    }

    #[test]
    fn a_matching_vocabulary_reports_nothing() {
        let n = crate::testing::tiny_tokenizer().vocab_size();
        assert!(state_with(Arc::new(FixedVocab(n))).vocab_mismatch().is_none());
    }

    #[test]
    fn a_mismatched_vocabulary_is_named_in_full() {
        let w = state_with(Arc::new(FixedVocab(999_999))).vocab_mismatch().unwrap();
        assert!(w.contains("999999"), "{w}");
    }

    #[test]
    fn the_banner_says_out_loud_that_the_generator_is_a_stub() {
        let s =
            state_with(Arc::new(EchoGenerator::new(crate::testing::tiny_tokenizer().vocab_size())));
        let b = s.banner();
        assert!(b.contains("STUB"), "{b}");
        assert!(b.contains("test-model"), "{b}");
    }
}
