//! Rendering a chat conversation into the exact prompt string a model expects.
//!
//! Instruction-tuned models are trained on one specific turn format, and getting it wrong does
//! not error — it degrades. The model still answers, just worse, and there is no signal saying
//! so. That is why the template is *the model's own*: GGUF carries it in metadata under
//! `tokenizer.chat_template`, and Hugging Face repos carry it in `tokenizer_config.json` or a
//! `chat_template.jinja` beside it. We render the file, we do not reimplement the format.
//!
//! `minijinja` is the renderer. Jinja's chat-template dialect leans on a handful of Python
//! idioms (`.strip()`, `.startswith()`, `namespace()`, `raise_exception()`) that a plain Jinja
//! engine does not have, so those are registered explicitly below; a template that reaches for
//! something still missing fails loudly at render time with the template's own error text,
//! which is far better than a prompt that is quietly malformed.

use std::path::Path;

use anyhow::{Context, anyhow};
use minijinja::{Environment, Error, ErrorKind, Value, context};

use crate::gguf::GgufMetadata;

/// The ChatML fallback, used only when a model ships no template at all.
///
/// ChatML is the right default because it is what the MoE models MoEArc targets (Qwen3,
/// gpt-oss, and derivatives) already use. It is still a *guess*, and [`ChatTemplate::is_fallback`]
/// exists so the startup banner can say so out loud rather than let a user assume the model's
/// own format was found.
const CHATML: &str = "\
{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}";

/// Where a chat template came from. Reported at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateSource {
    /// `tokenizer.chat_template` in a GGUF header.
    Gguf,
    /// A `chat_template.jinja` file.
    JinjaFile,
    /// The `chat_template` field of `tokenizer_config.json`.
    TokenizerConfig,
    /// No template was found; [`CHATML`] is in use.
    FallbackChatMl,
}

impl std::fmt::Display for TemplateSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Gguf => "the model's own template, from GGUF metadata",
            Self::JinjaFile => "the model's own template, from chat_template.jinja",
            Self::TokenizerConfig => "the model's own template, from tokenizer_config.json",
            Self::FallbackChatMl => "the built-in ChatML fallback (this model shipped no template)",
        })
    }
}

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    source: TemplateSource,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").field("source", &self.source).finish()
    }
}

/// Everything a template may ask about the model, threaded through from the tokeniser.
#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

impl ChatTemplate {
    /// Compile a Jinja source string.
    pub fn from_source(src: &str, source: TemplateSource) -> anyhow::Result<Self> {
        let mut env = Environment::new();
        install_python_compat(&mut env);
        env.add_template_owned("chat", src.to_string())
            .map_err(|e| anyhow!("{e:#}"))
            .context("compiling the model's chat template")?;
        Ok(Self { env, source })
    }

    /// The ChatML fallback.
    pub fn chatml() -> Self {
        Self::from_source(CHATML, TemplateSource::FallbackChatMl)
            .expect("the built-in ChatML template must compile")
    }

    /// Find a model's template: GGUF metadata, then a sibling `chat_template.jinja`, then
    /// `tokenizer_config.json`, then ChatML.
    ///
    /// `path` may be a GGUF file or a model directory. Ordering is by specificity, not
    /// convenience: a GGUF's embedded template is the one that shipped with *those weights*.
    pub fn discover(path: &Path) -> anyhow::Result<Self> {
        if path.is_file()
            && !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json"))
            && let Ok(md) = GgufMetadata::from_path(path)
            && let Some(src) = md.chat_template()
        {
            return Self::from_source(src, TemplateSource::Gguf);
        }
        let dir = if path.is_dir() { path } else { path.parent().unwrap_or(Path::new(".")) };

        let jinja = dir.join("chat_template.jinja");
        if jinja.is_file() {
            let src = std::fs::read_to_string(&jinja)
                .with_context(|| format!("reading {}", jinja.display()))?;
            return Self::from_source(&src, TemplateSource::JinjaFile);
        }

        let cfg = dir.join("tokenizer_config.json");
        if cfg.is_file()
            && let Ok(text) = std::fs::read_to_string(&cfg)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(src) = json.get("chat_template").and_then(serde_json::Value::as_str)
        {
            return Self::from_source(src, TemplateSource::TokenizerConfig);
        }

        Ok(Self::chatml())
    }

    pub fn source(&self) -> &TemplateSource {
        &self.source
    }

    pub fn is_fallback(&self) -> bool {
        self.source == TemplateSource::FallbackChatMl
    }

    /// Render a conversation to the prompt string.
    ///
    /// `messages` is passed as raw JSON so that OpenAI's two content shapes — a string, or an
    /// array of typed parts — both reach the template exactly as the client sent them.
    /// Normalising to a string here would silently drop the image parts that multimodal
    /// templates are written to handle.
    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        add_generation_prompt: bool,
        ctx: &TemplateContext,
    ) -> anyhow::Result<String> {
        let tmpl = self.env.get_template("chat").map_err(|e| anyhow!("{e:#}"))?;
        tmpl.render(context! {
            messages => Value::from_serialize(messages),
            tools => tools.map(Value::from_serialize),
            add_generation_prompt => add_generation_prompt,
            bos_token => ctx.bos_token.as_deref(),
            eos_token => ctx.eos_token.as_deref(),
            // Some templates branch on these; absent means "no thinking block", which is the
            // conservative default for a server that has to be right for every client.
            enable_thinking => false,
            add_vision_id => false,
        })
        .map_err(|e| {
            // minijinja's `{e:#}` carries the template line and the failing expression. A
            // template error is a *configuration* error the operator has to see, so the detail
            // is preserved rather than flattened to "render failed".
            anyhow!("rendering the chat template failed: {e:#}")
        })
    }
}

/// Register the Python-flavoured helpers that chat templates in the wild depend on.
fn install_python_compat(env: &mut Environment<'static>) {
    // `raise_exception` is how a template rejects an input it cannot represent — e.g. a system
    // message containing an image. Mapping it onto a render error means the client gets a 400
    // naming the model's own reason, which is precisely the legible failure docs/ux.md asks for.
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });

    // Llama 3.x templates stamp a date into the system prompt. Implemented here rather than
    // pulling in a date library: the server needs six format specifiers, not a calendar.
    env.add_function("strftime_now", |fmt: String| -> Result<Value, Error> {
        Ok(Value::from(strftime_now(&fmt)))
    });

    // Python string and dict methods (`.strip()`, `.startswith()`, `.items()`, `.split()`).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
}

/// Days since the Unix epoch → (year, month, day). Howard Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn strftime_now(fmt: &str) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let tod = secs.rem_euclid(86_400);
    let name = MONTHS[(m - 1) as usize];

    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('B') => out.push_str(name),
            Some('b') => out.push_str(&name[..3]),
            Some('H') => out.push_str(&format!("{:02}", tod / 3600)),
            Some('M') => out.push_str(&format!("{:02}", (tod % 3600) / 60)),
            Some('S') => out.push_str(&format!("{:02}", tod % 60)),
            Some('%') => out.push('%'),
            // An unknown specifier is echoed rather than dropped: a template that asked for
            // something we do not implement should show that in its output, not silently lose it.
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs() -> serde_json::Value {
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hello"},
        ])
    }

    #[test]
    fn chatml_fallback_renders_the_expected_prompt() {
        let t = ChatTemplate::chatml();
        assert!(t.is_fallback());
        let out = t.render(&msgs(), None, true, &TemplateContext::default()).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn generation_prompt_is_optional() {
        let t = ChatTemplate::chatml();
        let out = t.render(&msgs(), None, false, &TemplateContext::default()).unwrap();
        assert!(out.ends_with("<|im_end|>\n"), "{out:?}");
    }

    #[test]
    fn bos_and_eos_reach_the_template() {
        let t = ChatTemplate::from_source("{{ bos_token }}|{{ eos_token }}", TemplateSource::Gguf)
            .unwrap();
        let ctx = TemplateContext { bos_token: Some("<s>".into()), eos_token: Some("</s>".into()) };
        assert_eq!(t.render(&json!([]), None, false, &ctx).unwrap(), "<s>|</s>");
    }

    #[test]
    fn raise_exception_becomes_a_render_error_carrying_the_message() {
        let t = ChatTemplate::from_source(
            "{{ raise_exception('System message cannot contain images.') }}",
            TemplateSource::Gguf,
        )
        .unwrap();
        let err = t.render(&json!([]), None, false, &TemplateContext::default()).unwrap_err();
        assert!(err.to_string().contains("cannot contain images"), "{err}");
    }

    #[test]
    fn python_string_and_dict_methods_work() {
        // These are not Jinja; they are the Python idioms real templates use. Without the
        // pycompat callback every Qwen and Llama template fails on the first `.strip()`.
        let t = ChatTemplate::from_source(
            "{{ '  x  '.strip() }}{{ 'abc'.startswith('a') }}{{ messages[0].content.split('-')[1] }}",
            TemplateSource::Gguf,
        )
        .unwrap();
        let out = t
            .render(
                &json!([{"role": "user", "content": "a-b"}]),
                None,
                false,
                &TemplateContext::default(),
            )
            .unwrap();
        // `True`, not `true`: minijinja renders booleans the way Jinja2 does, which is also
        // what the templates were written against.
        assert_eq!(out, "xTrueb");
    }

    #[test]
    fn namespace_and_loop_controls_are_available() {
        // Every recent Qwen template opens with `namespace(value=0)`, and several use `break`.
        let t = ChatTemplate::from_source(
            "{%- set ns = namespace(n=0) %}\
             {%- for m in messages %}{% if m.role == 'skip' %}{% continue %}{% endif %}\
             {%- set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}",
            TemplateSource::Gguf,
        )
        .unwrap();
        let out = t
            .render(
                &json!([{"role":"user"},{"role":"skip"},{"role":"user"}]),
                None,
                false,
                &TemplateContext::default(),
            )
            .unwrap();
        assert_eq!(out, "2");
    }

    #[test]
    fn structured_content_parts_survive_to_the_template() {
        let t = ChatTemplate::from_source(
            "{% for part in messages[0].content %}{{ part.type }}:{% endfor %}",
            TemplateSource::Gguf,
        )
        .unwrap();
        let m =
            json!([{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url"}]}]);
        assert_eq!(
            t.render(&m, None, false, &TemplateContext::default()).unwrap(),
            "text:image_url:"
        );
    }

    #[test]
    fn tools_are_passed_through() {
        let t = ChatTemplate::from_source("{{ tools[0].function.name }}", TemplateSource::Gguf)
            .unwrap();
        let tools = json!([{"type":"function","function":{"name":"get_weather"}}]);
        assert_eq!(
            t.render(&json!([]), Some(&tools), false, &TemplateContext::default()).unwrap(),
            "get_weather"
        );
    }

    #[test]
    fn a_broken_template_fails_at_compile_time_with_a_location() {
        let err = ChatTemplate::from_source("{% for x in %}", TemplateSource::Gguf).unwrap_err();
        assert!(err.to_string().contains("chat template"), "{err:#}");
    }

    #[test]
    fn strftime_covers_the_specifiers_llama_templates_use() {
        assert_eq!(strftime_now("%%"), "%");
        assert_eq!(strftime_now("%q"), "%q");
        let d = strftime_now("%d %b %Y");
        let parts: Vec<&str> = d.split(' ').collect();
        assert_eq!(parts.len(), 3, "{d}");
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 3);
        assert!(parts[2].parse::<i32>().unwrap() >= 2026, "{d}");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // a leap year boundary
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
