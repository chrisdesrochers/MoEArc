//! A metadata-only GGUF reader.
//!
//! GGUF puts a key/value block at the head of the file, before the tensor index and long
//! before the weights. Everything the *serving* layer needs — the vocabulary, the merge table,
//! the special token ids, the chat template — lives in that block. So this reads the header
//! and stops: opening a 22 GB model to answer "what is the EOS token" should cost a few
//! megabytes, not a memory map of the whole file.
//!
//! Tensor data is deliberately out of scope. `moearc-model` owns weights; this crate must not
//! grow a second, competing loader.
//!
//! Layout (GGUF v2/v3, little-endian): magic `GGUF`, `u32` version, `u64` tensor count,
//! `u64` kv count, then the kv pairs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use thiserror::Error;

const MAGIC: &[u8; 4] = b"GGUF";
/// Refuse a vocabulary larger than this. A corrupt length field otherwise asks for an
/// arbitrary allocation before anything has had a chance to notice the file is wrong.
const MAX_ARRAY_LEN: u64 = 8_000_000;
/// Same reasoning for a single string.
const MAX_STRING_LEN: u64 = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("{path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("not a GGUF file — expected the magic bytes `GGUF`, found {found:02x?}")]
    BadMagic { found: [u8; 4] },
    #[error("GGUF version {0} is not supported — this reader handles versions 2 and 3")]
    BadVersion(u32),
    #[error("unknown GGUF value type {0} at key `{1}` — the file is newer than this reader")]
    UnknownValueType(u32, String),
    #[error("GGUF header declares an implausible length ({0}); the file is truncated or corrupt")]
    ImplausibleLength(u64),
    #[error("GGUF string at key `{0}` is not valid UTF-8")]
    BadUtf8(String),
}

/// One metadata value. Numeric widths are collapsed to the widest signed/unsigned/float form:
/// nothing downstream cares whether an expert count was stored as `u32` or `u64`, and keeping
/// twelve variants would push that indifference onto every caller.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// An array of strings, or `None` if any element is not one.
    pub fn as_str_array(&self) -> Option<Vec<&str>> {
        self.as_array()?.iter().map(Self::as_str).collect()
    }
}

/// The parsed header.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv: HashMap<String, Value>,
}

impl GgufMetadata {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// `general.architecture`, e.g. `qwen2`.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// The Jinja chat template the model ships, if any.
    pub fn chat_template(&self) -> Option<&str> {
        self.get_str("tokenizer.chat_template")
    }

    /// Read the header of `path`, leaving the rest of the file untouched.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|source| GgufError::Io { path: path.display().to_string(), source })?;
        // 4 MiB: a 150k-entry vocabulary and its merge table are a few MB of strings, and
        // reading them a `u64` at a time through an unbuffered `File` is thousands of times
        // slower for no benefit.
        Self::read(BufReader::with_capacity(4 << 20, file)).map_err(|e| match e {
            GgufError::Io { source, .. } => {
                GgufError::Io { path: path.display().to_string(), source }
            }
            other => other,
        })
    }

    fn read(mut r: impl Read) -> Result<Self, GgufError> {
        let mut magic = [0u8; 4];
        read_exact(&mut r, &mut magic)?;
        if &magic != MAGIC {
            return Err(GgufError::BadMagic { found: magic });
        }
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::BadVersion(version));
        }
        let tensor_count = read_u64(&mut r)?;
        let kv_count = read_u64(&mut r)?;
        if kv_count > MAX_ARRAY_LEN {
            return Err(GgufError::ImplausibleLength(kv_count));
        }

        let mut kv = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(&mut r, "<key>")?;
            let ty = read_u32(&mut r)?;
            let value = read_value(&mut r, ty, &key)?;
            kv.insert(key, value);
        }
        Ok(Self { version, tensor_count, kv })
    }
}

fn read_exact(r: &mut impl Read, buf: &mut [u8]) -> Result<(), GgufError> {
    r.read_exact(buf).map_err(|source| GgufError::Io { path: String::new(), source })
}

macro_rules! read_le {
    ($name:ident, $ty:ty, $n:literal) => {
        fn $name(r: &mut impl Read) -> Result<$ty, GgufError> {
            let mut b = [0u8; $n];
            read_exact(r, &mut b)?;
            Ok(<$ty>::from_le_bytes(b))
        }
    };
}

read_le!(read_u8, u8, 1);
read_le!(read_i8, i8, 1);
read_le!(read_u16, u16, 2);
read_le!(read_i16, i16, 2);
read_le!(read_u32, u32, 4);
read_le!(read_i32, i32, 4);
read_le!(read_f32, f32, 4);
read_le!(read_u64, u64, 8);
read_le!(read_i64, i64, 8);
read_le!(read_f64, f64, 8);

fn read_string(r: &mut impl Read, key: &str) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    if len > MAX_STRING_LEN {
        return Err(GgufError::ImplausibleLength(len));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(r, &mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::BadUtf8(key.to_string()))
}

fn read_value(r: &mut impl Read, ty: u32, key: &str) -> Result<Value, GgufError> {
    Ok(match ty {
        0 => Value::U64(u64::from(read_u8(r)?)),
        1 => Value::I64(i64::from(read_i8(r)?)),
        2 => Value::U64(u64::from(read_u16(r)?)),
        3 => Value::I64(i64::from(read_i16(r)?)),
        4 => Value::U64(u64::from(read_u32(r)?)),
        5 => Value::I64(i64::from(read_i32(r)?)),
        6 => Value::F64(f64::from(read_f32(r)?)),
        7 => Value::Bool(read_u8(r)? != 0),
        8 => Value::String(read_string(r, key)?),
        9 => {
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)?;
            if len > MAX_ARRAY_LEN {
                return Err(GgufError::ImplausibleLength(len));
            }
            let mut out = Vec::with_capacity(len as usize);
            for _ in 0..len {
                out.push(read_value(r, elem_ty, key)?);
            }
            Value::Array(out)
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(read_i64(r)?),
        12 => Value::F64(read_f64(r)?),
        other => return Err(GgufError::UnknownValueType(other, key.to_string())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a GGUF header in memory. Testing the reader against a real 500 MB model in a unit
    /// test would make the suite depend on a file nobody checked in.
    fn header(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, ty, payload) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(payload);
        }
        b
    }

    fn gstr(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn reads_scalars_strings_and_arrays() {
        let mut arr = 8u32.to_le_bytes().to_vec();
        arr.extend_from_slice(&2u64.to_le_bytes());
        arr.extend(gstr("hello"));
        arr.extend(gstr("world"));

        let bytes = header(&[
            ("general.architecture", 8, gstr("qwen2")),
            ("tokenizer.ggml.eos_token_id", 4, 151_645u32.to_le_bytes().to_vec()),
            ("tokenizer.ggml.tokens", 9, arr),
        ]);
        let md = GgufMetadata::read(&bytes[..]).unwrap();
        assert_eq!(md.version, 3);
        assert_eq!(md.architecture(), Some("qwen2"));
        assert_eq!(md.get_u64("tokenizer.ggml.eos_token_id"), Some(151_645));
        assert_eq!(
            md.get("tokenizer.ggml.tokens").unwrap().as_str_array(),
            Some(vec!["hello", "world"])
        );
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        let err = GgufMetadata::read(&b"\x89PNG\r\n\x1a\n"[..]).unwrap_err();
        assert!(matches!(err, GgufError::BadMagic { .. }), "{err}");
        // The message names the cause, per docs/ux.md — not just "failed to load".
        assert!(err.to_string().contains("GGUF"));
    }

    #[test]
    fn rejects_an_implausible_length_instead_of_allocating() {
        let bytes = header(&[("k", 8, u64::MAX.to_le_bytes().to_vec())]);
        assert!(matches!(
            GgufMetadata::read(&bytes[..]).unwrap_err(),
            GgufError::ImplausibleLength(_)
        ));
    }

    #[test]
    fn rejects_an_unsupported_version() {
        let mut bytes = header(&[]);
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(GgufMetadata::read(&bytes[..]).unwrap_err(), GgufError::BadVersion(99)));
    }
}
