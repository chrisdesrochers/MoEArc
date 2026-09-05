//! A minimal GGUF header reader.
//!
//! GGUF is a flat, forward-only container: a magic, a version, two counts, a typed key/value
//! block, then a tensor index, then the tensor data blob. Everything this crate needs lives in
//! the first two sections, so this reader never touches the blob — on the 20.6 GiB model it
//! reads about 11 MB and stops.
//!
//! Structure and field order follow the format comment at the top of `ggml/include/gguf.h`
//! in `ggml-org/llama.cpp`; the value-type ids are that header's `enum gguf_type`.
//!
//! Hand-rolled rather than pulled from a crate. The parse is ~200 lines against a stable
//! on-disk layout, and the alternative is a dependency that drags in a tensor library we are
//! in the business of replacing. It is also deliberately *not* mmap-based: a buffered
//! sequential read with relative seeks over the header is both simpler and strictly less
//! dangerous than mapping a multi-gigabyte file we have no intention of reading.
//!
//! **Every length in this file is attacker-controlled.** A GGUF header is just numbers on
//! disk, and a corrupt or hostile one can claim a 2^63-element array. Each count is therefore
//! checked against the real file length before it is used to allocate or to loop.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use crate::ModelError;

/// GGUF's default tensor-data alignment, from `GGUF_DEFAULT_ALIGNMENT` in `gguf.h`.
const DEFAULT_ALIGNMENT: u64 = 32;

/// ggml's tensor rank limit, from `GGML_MAX_DIMS` in `ggml.h`. A header claiming more is
/// corrupt, and bounding it here keeps the dimension loop finite.
const MAX_DIMS: u32 = 4;

/// Smallest number of bytes a single key/value pair can occupy: an 8-byte key length, an empty
/// key, a 4-byte value type and a 1-byte value. Used only to reject an impossible KV count
/// before it is used as a loop bound.
const MIN_KV_BYTES: u64 = 8 + 4 + 1;

/// Smallest number of bytes one tensor-index entry can occupy: an 8-byte name length, an empty
/// name, a 4-byte rank, a 4-byte type and an 8-byte offset.
const MIN_TENSOR_BYTES: u64 = 8 + 4 + 4 + 8;

/// Longest string this reader will materialise. Chat templates run to tens of kilobytes, so
/// the limit has to be generous; it exists to stop a corrupt length from becoming a 16 EiB
/// allocation, not to enforce a format rule.
const MAX_STRING_BYTES: u64 = 1 << 26;

/// Longest array this reader keeps element-by-element. Above it the elements are skipped and
/// only the type and count are retained.
///
/// This is what keeps the reader's memory flat on real files: `tokenizer.ggml.tokens` and
/// `tokenizer.ggml.merges` hold hundreds of thousands of strings apiece and account for nearly
/// all of a GGUF header's bulk, and nothing in this crate reads them. Every array that *is*
/// consulted — RoPE sections, per-layer head counts — is a handful of elements.
const MAX_KEPT_ARRAY: u64 = 1024;

/// A GGUF metadata value.
///
/// The integer widths are kept distinct rather than widened on read, because a key's declared
/// type is itself evidence: `general.alignment` is specified as `UINT32` and a file that writes
/// it as something else is telling us something. Callers that only want the number use
/// [`Value::as_u64`], which accepts any integer variant — writers do vary on width for the
/// same key, and rejecting a `UINT64` block count would be pedantry, not safety.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// A short array, retained in full.
    Array(Vec<Value>),
    /// A long array whose elements were skipped. See [`MAX_KEPT_ARRAY`].
    SkippedArray {
        elem_type: u32,
        len: u64,
    },
}

impl Value {
    /// The value as an unsigned integer, if it is one and is non-negative.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            Self::Bool(v) => Some(u64::from(v)),
            _ => None,
        }
    }

    /// The value as a string slice, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The elements of a retained array, if this is one.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// One entry of the tensor index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions, fastest-varying first, exactly as stored.
    pub dims: Vec<u64>,
    /// The ggml type id; resolve it with [`crate::quant::lookup`].
    pub type_id: u32,
    /// Byte offset within the tensor-data blob, i.e. relative to [`GgufHeader::data_offset`].
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count — the product of the dimensions.
    /// Saturating, because the dimensions are untrusted: a wrapped product would look small and
    /// plausible, where `u64::MAX` fails every downstream check loudly.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().fold(1u64, |acc, &d| acc.saturating_mul(d))
    }

    /// Bytes this tensor occupies on disk, unpadded.
    ///
    /// This is ggml's `ggml_nbytes` for the contiguous case: `nelements / blck_size *
    /// type_size`. The element count must be a whole number of blocks, which ggml guarantees by
    /// requiring the first dimension to be a multiple of the block size; a file that violates it
    /// is rejected rather than rounded, since rounding would put every later offset out by a
    /// partial block.
    pub fn nbytes(&self) -> Result<u64, ModelError> {
        let t = crate::quant::lookup(self.type_id).ok_or_else(|| {
            ModelError::UnknownTensorType { tensor: self.name.clone(), type_id: self.type_id }
        })?;
        let elements = self.n_elements();
        if elements % t.block_size != 0 {
            return Err(ModelError::ElementsNotBlockAligned {
                tensor: self.name.clone(),
                elements,
                block_size: t.block_size,
            });
        }
        Ok((elements / t.block_size).saturating_mul(t.type_size))
    }
}

/// Everything in a GGUF file before the tensor data.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// 2 or 3; see [`read`].
    pub version: u32,
    /// Tensor-data alignment, from `general.alignment` or [`DEFAULT_ALIGNMENT`].
    pub alignment: u64,
    /// Absolute file offset at which the tensor-data blob starts.
    pub data_offset: u64,
    /// Size of the file on disk.
    pub file_size: u64,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufHeader {
    /// Look up a metadata key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Look up a key that must be present and must be an unsigned integer.
    pub fn u64_key(&self, key: &str) -> Result<u64, ModelError> {
        let v = self.get(key).ok_or_else(|| ModelError::MissingKey(key.to_string()))?;
        v.as_u64().ok_or_else(|| ModelError::WrongKeyType {
            key: key.to_string(),
            want: "an unsigned integer",
        })
    }

    /// Look up a key that must be present, must be an unsigned integer, and must fit in a `u32`.
    pub fn u32_key(&self, key: &str) -> Result<u32, ModelError> {
        let v = self.u64_key(key)?;
        u32::try_from(v).map_err(|_| ModelError::WrongKeyType {
            key: key.to_string(),
            want: "a value that fits in u32",
        })
    }

    /// Look up a key that must be present and must be a string.
    pub fn str_key(&self, key: &str) -> Result<&str, ModelError> {
        let v = self.get(key).ok_or_else(|| ModelError::MissingKey(key.to_string()))?;
        v.as_str()
            .ok_or_else(|| ModelError::WrongKeyType { key: key.to_string(), want: "a string" })
    }
}

/// Read the header of a GGUF file.
///
/// Accepts versions 2 and 3. Version 1 used 32-bit lengths and counts throughout and is a
/// genuinely different layout — llama.cpp itself refuses it — so it is rejected rather than
/// half-supported. Versions above 3 are rejected because a bump means a layout change we have
/// not read; guessing would produce plausible numbers from a misparse, which is worse than an
/// error.
pub fn read(path: &Path) -> Result<GgufHeader, ModelError> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut r = Reader::new(file, file_size);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(ModelError::BadMagic { found: magic });
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(ModelError::UnsupportedVersion(version));
    }

    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    // Reject impossible counts *before* looping on them. Each entry has a floor on its encoded
    // size, so a count above file_size/floor cannot describe this file at any content.
    if kv_count.saturating_mul(MIN_KV_BYTES) > file_size {
        return Err(ModelError::ImplausibleCount {
            what: "metadata keys",
            count: kv_count,
            file_size,
        });
    }
    if tensor_count.saturating_mul(MIN_TENSOR_BYTES) > file_size {
        return Err(ModelError::ImplausibleCount {
            what: "tensors",
            count: tensor_count,
            file_size,
        });
    }

    let mut metadata = HashMap::with_capacity(kv_count.min(4096) as usize);
    for _ in 0..kv_count {
        let key = r.string()?;
        let type_id = r.u32()?;
        let value = r.value(type_id, &key)?;
        // Later wins, matching llama.cpp, which warns on a duplicate but keeps parsing.
        metadata.insert(key, value);
    }

    // `general.alignment` is specified as UINT32 and must be a power of two — `gguf.cpp`
    // enforces both. It is load-bearing: it sets where the data blob starts, so a wrong value
    // shifts every tensor.
    let alignment = match metadata.get("general.alignment") {
        Some(v) => v.as_u64().ok_or_else(|| ModelError::WrongKeyType {
            key: "general.alignment".into(),
            want: "an unsigned integer",
        })?,
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ModelError::BadAlignment { alignment });
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > MAX_DIMS {
            return Err(ModelError::TooManyDims { tensor: name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let type_id = r.u32()?;
        let offset = r.u64()?;
        tensors.push(TensorInfo { name, dims, type_id, offset });
    }

    let data_offset = pad_to(r.pos, alignment);

    // Prove the file actually holds what the index claims. This is the check that turns a
    // truncated download into a typed error instead of a confident, wrong byte total: the
    // metadata alone parses fine on a half-downloaded file, because it all lives up front.
    for t in &tensors {
        let padded = pad_to(t.nbytes()?, alignment);
        // Saturating rather than checked: an overflow here can only come from a corrupt shape,
        // and u64::MAX fails the very next comparison, which reports the tensor by name.
        let end = data_offset.saturating_add(t.offset).saturating_add(padded);
        if end > file_size {
            return Err(ModelError::TensorDataOverrunsFile {
                tensor: t.name.clone(),
                end,
                file_size,
            });
        }
    }

    Ok(GgufHeader { version, alignment, data_offset, file_size, metadata, tensors })
}

/// Round `n` up to a multiple of `align`, which must be a power of two. ggml's `GGML_PAD`.
///
/// Saturating: `n` can come from a corrupt tensor shape, and a wrapped result would understate
/// the span and let the file-length check pass on a file that cannot hold it.
fn pad_to(n: u64, align: u64) -> u64 {
    n.div_ceil(align).saturating_mul(align)
}

/// A position-tracking, length-checked reader over the header region.
struct Reader<R> {
    inner: BufReader<R>,
    pos: u64,
    file_size: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn new(inner: R, file_size: u64) -> Self {
        // 1 MiB: the header is read strictly sequentially in small pieces, and this size keeps
        // the ~11 MB header of a large model to a handful of syscalls.
        Self { inner: BufReader::with_capacity(1 << 20, inner), pos: 0, file_size }
    }

    /// Fail before reading if `n` bytes cannot exist. Without this a truncated file surfaces as
    /// a bare `UnexpectedEof` with no offset, which is a far worse diagnostic.
    fn need(&self, n: u64) -> Result<(), ModelError> {
        if self.pos.saturating_add(n) > self.file_size {
            return Err(ModelError::Truncated {
                offset: self.pos,
                needed: n,
                file_size: self.file_size,
            });
        }
        Ok(())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ModelError> {
        self.need(buf.len() as u64)?;
        self.inner.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    /// Advance without reading.
    ///
    /// `seek_relative` rather than `Seek::seek`: the latter drops the whole buffer on every
    /// call, and the hot use of this is skipping a few hundred thousand short tokenizer strings
    /// one at a time. `seek_relative` stays inside the buffer when it can, which turns that from
    /// hundreds of thousands of syscalls into a handful.
    fn skip(&mut self, n: u64) -> Result<(), ModelError> {
        self.need(n)?;
        let delta = i64::try_from(n).map_err(|_| ModelError::Truncated {
            offset: self.pos,
            needed: n,
            file_size: self.file_size,
        })?;
        self.inner.seek_relative(delta)?;
        self.pos += n;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// A GGUF string: a `u64` byte length then the bytes, with no terminator.
    fn string(&mut self) -> Result<String, ModelError> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(ModelError::Truncated {
                offset: self.pos,
                needed: len,
                file_size: self.file_size,
            });
        }
        self.need(len)?;
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        // Lossy on purpose. A mangled byte in a tokenizer merge must not stop us reporting an
        // expert count; nothing this crate decides on is affected by a replacement character.
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// One value of the given GGUF type. `key` is carried only for error messages.
    fn value(&mut self, type_id: u32, key: &str) -> Result<Value, ModelError> {
        Ok(match type_id {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            // "All bool values are stored as int8_t" — the format comment in `gguf.h`.
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => self.array(key)?,
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(ModelError::BadValueType { key: key.to_string(), type_id: other }),
        })
    }

    fn array(&mut self, key: &str) -> Result<Value, ModelError> {
        let elem_type = self.u32()?;
        let len = self.u64()?;
        if elem_type == 9 {
            // ggml has no nested arrays; `gguf.cpp` rejects them outright.
            return Err(ModelError::NestedArray { key: key.to_string() });
        }
        // Fixed-width types have a known stride, so a bogus length is caught immediately rather
        // than after millions of reads. Strings have no stride and must be walked.
        if let Some(stride) = fixed_width(elem_type) {
            self.need(len.saturating_mul(stride))?;
        } else if elem_type != 8 {
            return Err(ModelError::BadValueType { key: key.to_string(), type_id: elem_type });
        }

        if len > MAX_KEPT_ARRAY {
            match fixed_width(elem_type) {
                Some(stride) => self.skip(len * stride)?,
                None => {
                    for _ in 0..len {
                        let n = self.u64()?;
                        self.skip(n)?;
                    }
                }
            }
            return Ok(Value::SkippedArray { elem_type, len });
        }

        let mut out = Vec::with_capacity(len as usize);
        for _ in 0..len {
            out.push(self.value(elem_type, key)?);
        }
        Ok(Value::Array(out))
    }
}

/// Bytes one element of a fixed-width GGUF value type occupies, or [`None`] for strings and
/// arrays, which are self-describing.
fn fixed_width(type_id: u32) -> Option<u64> {
    Some(match type_id {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}
