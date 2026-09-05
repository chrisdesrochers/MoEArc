//! The GGML quantisation type table.
//!
//! A GGUF tensor stores its element count and a numeric type id; turning that into a byte
//! count needs the type's *block geometry* — how many elements one block holds and how many
//! bytes that block occupies on disk. This module is that table and nothing else.
//!
//! 🔴 **Provenance of every constant here.** Both columns were transcribed from a checkout of
//! `ggml-org/llama.cpp`, and each was taken from two places that had to agree:
//!
//! - Type ids: `gguf-py/gguf/constants.py`, `class GGMLQuantizationType(IntEnum)`.
//! - Block geometry: `gguf-py/gguf/constants.py`, `GGML_QUANT_SIZES`, which is a
//!   `{type: (block_size, type_size)}` map written out arithmetically (e.g. Q4_K is
//!   `(256, 2 + 2 + QK_K // 2 + 12)` with `QK_K = 256`, so 144 bytes per 256 elements).
//! - Cross-checked against `ggml/src/ggml.c`, `static const struct ggml_type_traits
//!   type_traits[GGML_TYPE_COUNT]`, whose `.blck_size` / `.type_size` fields are the values the
//!   C library actually uses. That table expresses `type_size` as `sizeof(block_q4_K)` and so
//!   on, so it confirms the *shape* of the table and the ids, while the Python map supplies the
//!   arithmetic. The two agree on every type listed below.
//!
//! The gaps in the id sequence (4, 5, 31–33, 36–38) are types ggml has removed; `ggml.c` keeps
//! them as zeroed placeholder rows. They are deliberately absent here, so a file claiming one
//! gets [`None`] rather than a silently wrong size.

/// One row of the ggml type table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantType {
    /// The numeric id stored in the GGUF tensor index.
    pub id: u32,
    /// The ggml type name, lowercased as `ggml.c` spells it.
    pub name: &'static str,
    /// Elements per block. `1` for the unquantised types.
    pub block_size: u64,
    /// Bytes one block occupies.
    pub type_size: u64,
}

impl QuantType {
    /// Whether this type packs multiple elements into a shared block.
    pub fn is_quantised(self) -> bool {
        self.block_size > 1
    }
}

/// `QK_K`, the K-quant super-block size, from `gguf-py/gguf/constants.py`.
const QK_K: u64 = 256;

/// The table, ordered by id.
///
/// Kept as a flat slice rather than a match: it is data, it is short, and a lookup that walks it
/// is not on any hot path — [`inspect`](crate::inspect) consults it once per tensor at load.
static TYPES: &[QuantType] = &[
    QuantType { id: 0, name: "f32", block_size: 1, type_size: 4 },
    QuantType { id: 1, name: "f16", block_size: 1, type_size: 2 },
    QuantType { id: 2, name: "q4_0", block_size: 32, type_size: 2 + 16 },
    QuantType { id: 3, name: "q4_1", block_size: 32, type_size: 2 + 2 + 16 },
    QuantType { id: 6, name: "q5_0", block_size: 32, type_size: 2 + 4 + 16 },
    QuantType { id: 7, name: "q5_1", block_size: 32, type_size: 2 + 2 + 4 + 16 },
    QuantType { id: 8, name: "q8_0", block_size: 32, type_size: 2 + 32 },
    QuantType { id: 9, name: "q8_1", block_size: 32, type_size: 4 + 4 + 32 },
    QuantType { id: 10, name: "q2_K", block_size: QK_K, type_size: 2 + 2 + QK_K / 16 + QK_K / 4 },
    QuantType { id: 11, name: "q3_K", block_size: QK_K, type_size: 2 + QK_K / 4 + QK_K / 8 + 12 },
    QuantType { id: 12, name: "q4_K", block_size: QK_K, type_size: 2 + 2 + QK_K / 2 + 12 },
    QuantType {
        id: 13,
        name: "q5_K",
        block_size: QK_K,
        type_size: 2 + 2 + QK_K / 2 + QK_K / 8 + 12,
    },
    QuantType {
        id: 14,
        name: "q6_K",
        block_size: QK_K,
        type_size: 2 + QK_K / 2 + QK_K / 4 + QK_K / 16,
    },
    QuantType { id: 15, name: "q8_K", block_size: QK_K, type_size: 4 + QK_K + QK_K / 8 },
    QuantType { id: 16, name: "iq2_xxs", block_size: QK_K, type_size: 2 + QK_K / 4 },
    QuantType { id: 17, name: "iq2_xs", block_size: QK_K, type_size: 2 + QK_K / 4 + QK_K / 32 },
    QuantType { id: 18, name: "iq3_xxs", block_size: QK_K, type_size: 2 + QK_K / 4 + QK_K / 8 },
    QuantType { id: 19, name: "iq1_s", block_size: QK_K, type_size: 2 + QK_K / 8 + QK_K / 16 },
    QuantType { id: 20, name: "iq4_nl", block_size: 32, type_size: 2 + 16 },
    QuantType {
        id: 21,
        name: "iq3_s",
        block_size: QK_K,
        type_size: 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4,
    },
    QuantType { id: 22, name: "iq2_s", block_size: QK_K, type_size: 2 + QK_K / 4 + QK_K / 16 },
    QuantType { id: 23, name: "iq4_xs", block_size: QK_K, type_size: 2 + 2 + QK_K / 2 + QK_K / 64 },
    QuantType { id: 24, name: "i8", block_size: 1, type_size: 1 },
    QuantType { id: 25, name: "i16", block_size: 1, type_size: 2 },
    QuantType { id: 26, name: "i32", block_size: 1, type_size: 4 },
    QuantType { id: 27, name: "i64", block_size: 1, type_size: 8 },
    QuantType { id: 28, name: "f64", block_size: 1, type_size: 8 },
    QuantType {
        id: 29,
        name: "iq1_m",
        block_size: QK_K,
        type_size: QK_K / 8 + QK_K / 16 + QK_K / 32,
    },
    QuantType { id: 30, name: "bf16", block_size: 1, type_size: 2 },
    QuantType { id: 34, name: "tq1_0", block_size: QK_K, type_size: 2 + 4 * 13 },
    QuantType { id: 35, name: "tq2_0", block_size: QK_K, type_size: 2 + 64 },
    QuantType { id: 39, name: "mxfp4", block_size: 32, type_size: 1 + 16 },
    QuantType { id: 40, name: "nvfp4", block_size: 64, type_size: 4 + 32 },
    QuantType { id: 41, name: "q1_0", block_size: 128, type_size: 2 + 16 },
    QuantType { id: 42, name: "q2_0", block_size: 64, type_size: 2 + 16 },
];

/// Look up a ggml type by its GGUF type id.
///
/// Returns [`None`] for ids ggml has retired or never defined. Callers must treat that as an
/// error rather than a default — guessing a size here would silently corrupt every byte count
/// downstream, which is exactly the failure mode this crate exists to prevent.
pub fn lookup(id: u32) -> Option<QuantType> {
    TYPES.iter().copied().find(|t| t.id == id)
}

/// Every type this crate knows, for tests and diagnostics.
pub fn all() -> &'static [QuantType] {
    TYPES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_k_quants_have_the_sizes_llama_cpp_uses() {
        // Spot values, recomputed by hand from the GGML_QUANT_SIZES arithmetic:
        //   Q4_K = 2 + 2 + 128 + 12 = 144 B / 256 elements
        //   Q5_K = 2 + 2 + 128 + 32 + 12 = 176 B / 256 elements
        //   Q6_K = 2 + 128 + 64 + 16 = 210 B / 256 elements
        //   Q8_0 = 2 + 32 = 34 B / 32 elements
        assert_eq!(lookup(12).unwrap().type_size, 144);
        assert_eq!(lookup(13).unwrap().type_size, 176);
        assert_eq!(lookup(14).unwrap().type_size, 210);
        assert_eq!(lookup(8).unwrap().type_size, 34);
        assert_eq!(lookup(0).unwrap().type_size, 4);
    }

    #[test]
    fn retired_type_ids_are_not_silently_resolved() {
        // 4 and 5 were Q4_2/Q4_3; 31..=33 the Q4_0_M_N repacks; 36..=38 the IQ4_NL repacks.
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 9999] {
            assert!(lookup(id).is_none(), "type id {id} should not resolve");
        }
    }

    #[test]
    fn the_table_is_internally_consistent() {
        for t in all() {
            assert!(t.block_size > 0, "{} has a zero block size", t.name);
            assert!(t.type_size > 0, "{} has a zero type size", t.name);
        }
        // Ids are unique and sorted, so a lookup can never be ambiguous.
        for w in all().windows(2) {
            assert!(w[0].id < w[1].id, "type table is not sorted by id");
        }
    }
}
