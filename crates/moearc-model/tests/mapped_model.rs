//! Integration tests for [`MappedModel`] against a real model file.
//!
//! Skipped unless `MOEARC_TEST_GGUF` names a GGUF on disk. The companion to `real_model.rs`:
//! that one checks what the header *says*, this one checks that the bytes are where this crate
//! says they are.
//!
//! ```text
//! MOEARC_TEST_GGUF=/path/to/model.gguf cargo test -p moearc-model --test mapped_model -- --nocapture
//! ```

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use moearc_model::tensors::{ExpertBank, MappedModel};

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MOEARC_TEST_GGUF")?);
    p.is_file().then_some(p)
}

/// Read `len` bytes at `offset` with an ordinary buffered file read.
///
/// 🔴 **The point of this function is that it does not go through the mapping.** Comparing a
/// slice of the map against another slice of the same map proves nothing about whether the
/// offset is right; it would agree with itself whatever the stride arithmetic did. A separate
/// `seek` and `read` on a separate file handle is the independent second opinion.
fn read_at(path: &std::path::Path, offset: u64, len: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).unwrap();
    buf
}

#[test]
fn every_expert_slice_is_where_an_independent_reader_finds_it() {
    let Some(path) = model_path() else {
        eprintln!("skipped: set MOEARC_TEST_GGUF to a GGUF file to run this");
        return;
    };
    let m = MappedModel::open(&path).expect("open should succeed on a real model");
    let layout = m.layout();
    println!("{:?}", m);
    println!("{layout:#?}");

    let Some(n_experts) = m.expert_count() else {
        eprintln!("skipped: {} is not a mixture-of-experts model", path.display());
        return;
    };
    assert!(layout.expert_blocks > 0);

    // First, last and a middle block: enough to catch an offset that is right at the start of the
    // file and drifts, without reading every expert of every block on a 20 GiB model.
    let blocks: Vec<u32> = {
        let mut b: Vec<u32> = (0..layout.block_count)
            .filter(|&i| m.has_tensor(&format!("blk.{i}.{}", ExpertBank::Gate.suffix())))
            .collect();
        b.dedup();
        match b.len() {
            0 => vec![],
            n => vec![b[0], b[n / 2], b[n - 1]],
        }
    };

    let mut checked = 0u64;
    for &block in &blocks {
        for bank in ExpertBank::ALL {
            let Ok(parent) = m.block_tensor(block, bank.suffix()) else { continue };
            assert_eq!(
                parent.dims.last().copied(),
                Some(u64::from(n_experts)),
                "{} does not stack {n_experts} experts",
                parent.name
            );

            let mut seen = std::collections::HashSet::new();
            let mut total = 0u64;
            for k in 0..n_experts {
                let e = m.expert(block, bank, k).unwrap();

                // 1. The slice sits inside its parent, at the stride the parent implies.
                assert_eq!(e.len() as u64 * u64::from(n_experts), parent.len() as u64);
                assert_eq!(
                    e.file_offset,
                    parent.file_offset + u64::from(k) * e.len() as u64,
                    "{} expert {k} is not at its own stride",
                    parent.name
                );

                // 2. The bytes match what a plain seek-and-read finds at that offset.
                assert_eq!(
                    e.data,
                    read_at(&path, e.file_offset, e.len()).as_slice(),
                    "{} expert {k} does not match an independent read at offset {}",
                    parent.name,
                    e.file_offset
                );

                // 3. The experts differ from each other, so (2) cannot be passing vacuously on a
                //    run of identical bytes. Real trained weights never repeat; a fill of zeros
                //    would make every offset look correct.
                assert!(
                    seen.insert(e.data.to_vec()),
                    "{} expert {k} is byte-identical to an earlier expert",
                    parent.name
                );

                total += e.len() as u64;
                checked += 1;
            }
            // 4. The experts tile the bank exactly: no gap, no overlap, nothing left over.
            assert_eq!(total, parent.len() as u64, "{} experts do not tile the bank", parent.name);
        }
    }
    println!("verified {checked} expert slices across blocks {blocks:?}");
    assert!(checked > 0, "no expert slices were checked");
}

#[test]
fn an_out_of_range_expert_is_refused_on_a_real_bank() {
    let Some(path) = model_path() else { return };
    let m = MappedModel::open(&path).unwrap();
    let Some(n) = m.expert_count() else { return };
    let block = (0..m.layout().block_count)
        .find(|&i| m.has_tensor(&format!("blk.{i}.{}", ExpertBank::Gate.suffix())));
    let Some(block) = block else { return };
    assert!(m.expert(block, ExpertBank::Gate, n).is_err(), "expert {n} of {n} must not resolve");
    assert!(m.expert(block, ExpertBank::Gate, n - 1).is_ok());
}

#[test]
fn mapping_a_model_does_not_read_it_into_memory() {
    let Some(path) = model_path() else { return };
    let Some(before) = vm_rss() else {
        eprintln!("skipped: no /proc/self/status");
        return;
    };
    let m = MappedModel::open(&path).unwrap();
    let after = vm_rss().unwrap();
    let file_size = m.mapped_bytes() as u64;

    // The bound is generous on purpose — it has to hold for a 4 GiB model and a 20 GiB one, and
    // the header parse legitimately allocates a few MB of metadata. What it rules out is the
    // thing that would actually be wrong: resident memory growing with the size of the file.
    let growth = after.saturating_sub(before);
    println!("VmRSS {before} -> {after} B (+{growth}) for a {file_size} B file");
    assert!(
        growth < 64 * 1024 * 1024,
        "mapping a {file_size} B file added {growth} B of RSS; it should add almost none"
    );
    assert!(growth * 4 < file_size, "RSS growth is proportional to file size — this is not a map");
}

fn vm_rss() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    Some(line.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
}
