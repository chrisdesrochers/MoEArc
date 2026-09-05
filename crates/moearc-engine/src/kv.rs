//! Handing out KV cache pages.
//!
//! [`super::memory`] decides *how many* pages fit on the card. This decides *who gets which*,
//! at run time, across concurrent sequences.
//!
//! Paged rather than contiguous, for the reason vLLM made standard: a contiguous per-sequence
//! allocation has to be sized for the longest output the sequence might produce, and almost
//! every sequence then wastes most of it. Pages are handed out as a sequence actually grows, so
//! the waste is bounded by one partly-filled page per sequence instead of by the context limit.
//!
//! No device memory is touched here. A page is an index; the caller maps indices to offsets in
//! whatever buffer it allocated. That keeps the allocator — where the leaks and double-frees
//! live — testable without a GPU.

use std::collections::HashMap;

/// An index into the KV page pool.
pub type PageId = u32;

/// Identifies one in-flight sequence.
pub type SeqId = u64;

/// Why a KV operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// No free pages. The caller must evict or refuse a request.
    Exhausted { requested: usize, free: usize },
    /// The sequence is not known — freed already, or never begun.
    UnknownSequence(SeqId),
    /// A sequence would exceed the configured context limit.
    ContextLimit { seq: SeqId, tokens: u32, limit: u32 },
    /// A pool with no pages, or a page holding no tokens.
    InvalidGeometry(&'static str),
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted { requested, free } => write!(
                f,
                "KV cache exhausted: {requested} page(s) needed, {free} free — reduce \
                 concurrent requests or the context length"
            ),
            Self::UnknownSequence(s) => write!(f, "sequence {s} is not active"),
            Self::ContextLimit { seq, tokens, limit } => {
                write!(f, "sequence {seq} reached {tokens} tokens, past the {limit}-token limit")
            }
            Self::InvalidGeometry(m) => write!(f, "invalid KV geometry: {m}"),
        }
    }
}

impl std::error::Error for KvError {}

/// One sequence's page table.
#[derive(Debug, Clone, Default)]
pub struct SeqPages {
    /// Pages in logical order. Token `t` lives in `pages[t / page_tokens]`.
    pub pages: Vec<PageId>,
    /// Tokens written so far.
    pub tokens: u32,
}

/// How full the pool is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvUsage {
    pub total_pages: u32,
    pub used_pages: u32,
    pub sequences: u32,
    /// Token slots allocated but not yet written — the tail of each sequence's last page.
    /// This is the waste paging is meant to bound; reporting it keeps the claim honest.
    pub slack_tokens: u32,
}

impl KvUsage {
    pub fn free_pages(&self) -> u32 {
        self.total_pages - self.used_pages
    }
    pub fn utilisation(&self) -> f64 {
        if self.total_pages == 0 { 0.0 } else { self.used_pages as f64 / self.total_pages as f64 }
    }
}

/// A pool of KV pages shared by concurrent sequences.
pub struct PagedKvCache {
    page_tokens: u32,
    context_limit: u32,
    /// Free list. Popped from the back so a just-freed page is reused first — it is the most
    /// likely to still be in cache, and reuse order is otherwise arbitrary.
    free: Vec<PageId>,
    seqs: HashMap<SeqId, SeqPages>,
    total_pages: u32,
}

impl PagedKvCache {
    /// Create a pool of `total_pages`, each holding `page_tokens` tokens.
    pub fn new(total_pages: u32, page_tokens: u32, context_limit: u32) -> Result<Self, KvError> {
        if total_pages == 0 {
            return Err(KvError::InvalidGeometry("a pool needs at least one page"));
        }
        if page_tokens == 0 {
            return Err(KvError::InvalidGeometry("a page must hold at least one token"));
        }
        Ok(Self {
            page_tokens,
            context_limit,
            // Reversed so `pop()` yields page 0 first, which makes tests readable and gives
            // deterministic behaviour on a fresh pool.
            free: (0..total_pages).rev().collect(),
            seqs: HashMap::new(),
            total_pages,
        })
    }

    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    /// Begin a sequence with room for `prompt_tokens`.
    pub fn begin(&mut self, seq: SeqId, prompt_tokens: u32) -> Result<&SeqPages, KvError> {
        if prompt_tokens > self.context_limit {
            return Err(KvError::ContextLimit {
                seq,
                tokens: prompt_tokens,
                limit: self.context_limit,
            });
        }
        let needed = prompt_tokens.div_ceil(self.page_tokens) as usize;
        if needed > self.free.len() {
            return Err(KvError::Exhausted { requested: needed, free: self.free.len() });
        }
        let mut pages = Vec::with_capacity(needed);
        for _ in 0..needed {
            pages.push(self.free.pop().expect("checked above"));
        }
        self.seqs.insert(seq, SeqPages { pages, tokens: prompt_tokens });
        Ok(self.seqs.get(&seq).expect("just inserted"))
    }

    /// Append one token, allocating a page only when the current one is full.
    ///
    /// Returns the page and the slot within it that the token occupies, which is what a kernel
    /// needs to write into.
    pub fn append(&mut self, seq: SeqId) -> Result<(PageId, u32), KvError> {
        let limit = self.context_limit;
        let page_tokens = self.page_tokens;

        let s = self.seqs.get(&seq).ok_or(KvError::UnknownSequence(seq))?;
        if s.tokens >= limit {
            return Err(KvError::ContextLimit { seq, tokens: s.tokens, limit });
        }
        let index_in_page = s.tokens % page_tokens;
        let need_new_page =
            index_in_page == 0 && s.tokens as usize >= s.pages.len() * page_tokens as usize;

        if need_new_page {
            // Take the page BEFORE mutating the sequence, so an exhausted pool leaves the
            // sequence exactly as it was rather than half-extended.
            let page = self.free.pop().ok_or(KvError::Exhausted { requested: 1, free: 0 })?;
            let s = self.seqs.get_mut(&seq).expect("checked above");
            s.pages.push(page);
        }

        let s = self.seqs.get_mut(&seq).expect("checked above");
        let page = s.pages[(s.tokens / page_tokens) as usize];
        let slot = s.tokens % page_tokens;
        s.tokens += 1;
        Ok((page, slot))
    }

    /// Release a sequence's pages.
    pub fn end(&mut self, seq: SeqId) -> Result<u32, KvError> {
        let s = self.seqs.remove(&seq).ok_or(KvError::UnknownSequence(seq))?;
        let n = s.pages.len() as u32;
        // Returned in reverse so the most recently allocated page is popped first.
        for p in s.pages.into_iter().rev() {
            self.free.push(p);
        }
        Ok(n)
    }

    /// A sequence's page table, for building the block table a kernel indexes.
    pub fn pages_of(&self, seq: SeqId) -> Result<&SeqPages, KvError> {
        self.seqs.get(&seq).ok_or(KvError::UnknownSequence(seq))
    }

    /// Whether `tokens` more tokens can be admitted right now. Used to decide whether to accept
    /// a request rather than to fail it halfway through.
    pub fn can_admit(&self, tokens: u32) -> bool {
        tokens <= self.context_limit
            && tokens.div_ceil(self.page_tokens) as usize <= self.free.len()
    }

    pub fn usage(&self) -> KvUsage {
        let used: u32 = self.seqs.values().map(|s| s.pages.len() as u32).sum();
        let slack: u32 = self
            .seqs
            .values()
            .map(|s| {
                let capacity = s.pages.len() as u32 * self.page_tokens;
                capacity.saturating_sub(s.tokens)
            })
            .sum();
        KvUsage {
            total_pages: self.total_pages,
            used_pages: used,
            sequences: self.seqs.len() as u32,
            slack_tokens: slack,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> PagedKvCache {
        PagedKvCache::new(16, 4, 1024).unwrap()
    }

    #[test]
    fn a_prompt_takes_only_the_pages_it_needs() {
        let mut kv = pool();
        // 9 tokens over 4-token pages = 3 pages, the last one a quarter used.
        let s = kv.begin(1, 9).unwrap();
        assert_eq!(s.pages.len(), 3);
        let u = kv.usage();
        assert_eq!(u.used_pages, 3);
        assert_eq!(u.slack_tokens, 3, "12 slots for 9 tokens");
    }

    #[test]
    fn appending_allocates_only_when_a_page_fills() {
        let mut kv = pool();
        kv.begin(1, 4).unwrap(); // exactly one full page
        assert_eq!(kv.usage().used_pages, 1);
        let (_, slot) = kv.append(1).unwrap(); // must open a second page
        assert_eq!(slot, 0);
        assert_eq!(kv.usage().used_pages, 2);
        for _ in 0..3 {
            kv.append(1).unwrap(); // fills page 2
        }
        assert_eq!(kv.usage().used_pages, 2, "no page opened before it was needed");
        kv.append(1).unwrap();
        assert_eq!(kv.usage().used_pages, 3);
    }

    #[test]
    fn tokens_land_where_the_page_table_says() {
        let mut kv = pool();
        kv.begin(1, 0).unwrap();
        let mut seen = Vec::new();
        for _ in 0..10 {
            seen.push(kv.append(1).unwrap());
        }
        let table = kv.pages_of(1).unwrap().pages.clone();
        for (i, (page, slot)) in seen.iter().enumerate() {
            assert_eq!(*page, table[i / 4], "token {i} is in the wrong page");
            assert_eq!(*slot, (i % 4) as u32, "token {i} is in the wrong slot");
        }
    }

    #[test]
    fn ending_a_sequence_returns_every_page() {
        let mut kv = pool();
        kv.begin(1, 12).unwrap();
        kv.begin(2, 8).unwrap();
        assert_eq!(kv.usage().used_pages, 5);
        assert_eq!(kv.end(1).unwrap(), 3);
        assert_eq!(kv.usage().used_pages, 2);
        assert_eq!(kv.end(2).unwrap(), 2);
        assert_eq!(kv.usage().used_pages, 0);
        assert_eq!(kv.usage().free_pages(), 16);
    }

    #[test]
    fn pages_are_reused_rather_than_leaked() {
        // Cycle far more sequences than there are pages. A leak shows up as exhaustion.
        let mut kv = pool();
        for i in 0..200u64 {
            kv.begin(i, 12).unwrap();
            for _ in 0..5 {
                kv.append(i).unwrap();
            }
            kv.end(i).unwrap();
        }
        assert_eq!(kv.usage().used_pages, 0);
        assert_eq!(kv.usage().free_pages(), 16);
    }

    #[test]
    fn exhaustion_is_reported_and_leaves_state_untouched() {
        let mut kv = PagedKvCache::new(2, 4, 1024).unwrap();
        kv.begin(1, 8).unwrap(); // takes both pages
        let before = kv.usage();
        let err = kv.begin(2, 4).unwrap_err();
        assert_eq!(err, KvError::Exhausted { requested: 1, free: 0 });
        assert_eq!(kv.usage(), before, "a failed begin must not consume anything");
        // And an append that cannot get a page leaves the sequence unchanged.
        let tokens_before = kv.pages_of(1).unwrap().tokens;
        assert!(matches!(kv.append(1).unwrap_err(), KvError::Exhausted { .. }));
        assert_eq!(kv.pages_of(1).unwrap().tokens, tokens_before, "sequence was half-extended");
    }

    #[test]
    fn can_admit_agrees_with_what_begin_actually_does() {
        // A scheduler decides with can_admit and then calls begin; if they disagree it will
        // fail a request it just promised to accept.
        for tokens in [0u32, 1, 4, 5, 60, 64, 65, 2000] {
            let mut kv = pool();
            let predicted = kv.can_admit(tokens);
            let actual = kv.begin(1, tokens).is_ok();
            assert_eq!(predicted, actual, "disagreement at {tokens} tokens");
        }
    }

    #[test]
    fn the_context_limit_is_enforced_at_both_entry_points() {
        let mut kv = PagedKvCache::new(64, 4, 8).unwrap();
        assert!(matches!(kv.begin(1, 9), Err(KvError::ContextLimit { .. })));
        kv.begin(2, 8).unwrap();
        assert!(matches!(kv.append(2), Err(KvError::ContextLimit { .. })));
    }

    #[test]
    fn unknown_sequences_are_refused_rather_than_panicking() {
        let mut kv = pool();
        assert_eq!(kv.append(99).unwrap_err(), KvError::UnknownSequence(99));
        assert_eq!(kv.end(99).unwrap_err(), KvError::UnknownSequence(99));
        assert!(kv.pages_of(99).is_err());
    }

    #[test]
    fn no_two_live_sequences_share_a_page() {
        // The bug that would corrupt one request's output with another's attention.
        let mut kv = pool();
        for i in 0..4u64 {
            kv.begin(i, 4).unwrap();
        }
        let mut all: Vec<PageId> =
            (0..4u64).flat_map(|i| kv.pages_of(i).unwrap().pages.clone()).collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "a page was handed to two sequences");
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(PagedKvCache::new(0, 4, 16).is_err());
        assert!(PagedKvCache::new(4, 0, 16).is_err());
    }
}
