//! GPU device discovery.
//!
//! Answers the first question the tool asks on behalf of the user: what card is in this
//! machine, how much of it can we use, and if the answer is "none", exactly why.
//!
//! See `docs/ux.md` — failure here must be legible. A missing kernel driver is the one
//! dependency we are allowed to ask the user for, and it must be named precisely.
