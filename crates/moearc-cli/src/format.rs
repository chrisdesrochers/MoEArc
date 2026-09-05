//! Number formatting, in one place because the TUI and the plain-text renderer must agree.
//!
//! Two units disagreeing across two code paths is how a user ends up believing a download
//! shrank when they piped it to a file.

/// Binary byte sizes: `11.4 GiB`. GiB not GB, because every other number a GPU reports is
/// binary and mixing the two makes free VRAM look 7% larger than it is.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{n} B") } else { format!("{value:.1} {}", UNITS[unit]) }
}

/// Transfer rate, same units as [`bytes`].
pub fn rate(bytes_per_sec: u64) -> String {
    format!("{}/s", bytes(bytes_per_sec))
}

/// Thousands-separated counts: `83,632`. Token counts are the numbers users compare against
/// each other most often, and unseparated six-digit numbers are hard to compare at a glance.
pub fn count(n: i64) -> String {
    let negative = n < 0;
    let digits = n.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    if negative {
        out.push('-');
    }
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Elapsed or remaining time, coarse on purpose: a download ETA accurate to the second is a
/// lie told precisely.
pub fn duration(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {:02}s", secs / 60, secs % 60),
        _ => format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60),
    }
}

/// A ratio as a whole percent.
pub fn percent(numerator: i64, denominator: i64) -> String {
    if denominator == 0 {
        return "—".to_string();
    }
    format!("{}%", (numerator as f64 / denominator as f64 * 100.0).round() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_stays_in_binary_units() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1024), "1.0 KiB");
        assert_eq!(bytes(12_241_698_816), "11.4 GiB");
    }

    #[test]
    fn counts_are_grouped() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1000), "1,000");
        assert_eq!(count(83_632), "83,632");
        assert_eq!(count(-1_234_567), "-1,234,567");
    }

    #[test]
    fn durations_round_up_the_scale() {
        assert_eq!(duration(9), "9s");
        assert_eq!(duration(134), "2m 14s");
        assert_eq!(duration(7_260), "2h 01m");
    }

    #[test]
    fn percent_of_nothing_is_not_a_division_by_zero() {
        assert_eq!(percent(1, 0), "—");
        assert_eq!(percent(62, 128), "48%");
    }
}
