//! Run the inference lanes at a fraction of their speed.
//!
//! At a duty of `d` percent, a lane rests after each batch for as long as the
//! batch took, times `(100 - d) / d`. Rests last a millisecond or two, far shorter
//! than the chip takes to heat or cool, so the card sees a steady lower load
//! rather than the on-off swings of pausing the whole process. Actors wait on
//! their replies meanwhile, so the CPU slows with it: on a GPU-bound generator
//! the whole process runs at about `d` percent.
//!
//! One setting per process. A generator that swaps models runs two pools for a
//! moment, and both should be paced alike.

use std::{
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};

/// The lowest duty accepted. Lower would let rests outlast the inference
/// timeout's margin on a slow batch, and is not quiet any way that matters.
pub const MINIMUM_DUTY_PERCENT: u32 = 10;

static DUTY_PERCENT: AtomicU32 = AtomicU32::new(100);

/// Set the lanes' duty, in percent of full speed.
pub fn set_duty_percent(percent: u32) -> Result<(), String> {
    if !(MINIMUM_DUTY_PERCENT..=100).contains(&percent) {
        return Err(format!(
            "duty must be from {MINIMUM_DUTY_PERCENT} to 100 percent, not {percent}"
        ));
    }
    DUTY_PERCENT.store(percent, Ordering::Relaxed);
    Ok(())
}

pub fn duty_percent() -> u32 {
    DUTY_PERCENT.load(Ordering::Relaxed)
}

/// How long a lane rests after a batch that took `busy`.
pub(crate) fn rest_after(busy: Duration) -> Duration {
    rest_at(busy, duty_percent())
}

fn rest_at(busy: Duration, duty: u32) -> Duration {
    busy * (100 - duty) / duty
}

#[cfg(test)]
mod tests {
    use super::*;

    // Setting the duty here would slow lanes in every other test of this crate,
    // so the setter and the lanes are tested in `tests/pace.rs`, a process of its own.
    #[test]
    fn rest_makes_up_the_rest_of_the_cycle() {
        let busy = Duration::from_millis(10);
        assert_eq!(rest_at(busy, 100), Duration::ZERO);
        assert_eq!(rest_at(busy, 50), busy);
        assert_eq!(rest_at(busy, 25), busy * 3);
        assert_eq!(rest_at(busy, 80), Duration::from_micros(2500));
    }
}
