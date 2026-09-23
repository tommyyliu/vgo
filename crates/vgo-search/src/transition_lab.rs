//! Scoped, bounded, thread-local support caches. No certificates live in nodes.
//! Only Action transitions on the calling thread participate; core::place and
//! evaluators running on other threads remain unchanged.
use std::{cell::RefCell, collections::VecDeque};
use vgo_core::iteration_lab::{SupportPolicy, SupportPosition};
use vgo_core::{MoveError, MoveResult, Position, Ruleset};

#[derive(Clone, Copy, Debug)]
pub struct SupportConfig {
    /// Retained entry payload budget, not peak allocation or RSS. LRU metadata
    /// has a separate hard cap of 64 entries. Zero bypasses preparation.
    pub byte_budget: usize,
    /// Compare every accelerated transition's full result to the baseline.
    /// Correctness mode only: do not enable for throughput measurements.
    pub verify: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SupportMetrics {
    pub calls: u64,
    pub hits: u64,
    pub builds: u64,
    pub evictions: u64,
    pub oversized: u64,
    pub peak_retained_bytes: usize,
}

struct Entry {
    hash: u64,
    bytes: usize,
    cache: SupportPosition<'static>,
}
struct Backend {
    config: SupportConfig,
    entries: VecDeque<Entry>,
    retained: usize,
    metrics: SupportMetrics,
}

thread_local! { static BACKEND: RefCell<Option<Backend>> = const { RefCell::new(None) }; }

/// Enable the experiment for one synchronous operation. Scopes nest and restore
/// the prior backend even on panic; all entries are dropped at scope exit.
pub fn with_support_backend<T>(
    config: SupportConfig,
    operation: impl FnOnce() -> T,
) -> (T, SupportMetrics) {
    struct Restore(Option<Backend>);
    impl Drop for Restore {
        fn drop(&mut self) {
            BACKEND.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let previous = BACKEND.with(|slot| {
        slot.replace(Some(Backend {
            config,
            entries: VecDeque::new(),
            retained: 0,
            metrics: SupportMetrics::default(),
        }))
    });
    let restore = Restore(previous);
    let result = operation();
    let metrics = BACKEND.with(|slot| slot.borrow().as_ref().unwrap().metrics);
    drop(restore);
    (result, metrics)
}

fn fingerprint(position: &Position) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut add = |bits| {
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    add(position.radius().to_bits());
    for stone in position.stones() {
        add(stone.x.to_bits());
        add(stone.y.to_bits());
        add(stone.color as u64);
    }
    hash
}

pub(crate) fn place(position: &Position, x: f64, y: f64) -> Option<Result<MoveResult, MoveError>> {
    BACKEND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let backend = slot.as_mut()?;
        if backend.config.byte_budget == 0 || position.ruleset() != Ruleset::Vgo {
            return None;
        }
        backend.metrics.calls += 1;
        let hash = fingerprint(position);
        let hit = backend
            .entries
            .iter()
            .position(|entry| entry.hash == hash && entry.cache.matches(position));
        let entry = if let Some(index) = hit {
            backend.metrics.hits += 1;
            backend.entries.remove(index).unwrap()
        } else {
            backend.metrics.builds += 1;
            let cache = SupportPosition::new(position, SupportPolicy::All).into_owned();
            let bytes = cache.retained_bytes();
            if bytes > backend.config.byte_budget {
                backend.metrics.oversized += 1;
                // Decline retention and acceleration when this parent exceeds
                // the budget. Preparation scratch is not a peak-memory bound.
                return None;
            }
            while backend.retained + bytes > backend.config.byte_budget
                || backend.entries.len() >= 64
            {
                backend.retained -= backend.entries.pop_back().unwrap().bytes;
                backend.metrics.evictions += 1;
            }
            backend.retained += bytes;
            backend.metrics.peak_retained_bytes =
                backend.metrics.peak_retained_bytes.max(backend.retained);
            Entry { hash, bytes, cache }
        };
        backend.entries.push_front(entry);
        // Keep bookkeeping consistent even if a caller catches a transition
        // panic inside this scope rather than unwinding the whole scope.
        let result = backend.entries.front().unwrap().cache.place(x, y);
        if backend.config.verify {
            vgo_core::iteration_lab::assert_same(&vgo_core::place(position, x, y), &result);
        }
        Some(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgo_core::{Color, Stone};

    #[test]
    fn full_naive_search_matches_with_both_rules_and_leaf_batching() {
        for radius in [0.1, 1.0 / 18.0, 1.0 / 38.0] {
            for rules in [Ruleset::Vgo, Ruleset::Official] {
                let parent = Position::new(
                    radius,
                    vec![
                        Stone::new(radius, radius, Color::Black),
                        Stone::new(1.0 - radius, radius, Color::White),
                    ],
                    Color::Black,
                )
                .with_ruleset(rules)
                .with_komi(0.104);
                for batch in [1, 4] {
                    let mut config = crate::SearchConfig::canary(16);
                    config.leaf_batch = batch;
                    let expected = crate::search(&parent, config, 73);
                    let (actual, metrics) = with_support_backend(
                        SupportConfig {
                            byte_budget: 1024 * 1024,
                            verify: true,
                        },
                        || crate::search(&parent, config, 73),
                    );
                    assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
                    let run_stepped = || {
                        let mut search = crate::SteppedSearch::new(parent.clone(), config, 73, 0);
                        crate::drive_stepped(&mut search, &crate::NaiveEvaluator).unwrap();
                        search.finish().unwrap()
                    };
                    // Compare each driver with itself: baseline sequential and
                    // stepped drivers can count generated candidates differently.
                    let expected_stepped = run_stepped();
                    let (stepped, _) = with_support_backend(
                        SupportConfig {
                            byte_budget: 1024 * 1024,
                            verify: true,
                        },
                        run_stepped,
                    );
                    assert_eq!(format!("{expected_stepped:?}"), format!("{stepped:?}"));
                    assert!(metrics.peak_retained_bytes <= 1024 * 1024);
                    if rules == Ruleset::Vgo {
                        assert!(metrics.hits > 0);
                        assert!(metrics.builds > 0);
                    } else {
                        assert_eq!(metrics.calls, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn payload_budget_eviction_and_metadata_identity() {
        let parent = Position::new(0.05, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
        let other = parent.clone().with_komi(0.104);
        let budget = SupportPosition::new(&parent, SupportPolicy::All)
            .into_owned()
            .retained_bytes();
        let (_, metrics) = with_support_backend(
            SupportConfig {
                byte_budget: budget,
                verify: true,
            },
            || {
                for p in [&parent, &parent, &other, &parent] {
                    place(p, 0.9, 0.9).unwrap().unwrap();
                }
            },
        );
        assert_eq!(metrics.builds, 3);
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.evictions, 2);
        assert_eq!(metrics.peak_retained_bytes, budget);
        let (_, metrics) = with_support_backend(
            SupportConfig {
                byte_budget: 1,
                verify: false,
            },
            || assert!(place(&parent, 0.9, 0.9).is_none()),
        );
        assert_eq!(metrics.oversized, 1);
        assert_eq!(metrics.peak_retained_bytes, 0);
        let (_, metrics) = with_support_backend(
            SupportConfig {
                byte_budget: 0,
                verify: false,
            },
            || assert!(place(&parent, 0.9, 0.9).is_none()),
        );
        assert_eq!(metrics.builds, 0);
    }

    #[test]
    fn scopes_restore_on_panic_and_do_not_cross_threads() {
        let parent = Position::new(0.05, vec![], Color::Black);
        assert!(place(&parent, 0.5, 0.5).is_none());
        let (_, metrics) = with_support_backend(
            SupportConfig {
                byte_budget: 4096,
                verify: true,
            },
            || {
                place(&parent, 0.5, 0.5).unwrap().unwrap();
                assert!(
                    std::panic::catch_unwind(|| with_support_backend(
                        SupportConfig {
                            byte_budget: 0,
                            verify: false
                        },
                        || panic!("test unwind")
                    ))
                    .is_err()
                );
                std::thread::spawn(|| {
                    assert!(place(&Position::new(0.05, vec![], Color::Black), 0.5, 0.5).is_none())
                })
                .join()
                .unwrap();
                place(&parent, 0.5, 0.5).unwrap().unwrap();
            },
        );
        assert_eq!(metrics.hits, 1);
        assert!(place(&parent, 0.5, 0.5).is_none());
    }
}
