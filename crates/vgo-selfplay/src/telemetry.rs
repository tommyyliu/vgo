//! Low-frequency generator accounting. Search timings include inference waits;
//! broker queue timings are position-weighted. Neither is a CPU-time estimate.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{Value, json};
use vgo_inference::BrokerMetrics;

use crate::generation::GenerationProgress;

#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub searched_plies: u64,
    pub policy_eligible_plies: u64,
    pub simulations: u64,
    pub evaluations: u64,
    pub search_nanoseconds: u64,
    pub published_games: u64,
    pub published_samples: u64,
    pub capped_games: u64,
    pub soft_resigned_games: u64,
    pub discarded_games: u64,
    pub write_nanoseconds: u64,
}

impl Counters {
    fn add(&mut self, other: Self) {
        self.searched_plies += other.searched_plies;
        self.policy_eligible_plies += other.policy_eligible_plies;
        self.simulations += other.simulations;
        self.evaluations += other.evaluations;
        self.search_nanoseconds += other.search_nanoseconds;
        self.published_games += other.published_games;
        self.published_samples += other.published_samples;
        self.capped_games += other.capped_games;
        self.soft_resigned_games += other.soft_resigned_games;
        self.discarded_games += other.discarded_games;
        self.write_nanoseconds += other.write_nanoseconds;
    }

    pub fn delta_since(self, previous: Self) -> Self {
        Self {
            searched_plies: self.searched_plies.saturating_sub(previous.searched_plies),
            policy_eligible_plies: self
                .policy_eligible_plies
                .saturating_sub(previous.policy_eligible_plies),
            simulations: self.simulations.saturating_sub(previous.simulations),
            evaluations: self.evaluations.saturating_sub(previous.evaluations),
            search_nanoseconds: self
                .search_nanoseconds
                .saturating_sub(previous.search_nanoseconds),
            published_games: self
                .published_games
                .saturating_sub(previous.published_games),
            published_samples: self
                .published_samples
                .saturating_sub(previous.published_samples),
            capped_games: self.capped_games.saturating_sub(previous.capped_games),
            soft_resigned_games: self
                .soft_resigned_games
                .saturating_sub(previous.soft_resigned_games),
            discarded_games: self
                .discarded_games
                .saturating_sub(previous.discarded_games),
            write_nanoseconds: self
                .write_nanoseconds
                .saturating_sub(previous.write_nanoseconds),
        }
    }

    pub fn json(self, seconds: f64) -> Value {
        let rate = |count: u64| {
            if seconds > 0.0 {
                count as f64 / seconds
            } else {
                0.0
            }
        };
        json!({
            "searched_plies": self.searched_plies,
            "policy_eligible_plies": self.policy_eligible_plies,
            "simulations": self.simulations,
            "evaluations": self.evaluations,
            "search_wall_seconds_summed": self.search_nanoseconds as f64 / 1e9,
            "published_games": self.published_games,
            "published_samples": self.published_samples,
            "capped_games": self.capped_games,
            "soft_resigned_games": self.soft_resigned_games,
            "discarded_games": self.discarded_games,
            "write_wall_seconds_summed": self.write_nanoseconds as f64 / 1e9,
            "searched_plies_per_second": rate(self.searched_plies),
            "published_games_per_hour": rate(self.published_games) * 3600.0,
            "published_samples_per_hour": rate(self.published_samples) * 3600.0,
        })
    }
}

/// Continuous board bands are grouped by rounded reciprocal radius. Game
/// stages are thirds of that game's ply cap, not a claim of geometric settlement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Bucket {
    pub board_units_rounded: u32,
    pub stage: u32,
    pub soft_resigned: bool,
}

impl Bucket {
    fn from_progress(progress: GenerationProgress) -> Self {
        Self {
            board_units_rounded: (1.0 / progress.radius).round() as u32,
            stage: ((u64::from(progress.ply) * 3) / u64::from(progress.maximum_plies.max(1))).min(2)
                as u32,
            soft_resigned: progress.soft_resigned,
        }
    }
}

#[derive(Default)]
struct ActorState {
    totals: Counters,
    boards: BTreeMap<u32, Counters>,
    buckets: BTreeMap<Bucket, Counters>,
    phase: &'static str,
}

pub struct GenerationTelemetry {
    actors: Vec<Mutex<ActorState>>,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub totals: Counters,
    pub boards: BTreeMap<u32, Counters>,
    pub buckets: BTreeMap<Bucket, Counters>,
    pub actors: BTreeMap<&'static str, usize>,
}

impl GenerationTelemetry {
    pub fn new(actors: usize) -> Self {
        Self {
            actors: (0..actors)
                .map(|_| {
                    Mutex::new(ActorState {
                        phase: "idle",
                        ..ActorState::default()
                    })
                })
                .collect(),
        }
    }

    pub fn phase(&self, actor: usize, phase: &'static str) {
        self.actors[actor]
            .lock()
            .expect("telemetry actor lock")
            .phase = phase;
    }

    pub fn searched(&self, actor: usize, progress: GenerationProgress) {
        let counters = Counters {
            searched_plies: 1,
            policy_eligible_plies: u64::from(progress.policy_eligible),
            simulations: u64::from(progress.simulations),
            evaluations: progress.evaluations,
            search_nanoseconds: progress
                .search_duration
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
            ..Counters::default()
        };
        let mut state = self.actors[actor].lock().expect("telemetry actor lock");
        state.totals.add(counters);
        state
            .boards
            .entry(Bucket::from_progress(progress).board_units_rounded)
            .or_default()
            .add(counters);
        state
            .buckets
            .entry(Bucket::from_progress(progress))
            .or_default()
            .add(counters);
    }

    pub fn published(
        &self,
        actor: usize,
        radius: f64,
        samples: usize,
        capped: bool,
        soft_resigned: bool,
        duration: Duration,
    ) {
        let mut state = self.actors[actor].lock().expect("telemetry actor lock");
        let counters = Counters {
            published_games: u64::from(samples > 0),
            published_samples: samples as u64,
            capped_games: u64::from(samples > 0 && capped),
            soft_resigned_games: u64::from(samples > 0 && soft_resigned),
            discarded_games: u64::from(samples == 0),
            write_nanoseconds: duration.as_nanos().min(u128::from(u64::MAX)) as u64,
            ..Counters::default()
        };
        state.totals.add(counters);
        state
            .boards
            .entry((1.0 / radius).round() as u32)
            .or_default()
            .add(counters);
    }

    pub fn discarded(&self, actor: usize) {
        self.actors[actor]
            .lock()
            .expect("telemetry actor lock")
            .totals
            .discarded_games += 1;
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut snapshot = Snapshot::default();
        for actor in &self.actors {
            // Only one actor is locked at a time; search is never held up by
            // serialization, disk access, or a broker metrics read.
            let state = actor.lock().expect("telemetry actor lock");
            snapshot.totals.add(state.totals);
            for (&board, &counters) in &state.boards {
                snapshot.boards.entry(board).or_default().add(counters);
            }
            for (&bucket, &counters) in &state.buckets {
                snapshot.buckets.entry(bucket).or_default().add(counters);
            }
            *snapshot.actors.entry(state.phase).or_default() += 1;
        }
        snapshot
    }
}

impl Snapshot {
    pub fn interval_json(&self, previous: &Self, seconds: f64) -> Value {
        let boards: Vec<Value> = self.boards.iter().map(|(board, counters)| json!({
            "board_units_rounded": board,
            "counters": counters.delta_since(previous.boards.get(board).copied().unwrap_or_default()).json(seconds),
        })).collect();
        let buckets: Vec<Value> = self.buckets.iter().map(|(bucket, counters)| {
            json!({
                "board_units_rounded": bucket.board_units_rounded,
                "stage": (["opening", "middle", "late"][bucket.stage as usize]),
                "soft_resigned": bucket.soft_resigned,
                "counters": counters.delta_since(previous.buckets.get(bucket).copied().unwrap_or_default()).json(seconds),
            })
        }).collect();
        json!({
            "interval": self.totals.delta_since(previous.totals).json(seconds),
            "actors": self.actors,
            "boards": boards,
            "search_buckets": buckets,
        })
    }
}

pub fn broker_json(metrics: BrokerMetrics, batch_capacity: usize) -> Value {
    let average = if metrics.batches > 0 {
        metrics.positions as f64 / metrics.batches as f64
    } else {
        0.0
    };
    json!({
        "requests": metrics.requests, "positions": metrics.positions,
        "batches": metrics.batches, "failures": metrics.failures,
        "average_batch_size": average,
        "batch_capacity": batch_capacity,
        "batch_fill_fraction": if batch_capacity > 0 { average / batch_capacity as f64 } else { 0.0 },
        "maximum_batch_since_start": metrics.maximum_batch,
        "full_batches": metrics.full_batches, "deadline_batches": metrics.deadline_batches,
        "drain_batches": metrics.drain_batches,
        "encoding_wall_seconds_summed": metrics.encoding_nanoseconds as f64 / 1e9,
        "queue_position_seconds": metrics.queue_nanoseconds as f64 / 1e9,
        "channel_position_seconds": metrics.channel_nanoseconds as f64 / 1e9,
        "broker_queue_position_seconds": metrics.broker_queue_nanoseconds as f64 / 1e9,
        "inference_wall_seconds_summed": metrics.inference_nanoseconds as f64 / 1e9,
        "input_packing_wall_seconds_summed": metrics.input_packing_nanoseconds as f64 / 1e9,
        "session_run_wall_seconds_summed": metrics.session_run_nanoseconds as f64 / 1e9,
        "output_materialization_wall_seconds_summed": metrics.output_materialization_nanoseconds as f64 / 1e9,
        "idle_request_wait_seconds": metrics.idle_request_wait_nanoseconds as f64 / 1e9,
        "overlap_request_wait_seconds": metrics.overlap_request_wait_nanoseconds as f64 / 1e9,
        "batch_collection_seconds": metrics.batch_collection_nanoseconds as f64 / 1e9,
        "batch_submission_seconds": metrics.batch_submission_nanoseconds as f64 / 1e9,
        "completion_wait_seconds": metrics.completion_wait_nanoseconds as f64 / 1e9,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(radius: f64, ply: u32, soft_resigned: bool) -> GenerationProgress {
        GenerationProgress {
            radius,
            ply,
            maximum_plies: 12,
            soft_resigned,
            policy_eligible: true,
            simulations: 10,
            evaluations: 8,
            search_duration: Duration::from_millis(5),
        }
    }

    #[test]
    fn in_flight_work_is_distinct_from_published_data_and_intervals_do_not_recount() {
        let telemetry = GenerationTelemetry::new(2);
        telemetry.searched(0, progress(1.0 / 38.0, 0, false));
        let before = telemetry.snapshot();
        assert_eq!(before.totals.searched_plies, 1);
        assert_eq!(before.totals.published_samples, 0);
        telemetry.searched(1, progress(1.0 / 18.0, 8, true));
        telemetry.published(1, 1.0 / 18.0, 12, true, true, Duration::from_millis(2));
        let after = telemetry.snapshot();
        let interval = after.interval_json(&before, 2.0);
        assert_eq!(interval["interval"]["searched_plies"], 1);
        assert_eq!(interval["interval"]["published_samples"], 12);
        assert_eq!(interval["interval"]["published_games_per_hour"], 1800.0);
        assert_eq!(
            after.interval_json(&after, 2.0)["interval"]["searched_plies"],
            0
        );
        assert_eq!(after.totals.capped_games, 1);
        assert_eq!(after.totals.soft_resigned_games, 1);
        assert_eq!(after.boards[&18].published_samples, 12);
        assert_eq!(after.boards[&38].published_samples, 0);
    }

    #[test]
    fn stages_follow_each_games_cap_and_soft_tails_have_separate_buckets() {
        for (ply, stage) in [(0, 0), (3, 0), (4, 1), (7, 1), (8, 2), (11, 2)] {
            assert_eq!(
                Bucket::from_progress(progress(1.0 / 38.0, ply, false)).stage,
                stage
            );
        }
        let telemetry = GenerationTelemetry::new(1);
        telemetry.searched(0, progress(1.0 / 38.0, 8, false));
        telemetry.searched(0, progress(1.0 / 38.0, 9, true));
        assert_eq!(telemetry.snapshot().buckets.len(), 2);
        telemetry.published(0, 1.0 / 38.0, 0, true, true, Duration::ZERO);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.totals.published_games, 0);
        assert_eq!(snapshot.totals.capped_games, 0);
        assert_eq!(snapshot.totals.discarded_games, 1);
    }

    #[test]
    fn broker_fill_uses_interval_counts_and_empty_intervals_are_finite() {
        let before = BrokerMetrics {
            batches: 10,
            positions: 100,
            ..BrokerMetrics::default()
        };
        let after = BrokerMetrics {
            batches: 12,
            positions: 116,
            ..BrokerMetrics::default()
        };
        let json = broker_json(after.delta_since(before), 32);
        assert_eq!(json["average_batch_size"], 8.0);
        assert_eq!(json["batch_fill_fraction"], 0.25);
        assert_eq!(
            broker_json(BrokerMetrics::default(), 0)["batch_fill_fraction"],
            0.0
        );
        assert_eq!(
            Counters::default().json(0.0)["searched_plies_per_second"],
            0.0
        );
    }
}
