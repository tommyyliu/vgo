//! A round-robin over many models, with every game of every pairing in one
//! work queue.
//!
//! `vgo-arena` plays one pairing at a time: it spawns `--threads` workers and
//! they pull from a counter that stops at `pairs * 2`. Four pairs is eight
//! games, so sixty-four threads exit immediately and the match proceeds at
//! whatever eight concurrent games achieve. Running a fifteen-pairing
//! round-robin as fifteen such matches leaves the machine idle for most of it.
//!
//! Here the round is one queue. Every (pairing, pair, colour) triple is
//! enumerated up front and `--concurrency` workers pull from it until it
//! drains, so the last pairing's games overlap the first's. Each model gets one
//! `BatchedEvaluator` -- an inference provider with its own broker and queue --
//! and workers address whichever two a game needs.
//!
//! Batches run thinner than a single-pairing arena's, because attention is
//! split across every model rather than concentrated on two. That is the
//! deliberate trade: generation profiling put inference at 2.3% of lane time
//! against 23% for CPU rasterization, so keeping many games in flight matters
//! more than filling any one batch.
//!
//! Records are the `vgo.arena.v1` shape the Bradley-Terry fit already reads,
//! and each is written and flushed as its pairing completes rather than after
//! the last worker joins. A tournament runs for hours; tallying at the end
//! means a crash loses all of it, there is nothing to rate until the very end,
//! and the output file stays empty long enough to be mistaken for a hung
//! process. Readers parse by brace depth, so a file still being appended to
//! is valid input.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use vgo_core::{Color, Outcome, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::{EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::{ADJUDICATION_PLIES, award_by_area, play_game as run_playout};

#[derive(Parser, Debug)]
#[command(about = "Round-robin tournament with every game in one queue")]
struct Arguments {
    /// Repeatable. Every model plays every other model.
    #[arg(long)]
    model: Vec<PathBuf>,
    /// Add the naive evaluator to the field.
    ///
    /// It loses nearly every game to a trained checkpoint, so it contributes
    /// little to separating them -- a pairing decided 8-0 carries almost no
    /// information. Its value is that it is the one player whose strength does
    /// not depend on any run: anchoring on it puts every tournament's ratings
    /// on the same absolute scale, where anchoring on "the weakest checkpoint
    /// present" moves the zero every time the field changes.
    #[arg(long, default_value_t = false)]
    include_naive: bool,
    /// Play one reference player against every other, and nobody else.
    ///
    /// The default complete round-robin spends most of its games on pairings a
    /// previous tournament already covered. When the field is densely rated
    /// already and only the tie to a reference is missing, a star costs
    /// `field - 1` pairings instead of `field * (field - 1) / 2`.
    ///
    /// The reference is naive when `--include-naive` is set, otherwise the
    /// first `--model`. Note that a reference which never wins does not gain a
    /// stable rating from more games: a winless record has no finite
    /// maximum-likelihood estimate, so the fitted gap keeps widening with the
    /// game count instead of converging. Point this at a weak *checkpoint*,
    /// which loses most games but not all, rather than at naive.
    #[arg(long, default_value_t = false)]
    star: bool,
    /// Colour-swapped pairs per pairing; each pair is two games.
    #[arg(long, default_value_t = 4)]
    pairs: usize,
    /// Games in flight at once, across all pairings. Bounded by memory: each
    /// game holds a search tree, and at 128x128 policy resolution a node
    /// carries a 336 KB fine grid, so 80 games at 1600 simulations is a 41 GB
    /// worst case.
    #[arg(long, default_value_t = 64)]
    concurrency: usize,
    #[arg(long, default_value_t = 1600)]
    simulations: u32,
    #[arg(long, default_value_t = 16)]
    coarse_pool: usize,
    #[arg(long, default_value_t = 105)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 128)]
    resolution: usize,
    #[arg(long, default_value_t = 128)]
    policy_resolution: usize,
    #[arg(long, default_value_t = 0.055_714_285_714_285_716)]
    radius: f64,
    #[arg(long, default_value_t = 0.034)]
    komi: f64,
    #[arg(long, default_value_t = 4)]
    leaf_batch: usize,
    #[arg(long, default_value_t = 64)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value = "compact")]
    raster_kind: RasterKind,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, default_value_t = true)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

/// A pairing's running score, from the first model's side.
#[derive(Clone, Copy, Default)]
struct Tally {
    first_wins: usize,
    second_wins: usize,
    draws: usize,
    games: usize,
}

impl Tally {
    fn record(&mut self, first_is_black: bool, outcome: Option<Outcome>) {
        self.games += 1;
        let Some(outcome) = outcome else { return };
        let first_colour = if first_is_black {
            Color::Black
        } else {
            Color::White
        };
        match outcome.winner {
            Some(colour) if colour == first_colour => self.first_wins += 1,
            Some(_) => self.second_wins += 1,
            None => self.draws += 1,
        }
    }

    fn decided(&self) -> usize {
        self.first_wins + self.second_wins + self.draws
    }
}

/// Write one pairing's `vgo.arena.v1` record and flush it.
///
/// Flushing matters more than it looks: stdout is a pipe to a file here, so a
/// buffered record is not on disk, and the whole point of writing early is that
/// a run which dies partway still leaves usable results behind.
fn emit(first: &str, second: &str, tally: &Tally) {
    let decided = tally.decided();
    let score = if decided > 0 {
        (tally.first_wins as f64 + 0.5 * tally.draws as f64) / decided as f64
    } else {
        0.0
    };
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        concat!(
            "{{\n",
            "  \"schema\": \"vgo.arena.v1\",\n",
            "  \"candidate_model\": \"{}\",\n",
            "  \"opponent_model\": \"{}\",\n",
            "  \"games\": {},\n",
            "  \"completed\": {},\n",
            "  \"candidate_wins\": {},\n",
            "  \"candidate_losses\": {},\n",
            "  \"draws\": {},\n",
            "  \"candidate_score\": {:.6}\n",
            "}}"
        ),
        first,
        second,
        tally.games,
        decided,
        tally.first_wins,
        tally.second_wins,
        tally.draws,
        score,
    );
    let _ = out.flush();
}

/// One game: which pairing, which seats, which seed.
#[derive(Clone, Copy)]
struct Assignment {
    pairing: usize,
    /// Model index taking Black.
    black: usize,
    /// Model index taking White.
    white: usize,
    /// Whether the pairing's first model is Black in this game, so the tally
    /// knows which side a win belongs to.
    first_is_black: bool,
    seed: u64,
}

fn search_config(arguments: &Arguments) -> SearchConfig {
    let mut config = SearchConfig::canary(arguments.simulations);
    config.coarse_pool = arguments.coarse_pool;
    config.leaf_batch = arguments.leaf_batch.max(1);
    config
}

/// A seat in the field: a loaded model, or the built-in naive evaluator.
enum Player {
    Model(BatchedEvaluator),
    Naive(NaiveEvaluator),
}

impl Player {
    fn evaluator(&self) -> &dyn Evaluator {
        match self {
            Player::Model(evaluator) => evaluator,
            Player::Naive(evaluator) => evaluator,
        }
    }
}

fn play(
    black: &dyn Evaluator,
    white: &dyn Evaluator,
    seed: u64,
    arguments: &Arguments,
    config: SearchConfig,
) -> Result<Option<Outcome>, EvaluationError> {
    // Only the closing positions are needed to adjudicate a capped game, so
    // this stays bounded rather than retaining the whole game.
    let mut closing: VecDeque<Position> = VecDeque::with_capacity(ADJUDICATION_PLIES);
    let playout = run_playout(
        Position::new(arguments.radius, Vec::new(), Color::Black).with_komi(arguments.komi),
        arguments.maximum_plies,
        |position, _| {
            let evaluator: &dyn Evaluator = match position.to_move() {
                Color::Black => black,
                Color::White => white,
            };
            search_with_evaluator(position, config, seed, evaluator)
        },
        |step| {
            if closing.len() == ADJUDICATION_PLIES {
                closing.pop_front();
            }
            closing.push_back(step.transition.position.clone());
        },
    )?;
    // A game that runs out of plies is awarded by area, the same rule self-play
    // uses. Discarding it is not neutral: the games that survive a cap are the
    // ones that resolve quickly, which reflects style rather than strength.
    Ok(playout
        .outcome
        .or_else(|| closing.back().map(award_by_area)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let field_size = arguments.model.len() + usize::from(arguments.include_naive);
    if field_size < 2 {
        return Err("need at least two players (--model paths, --include-naive)".into());
    }
    if arguments.pairs == 0 || arguments.concurrency == 0 {
        return Err("--pairs and --concurrency must be positive".into());
    }

    // One provider per model, loaded once and shared by reference. At 32 MB a
    // model this is nothing against the card, and it means a twelve-model round
    // costs twelve loads rather than one per pairing.
    let mut evaluators = Vec::with_capacity(field_size);
    let mut labels: Vec<String> = Vec::with_capacity(field_size);
    for model in &arguments.model {
        labels.push(model.display().to_string());
        let service = OnnxBatchService::load(&OnnxServiceConfig {
            model: model.clone(),
            raster: RasterConfig::square_of(arguments.resolution, arguments.raster_kind),
            policy: Some(RasterConfig::square(arguments.policy_resolution)),
            maximum_batch: arguments.maximum_batch,
            provider: arguments.provider,
            device_id: arguments.device_id,
            fp16: arguments.fp16,
            cache_directory: arguments.cache_directory.clone(),
        })?;
        evaluators.push(Player::Model(BatchedEvaluator::spawn(
            BrokerConfig {
                maximum_delay: Duration::from_millis(arguments.delay_ms),
                queue_capacity: (arguments.concurrency * 2).max(arguments.maximum_batch * 2),
            },
            service,
        )?));
    }
    // Naive goes last so adding it does not renumber the models, and so its
    // record path is a bare token the rating scripts can recognise.
    if arguments.include_naive {
        evaluators.push(Player::Naive(NaiveEvaluator));
        labels.push("naive".to_string());
    }
    let evaluators = Arc::new(evaluators);
    let labels = Arc::new(labels);

    // Enumerate every game before playing any: that is what lets the last
    // pairing's games run alongside the first's.
    let mut pairings = Vec::new();
    let mut assignments = Vec::new();
    // In star mode the reference is the last seat when naive is in the field
    // (it is appended last) and seat 0 otherwise.
    let reference = if arguments.include_naive {
        field_size - 1
    } else {
        0
    };
    for first in 0..field_size {
        for second in (first + 1)..field_size {
            if arguments.star && first != reference && second != reference {
                continue;
            }
            let pairing = pairings.len();
            pairings.push((first, second));
            for pair in 0..arguments.pairs {
                // Both halves of a pair share a seed so they face the same
                // game with the colours swapped; no pairing is decided by who
                // moved first.
                let seed = arguments
                    .seed
                    .wrapping_add((pairing as u64).wrapping_mul(1_000_003))
                    .wrapping_add(pair as u64);
                assignments.push(Assignment {
                    pairing,
                    black: first,
                    white: second,
                    first_is_black: true,
                    seed,
                });
                assignments.push(Assignment {
                    pairing,
                    black: second,
                    white: first,
                    first_is_black: false,
                    seed,
                });
            }
        }
    }
    let workers = arguments.concurrency.min(assignments.len());
    eprintln!(
        "tournament: {} models, {} pairings{}, {} games, {} in flight",
        field_size,
        pairings.len(),
        if arguments.star {
            format!(" (star on {})", labels[reference])
        } else {
            String::new()
        },
        assignments.len(),
        workers,
    );

    let games_per_pairing = arguments.pairs * 2;
    let arguments = Arc::new(arguments);
    let assignments = Arc::new(assignments);
    let pairings = Arc::new(pairings);
    // Tallies are shared rather than per-worker so a pairing's record can be
    // written the moment its last game lands, instead of after every worker
    // joins. A run that dies partway then still leaves every completed pairing
    // on disk, and the file is readable while the run is in progress.
    let tallies = Arc::new(Mutex::new(vec![Tally::default(); pairings.len()]));
    let next = Arc::new(AtomicUsize::new(0));
    // Games finished, reported as they land. Results are only tallied after
    // every worker joins, so without this the process is silent for the whole
    // run -- and a silent hours-long job is indistinguishable from a hung one
    // by anything short of inspecting its threads.
    let finished = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let arguments = Arc::clone(&arguments);
        let assignments = Arc::clone(&assignments);
        let evaluators = Arc::clone(&evaluators);
        let next = Arc::clone(&next);
        let finished = Arc::clone(&finished);
        let pairings = Arc::clone(&pairings);
        let tallies = Arc::clone(&tallies);
        let labels = Arc::clone(&labels);
        let total = assignments.len();
        handles.push(thread::spawn(move || {
            let config = search_config(&arguments);
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(assignment) = assignments.get(index).copied() else {
                    break;
                };
                let outcome = play(
                    evaluators[assignment.black].evaluator(),
                    evaluators[assignment.white].evaluator(),
                    assignment.seed,
                    &arguments,
                    config,
                )?;
                {
                    let mut tallies = tallies.lock().expect("tournament tallies");
                    let tally = &mut tallies[assignment.pairing];
                    tally.record(assignment.first_is_black, outcome);
                    if tally.games == games_per_pairing {
                        let (first, second) = pairings[assignment.pairing];
                        // Written under the lock so concurrent completions
                        // cannot interleave two records on stdout.
                        emit(&labels[first], &labels[second], tally);
                    }
                }
                let done = finished.fetch_add(1, Ordering::Relaxed) + 1;
                // Every game early, then sparsely: the first few are what say
                // the run is alive at all, and after that a heartbeat is enough.
                if done <= 5 || done % 10 == 0 || done == total {
                    let elapsed = started.elapsed().as_secs_f64();
                    eprintln!(
                        "  {done}/{total} games, {elapsed:.0}s elapsed, \
                         {:.2}s/game, ~{:.0}s left",
                        elapsed / done as f64,
                        elapsed / done as f64 * (total - done) as f64,
                    );
                }
            }
            Ok::<_, EvaluationError>(())
        }));
    }

    for handle in handles {
        handle.join().expect("tournament worker")?;
    }

    // Every complete pairing has already been written. Anything left is a
    // pairing that lost games to an error, and it is reported rather than
    // emitted -- a partial record would be indistinguishable from a real one.
    let tallies = tallies.lock().expect("tournament tallies");
    let partial: Vec<_> = tallies
        .iter()
        .enumerate()
        .filter(|(_, tally)| tally.games != games_per_pairing)
        .collect();
    for (pairing, tally) in &partial {
        let (first, second) = pairings[*pairing];
        eprintln!(
            "  incomplete, not written: {} vs {} ({}/{} games)",
            labels[first], labels[second], tally.games, games_per_pairing,
        );
    }
    eprintln!(
        "tournament: {} of {} pairings written",
        pairings.len() - partial.len(),
        pairings.len(),
    );
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "tournament done: {} games in {:.0}s ({:.2}s/game)",
        assignments.len(),
        elapsed,
        elapsed / assignments.len() as f64,
    );
    Ok(())
}
