#![forbid(unsafe_code)]

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use vgo_core::{Color, Position};
use vgo_search::{SearchConfig, SearchStats, search};
use vgo_selfplay::{accumulate_search_stats, play_game as run_playout};

#[derive(Clone, Copy, Debug)]
struct MatchConfig {
    pairs: usize,
    first_simulations: u32,
    second_simulations: u32,
    maximum_plies: u32,
    radius: f64,
    threads: usize,
    seed: u64,
}

#[derive(Debug, Parser)]
struct Arguments {
    #[arg(long, default_value_t = 5)]
    pairs: usize,
    #[arg(long, default_value_t = 1_000)]
    first: u32,
    #[arg(long, default_value_t = 10)]
    second: u32,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct GameReport {
    first_score: f64,
    completed: bool,
    plies: u32,
    captures: u64,
    self_captures: u64,
    passes: u64,
    repetitions: u64,
    repetition_avoids: u64,
    search: SearchStats,
    elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct MatchReport {
    games: u64,
    completed: u64,
    first_wins: u64,
    second_wins: u64,
    draws: u64,
    first_points: f64,
    plies: u64,
    captures: u64,
    self_captures: u64,
    passes: u64,
    repetitions: u64,
    repetition_avoids: u64,
    search: SearchStats,
    game_time: Duration,
}

impl MatchReport {
    fn add(&mut self, game: GameReport) {
        self.games += 1;
        self.plies += u64::from(game.plies);
        self.captures += game.captures;
        self.self_captures += game.self_captures;
        self.passes += game.passes;
        self.repetitions += game.repetitions;
        self.repetition_avoids += game.repetition_avoids;
        self.game_time += game.elapsed;
        accumulate_search_stats(&mut self.search, game.search);
        if game.completed {
            self.completed += 1;
            self.first_points += game.first_score;
            if game.first_score == 1.0 {
                self.first_wins += 1;
            } else if game.first_score == 0.0 {
                self.second_wins += 1;
            } else {
                self.draws += 1;
            }
        }
    }

    fn print(self, config: MatchConfig, wall_time: Duration) {
        let score = if self.completed == 0 {
            0.0
        } else {
            self.first_points / self.completed as f64
        };
        let (lower, upper) = wilson_interval(self.first_points, self.completed);
        let truncated = self.games - self.completed;
        println!(
            concat!(
                "{{\n",
                "  \"first_simulations\": {},\n",
                "  \"second_simulations\": {},\n",
                "  \"pairs\": {},\n",
                "  \"games\": {},\n",
                "  \"completed\": {},\n",
                "  \"truncated\": {},\n",
                "  \"first_wins\": {},\n",
                "  \"second_wins\": {},\n",
                "  \"draws\": {},\n",
                "  \"first_score\": {:.6},\n",
                "  \"score_ci95\": [{:.6}, {:.6}],\n",
                "  \"average_plies\": {:.3},\n",
                "  \"captures\": {},\n",
                "  \"self_captures\": {},\n",
                "  \"passes\": {},\n",
                "  \"repetitions\": {},\n",
                "  \"repetition_avoids\": {},\n",
                "  \"search_simulations\": {},\n",
                "  \"evaluations\": {},\n",
                "  \"expanded_nodes\": {},\n",
                "  \"generated_candidates\": {},\n",
                "  \"terminal_leaves\": {},\n",
                "  \"depth_limited_leaves\": {},\n",
                "  \"maximum_search_depth\": {},\n",
                "  \"wall_seconds\": {:.6},\n",
                "  \"summed_game_seconds\": {:.6},\n",
                "  \"nodes_per_wall_second\": {:.3}\n",
                "}}"
            ),
            config.first_simulations,
            config.second_simulations,
            config.pairs,
            self.games,
            self.completed,
            truncated,
            self.first_wins,
            self.second_wins,
            self.draws,
            score,
            lower,
            upper,
            self.plies as f64 / self.games.max(1) as f64,
            self.captures,
            self.self_captures,
            self.passes,
            self.repetitions,
            self.repetition_avoids,
            self.search.simulations,
            self.search.evaluations,
            self.search.expanded_nodes,
            self.search.generated_candidates,
            self.search.terminal_leaves,
            self.search.depth_limited_leaves,
            self.search.maximum_depth,
            wall_time.as_secs_f64(),
            self.game_time.as_secs_f64(),
            self.search.expanded_nodes as f64 / wall_time.as_secs_f64().max(f64::EPSILON),
        );
    }
}

fn play_game(config: MatchConfig, first_color: Color, pair_seed: u64) -> GameReport {
    let started = Instant::now();
    let playout = run_playout(
        Position::new(config.radius, Vec::new(), Color::Black),
        config.maximum_plies,
        |position, _| {
            let simulations = if position.to_move() == first_color {
                config.first_simulations
            } else {
                config.second_simulations
            };
            Ok::<_, Infallible>(search(
                position,
                SearchConfig::canary(simulations),
                pair_seed,
            ))
        },
        |_| {},
    )
    .expect("naive search is infallible");
    let first_score = playout.outcome.map_or(0.0, |outcome| match outcome.winner {
        Some(winner) if winner == first_color => 1.0,
        Some(_) => 0.0,
        None => 0.5,
    });
    GameReport {
        first_score,
        completed: playout.completed(),
        plies: playout.stats.plies,
        captures: playout.stats.captures,
        self_captures: playout.stats.self_captures,
        passes: playout.stats.passes,
        repetitions: playout.stats.repetitions,
        repetition_avoids: playout.stats.repetition_avoids,
        search: playout.stats.search,
        elapsed: started.elapsed(),
    }
}

fn run_match(config: MatchConfig) -> MatchReport {
    let next_pair = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let workers = config.threads.max(1).min(config.pairs.max(1));
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let next_pair = Arc::clone(&next_pair);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            loop {
                let pair = next_pair.fetch_add(1, Ordering::Relaxed);
                if pair >= config.pairs {
                    break;
                }
                let seed = config.seed.wrapping_add(pair as u64);
                sender
                    .send(play_game(config, Color::Black, seed))
                    .expect("result receiver remains alive");
                sender
                    .send(play_game(config, Color::White, seed))
                    .expect("result receiver remains alive");
            }
        }));
    }
    drop(sender);
    let mut report = MatchReport::default();
    for game in receiver {
        report.add(game);
    }
    for handle in handles {
        handle.join().expect("self-play worker did not panic");
    }
    report
}

fn wilson_interval(points: f64, games: u64) -> (f64, f64) {
    if games == 0 {
        return (0.0, 1.0);
    }
    let count = games as f64;
    let proportion = points / count;
    let z = 1.959_963_984_540_054;
    let denominator = 1.0 + z * z / count;
    let center = (proportion + z * z / (2.0 * count)) / denominator;
    let margin = z
        * ((proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)).sqrt())
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn main() {
    let arguments = Arguments::parse();
    let config = MatchConfig {
        pairs: arguments.pairs,
        first_simulations: arguments.first,
        second_simulations: arguments.second,
        maximum_plies: arguments.maximum_plies,
        radius: arguments.radius,
        threads: arguments
            .threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(1, usize::from)),
        seed: arguments.seed,
    };
    assert!(config.pairs > 0, "--pairs must be positive");
    assert!(config.first_simulations > 0 && config.second_simulations > 0);
    assert!(config.maximum_plies > 0);
    let started = Instant::now();
    let report = run_match(config);
    report.print(config, started.elapsed());
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use vgo_core::Color;

    use super::{Arguments, MatchConfig, play_game, wilson_interval};

    #[test]
    fn short_game_run_is_bounded() {
        let config = MatchConfig {
            pairs: 1,
            first_simulations: 2,
            second_simulations: 2,
            maximum_plies: 4,
            radius: 1.0 / 6.0,
            threads: 1,
            seed: 1,
        };
        let report = play_game(config, Color::Black, 1);
        assert!(report.plies <= 4);
        assert_eq!(report.search.simulations, report.plies * 2);
    }

    #[test]
    fn empty_confidence_interval_is_uninformative() {
        assert_eq!(wilson_interval(0.0, 0), (0.0, 1.0));
    }

    #[test]
    fn malformed_cli_values_are_rejected() {
        assert!(Arguments::try_parse_from(["vgo-canary", "--pairs", "many"]).is_err());
    }
}
