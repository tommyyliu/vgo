#![forbid(unsafe_code)]

use std::{
    collections::HashMap,
    env,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use vgo_core::{Color, GameEvent, Phase, Position};
use vgo_search::{Action, SearchConfig, SearchStats, search};

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
        add_search_stats(&mut self.search, game.search);
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

fn add_search_stats(total: &mut SearchStats, next: SearchStats) {
    total.simulations += next.simulations;
    total.expanded_nodes += next.expanded_nodes;
    total.generated_candidates += next.generated_candidates;
    total.terminal_leaves += next.terminal_leaves;
    total.depth_limited_leaves += next.depth_limited_leaves;
    total.maximum_depth = total.maximum_depth.max(next.maximum_depth);
}

fn position_hash(position: &Position) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut word = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    word(position.radius().to_bits());
    word(u64::from(position.consecutive_passes()));
    word(match position.to_move() {
        Color::Black => 0,
        Color::White => 1,
    });
    word(match position.phase() {
        Phase::Playing => 0,
        Phase::Finished => 1,
    });
    let mut stones = position
        .stones()
        .iter()
        .map(|stone| {
            (
                stone.x.to_bits(),
                stone.y.to_bits(),
                match stone.color {
                    Color::Black => 0,
                    Color::White => 1,
                },
            )
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    for (x, y, color) in stones {
        word(x);
        word(y);
        word(color);
    }
    hash
}

fn play_game(config: MatchConfig, first_color: Color, pair_seed: u64) -> GameReport {
    let started = Instant::now();
    let mut position = Position::new(config.radius, Vec::new(), Color::Black);
    let mut report = GameReport::default();
    let mut seen = HashMap::new();
    seen.insert(position_hash(&position), 1_u32);

    for ply in 0..config.maximum_plies {
        let first_turn = position.to_move() == first_color;
        let simulations = if first_turn {
            config.first_simulations
        } else {
            config.second_simulations
        };
        let result = search(&position, SearchConfig::canary(simulations), pair_seed);
        add_search_stats(&mut report.search, result.stats);
        let mut selected = None;
        for action in result.actions_by_preference(position.to_move()) {
            let transition = action.apply(&position);
            if transition.position.phase() == Phase::Finished
                || !seen.contains_key(&position_hash(&transition.position))
            {
                selected = Some((action, transition));
                break;
            }
            report.repetition_avoids += 1;
        }
        let (action, transition) = selected.unwrap_or_else(|| {
            let action = Action::Pass;
            (action, action.apply(&position))
        });
        if action == Action::Pass {
            report.passes += 1;
        }
        report.captures += transition.captured as u64;
        report.self_captures += transition
            .events
            .iter()
            .filter_map(|event| match event {
                GameEvent::SelfCapture { count, .. } => Some(*count as u64),
                _ => None,
            })
            .sum::<u64>();
        position = transition.position;
        report.plies = ply + 1;
        if position.phase() == Phase::Finished {
            report.completed = true;
            let outcome = transition.analysis.outcome;
            report.first_score = match outcome.winner {
                Some(winner) if winner == first_color => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
            break;
        }
        let count = seen.entry(position_hash(&position)).or_insert(0);
        if *count > 0 {
            report.repetitions += 1;
        }
        *count += 1;
    }
    report.elapsed = started.elapsed();
    report
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

fn parse_value<T: std::str::FromStr>(arguments: &[String], name: &str, default: T) -> T {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let config = MatchConfig {
        pairs: parse_value(&arguments, "--pairs", 5),
        first_simulations: parse_value(&arguments, "--first", 1_000),
        second_simulations: parse_value(&arguments, "--second", 10),
        maximum_plies: parse_value(&arguments, "--max-plies", 48),
        radius: parse_value(&arguments, "--radius", 1.0 / 6.0),
        threads: parse_value(
            &arguments,
            "--threads",
            thread::available_parallelism().map_or(1, usize::from),
        ),
        seed: parse_value(&arguments, "--seed", 1),
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
    use super::{MatchConfig, play_game, position_hash, wilson_interval};
    use vgo_core::{Color, Position, Stone};

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
    fn repetition_hash_is_order_independent_but_color_absolute() {
        let stones = vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.75, 0.75, Color::White),
        ];
        let mut reversed = stones.clone();
        reversed.reverse();
        let swapped = stones
            .iter()
            .map(|stone| Stone::new(stone.x, stone.y, stone.color.other()))
            .collect();
        let first = Position::new(0.1, stones, Color::Black);
        let reordered = Position::new(0.1, reversed, Color::Black);
        let color_swapped = Position::new(0.1, swapped, Color::White);

        assert_eq!(position_hash(&first), position_hash(&reordered));
        assert_ne!(position_hash(&first), position_hash(&color_swapped));
    }
}
