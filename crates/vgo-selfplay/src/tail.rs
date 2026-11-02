//! The search tail: less search for games that look decided.
//!
//! A game that looks decided plays on at a cheaper budget, so search moves to
//! positions where it can matter, and the game still ends in a real result. This
//! rule only ever chooses a budget. It never names a winner.
//!
//! How sure a position has to be depends on the board and on how far the game has
//! gone. On big boards a confident early call was wrong about one time in five at
//! any threshold, while the same call past halfway was right 95% of the time or
//! better, so one threshold cannot suit both. A [`TailSchedule`] holds a threshold
//! for each board and stage, built from recent games by
//! `training/loop/tail-calibration.py`, and a stage with no threshold never starts
//! a tail. With no schedule, every game gets full search.
//!
//! A tail ends if the trailing side recovers. The exit sits well below the entry
//! so a game near the line doesn't flip back and forth.

use serde_json::Value;
use vgo_core::Color;

/// One board band's thresholds, one per stage.
#[derive(Clone, Debug, PartialEq)]
struct BoardThresholds {
    up_to_units: f64,
    thresholds: Vec<Option<f64>>,
}

/// The trailing value that starts a tail, by board and stage.
#[derive(Clone, Debug, PartialEq)]
pub struct TailSchedule {
    /// Where each stage starts, as a share of the board's ply cap. Before the
    /// first entry, no game can start a tail.
    stages: Vec<f64>,
    boards: Vec<BoardThresholds>,
}

impl TailSchedule {
    /// Parses what `tail-calibration.py` writes. Anything malformed is an error
    /// rather than an empty schedule: a run that silently loses its tail costs
    /// about a quarter of its search with nothing else looking wrong.
    pub fn parse(text: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("tail schedule: {error}"))?;
        let stages = root["stages"]
            .as_array()
            .ok_or("tail schedule has no stages")?
            .iter()
            .map(|stage| stage.as_f64().ok_or("a tail stage is not a number"))
            .collect::<Result<Vec<_>, _>>()?;
        if stages.is_empty()
            || !stages.is_sorted()
            || stages[0] < 0.0
            || stages[stages.len() - 1] > 1.0
        {
            return Err("tail stages must ascend within [0, 1]".to_owned());
        }
        let boards = root["boards"]
            .as_array()
            .ok_or("tail schedule has no boards")?
            .iter()
            .map(|board| parse_board(board, stages.len()))
            .collect::<Result<Vec<_>, _>>()?;
        if boards.is_empty() || !boards.is_sorted_by(|a, b| a.up_to_units < b.up_to_units) {
            return Err("tail boards must ascend".to_owned());
        }
        Ok(Self { stages, boards })
    }

    /// The threshold for this board and stage, if one is open there.
    fn threshold(&self, radius: f64, progress: f64) -> Option<f64> {
        let stage = self.stages.iter().rposition(|&start| progress >= start)?;
        let units = 1.0 / radius;
        let board = self
            .boards
            .iter()
            .find(|board| units <= board.up_to_units)?;
        board.thresholds[stage]
    }

    fn lowest_threshold(&self) -> Option<f64> {
        self.boards
            .iter()
            .flat_map(|board| board.thresholds.iter().flatten())
            .copied()
            .reduce(f64::min)
    }
}

fn parse_board(board: &Value, stages: usize) -> Result<BoardThresholds, String> {
    // Null means no upper limit, for the largest band.
    let up_to_units = board["up_to_units"].as_f64().unwrap_or(f64::INFINITY);
    let thresholds = board["thresholds"]
        .as_array()
        .ok_or("a tail board has no thresholds")?
        .iter()
        .map(|threshold| match threshold {
            Value::Null => Ok(None),
            other => match other.as_f64() {
                Some(value) if value > 0.0 && value < 1.0 => Ok(Some(value)),
                _ => Err(format!("{other} is not a tail threshold")),
            },
        })
        .collect::<Result<Vec<_>, _>>()?;
    if thresholds.len() != stages {
        return Err(format!(
            "a tail board has {} thresholds for {stages} stages",
            thresholds.len()
        ));
    }
    Ok(BoardThresholds {
        up_to_units,
        thresholds,
    })
}

/// Everything the tail needs, fixed for a run.
#[derive(Clone, Debug, PartialEq)]
pub struct TailSettings {
    /// `None` means full search throughout.
    pub schedule: Option<TailSchedule>,
    /// Simulations per ply in the tail.
    pub simulations: u32,
    /// Own turns in a row a side must trail past the threshold to start a tail.
    pub window: u32,
    /// A tail ends when the trailing side's value rises above this, negated.
    pub exit_value: f64,
    /// Own turns in a row past the exit that end a tail.
    pub exit_window: u32,
}

impl TailSettings {
    /// The exit must sit below every entry, or a tail could end as it starts.
    pub fn validate(&self) -> Result<(), String> {
        if self.simulations == 0 || self.window == 0 || self.exit_window == 0 {
            return Err("tail simulations and windows must be positive".to_owned());
        }
        match self
            .schedule
            .as_ref()
            .and_then(TailSchedule::lowest_threshold)
        {
            Some(lowest) if self.exit_value >= lowest => Err(format!(
                "tail exit {} must be below the lowest threshold {lowest}",
                self.exit_value
            )),
            _ => Ok(()),
        }
    }
}

/// One game's position in the rule.
pub struct TailTracker<'a> {
    settings: &'a TailSettings,
    radius: f64,
    maximum_plies: u32,
    // Counted per seat, because a value that favours one side reads the other way
    // on the next ply, and a shared count would be reset by the leader's turn.
    streak: [u32; 2],
    trailing: Option<usize>,
    recovery: u32,
}

impl<'a> TailTracker<'a> {
    #[must_use]
    pub fn new(settings: &'a TailSettings, radius: f64, maximum_plies: u32) -> Self {
        Self {
            settings,
            radius,
            maximum_plies,
            streak: [0; 2],
            trailing: None,
            recovery: 0,
        }
    }

    /// Whether the next ply is in the tail.
    #[must_use]
    pub fn active(&self) -> bool {
        self.trailing.is_some()
    }

    /// Takes the root value of the ply just searched, from Black's view.
    pub fn observe(&mut self, ply: u32, to_move: Color, root_black_value: f64) {
        let seat = usize::from(to_move == Color::White);
        let value = if to_move == Color::Black {
            root_black_value
        } else {
            -root_black_value
        };
        if let Some(trailing) = self.trailing {
            if seat == trailing {
                self.watch_for_recovery(value);
            }
            return;
        }
        let Some(schedule) = &self.settings.schedule else {
            return;
        };
        let progress = f64::from(ply) / f64::from(self.maximum_plies);
        let behind = schedule
            .threshold(self.radius, progress)
            .is_some_and(|threshold| value <= -threshold);
        self.streak[seat] = if behind { self.streak[seat] + 1 } else { 0 };
        if self.streak[seat] >= self.settings.window {
            self.trailing = Some(seat);
            self.streak = [0; 2];
        }
    }

    fn watch_for_recovery(&mut self, value: f64) {
        self.recovery = if value > -self.settings.exit_value {
            self.recovery + 1
        } else {
            0
        };
        if self.recovery >= self.settings.exit_window {
            self.trailing = None;
            self.recovery = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RADIUS: f64 = 1.0 / 18.0;

    /// Thresholds by stage for every board, starting a quarter of the way in.
    fn settings(thresholds: &str) -> TailSettings {
        let schedule = format!(
            r#"{{"stages": [0.25, 0.5], "boards": [{{"up_to_units": null, "thresholds": {thresholds}}}]}}"#
        );
        TailSettings {
            schedule: Some(TailSchedule::parse(&schedule).unwrap()),
            simulations: 8,
            window: 2,
            exit_value: 0.6,
            exit_window: 2,
        }
    }

    /// Feeds Black-relative values ply by ply and returns the plies in the tail.
    fn tail_plies(settings: &TailSettings, values: &[f64]) -> Vec<u32> {
        let mut tracker = TailTracker::new(settings, RADIUS, 100);
        let mut tail = Vec::new();
        for (ply, &value) in values.iter().enumerate() {
            let ply = u32::try_from(ply).unwrap();
            if tracker.active() {
                tail.push(ply);
            }
            let to_move = if ply % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            tracker.observe(ply, to_move, value);
        }
        tail
    }

    fn padded(mut values: Vec<f64>) -> Vec<f64> {
        let mut game = vec![0.0; 26];
        game.append(&mut values);
        game
    }

    #[test]
    fn no_schedule_means_full_search() {
        let mut settings = settings("[0.9, 0.9]");
        settings.schedule = None;
        assert!(tail_plies(&settings, &[-1.0; 80]).is_empty());
    }

    #[test]
    fn nothing_starts_before_the_first_stage() {
        // Sustained and extreme, but only 25 plies into a 100-ply cap.
        assert!(tail_plies(&settings("[0.9, 0.9]"), &[-1.0; 25]).is_empty());
    }

    #[test]
    fn a_sustained_deficit_starts_a_tail_after_the_window() {
        // Black trails on its own turns, which are the even plies.
        let tail = tail_plies(&settings("[0.9, 0.9]"), &[-1.0; 32]);
        assert_eq!(tail.first(), Some(&29)); // Plies 26 and 28 fill the window.
    }

    #[test]
    fn either_side_can_trail() {
        // Black leads, so White trails on the odd plies: 25 and 27 fill the window.
        assert_eq!(
            tail_plies(&settings("[0.9, 0.9]"), &[1.0; 32]).first(),
            Some(&28)
        );
    }

    #[test]
    fn one_good_turn_resets_the_streak() {
        // Black's plies 26, 28, 30 and 32 read -1, 0, -1 and -1.
        let values = padded(vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]);
        assert_eq!(
            tail_plies(&settings("[0.9, 0.9]"), &values).first(),
            Some(&33)
        );
    }

    #[test]
    fn a_closed_stage_starts_no_tail() {
        // Open only past halfway.
        assert_eq!(
            tail_plies(&settings("[null, 0.9]"), &[-1.0; 56]).first(),
            Some(&53)
        );
    }

    #[test]
    fn the_threshold_follows_the_stage() {
        // Value -0.8 clears the later stage's bar and not the earlier one's.
        assert_eq!(
            tail_plies(&settings("[0.95, 0.75]"), &[-0.8; 56]).first(),
            Some(&53)
        );
    }

    #[test]
    fn a_recovery_ends_the_tail_and_a_new_deficit_starts_another() {
        let mut values = vec![-1.0; 30]; // In the tail from ply 29.
        values.extend_from_slice(&[0.0; 4]); // Black's plies 30 and 32 recover.
        values.extend_from_slice(&[-1.0; 6]); // Plies 34 and 36 start it again.
        let tail = tail_plies(&settings("[0.9, 0.9]"), &values);
        assert_eq!(tail, vec![29, 30, 31, 32, 37, 38, 39]);
    }

    #[test]
    fn one_good_turn_does_not_end_the_tail() {
        // Black's plies alternate between recovered and not, so the count restarts.
        let mut values = vec![-1.0; 30];
        values.extend_from_slice(&[0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
        let tail = tail_plies(&settings("[0.9, 0.9]"), &values);
        assert_eq!(tail, (29..38).collect::<Vec<_>>());
    }

    #[test]
    fn thresholds_are_read_per_board() {
        let schedule = TailSchedule::parse(
            r#"{"stages": [0.25],
                "boards": [{"up_to_units": 18.5, "thresholds": [0.8]},
                           {"up_to_units": null, "thresholds": [null]}]}"#,
        )
        .unwrap();
        assert_eq!(schedule.threshold(1.0 / 18.0, 0.5), Some(0.8));
        assert_eq!(schedule.threshold(1.0 / 38.0, 0.5), None);
        assert_eq!(schedule.threshold(1.0 / 18.0, 0.1), None);
    }

    /// The calibrator replays streaks offline, so it reads the same cases.
    #[test]
    fn the_live_rule_matches_the_shared_cases() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../training/tests/fixtures/tail-rule.json"
        ))
        .unwrap();
        let threshold = fixture["threshold"].as_f64().unwrap();
        let schedule = format!(
            r#"{{"stages": [0.25], "boards": [{{"up_to_units": null, "thresholds": [{threshold}]}}]}}"#
        );
        let settings = TailSettings {
            schedule: Some(TailSchedule::parse(&schedule).unwrap()),
            simulations: 8,
            window: u32::try_from(fixture["window"].as_u64().unwrap()).unwrap(),
            exit_value: 0.6,
            exit_window: 2,
        };
        let cap = u32::try_from(fixture["maximum_plies"].as_u64().unwrap()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut values = Vec::new();
            for span in case["spans"].as_array().unwrap() {
                let span = span.as_array().unwrap();
                let count = span[1].as_u64().unwrap() - span[0].as_u64().unwrap();
                values.extend(std::iter::repeat_n(
                    span[2].as_f64().unwrap(),
                    count as usize,
                ));
            }
            assert_eq!(cap, 100, "tail_plies assumes this cap");
            let first = tail_plies(&settings, &values).first().copied();
            assert_eq!(
                first.map(u64::from),
                case["first_tail_ply"].as_u64(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn a_malformed_schedule_is_an_error() {
        for text in [
            "not json",
            r#"{"boards": []}"#,
            r#"{"stages": [0.5, 0.25], "boards": [{"thresholds": [0.9, 0.9]}]}"#,
            r#"{"stages": [0.25], "boards": []}"#,
            r#"{"stages": [0.25], "boards": [{"thresholds": [0.9, 0.9]}]}"#,
            r#"{"stages": [0.25], "boards": [{"thresholds": [1.5]}]}"#,
        ] {
            assert!(TailSchedule::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn the_exit_must_sit_below_every_threshold() {
        let mut bad = settings("[0.9, 0.5]");
        assert!(bad.validate().is_err());
        bad.exit_value = 0.4;
        assert!(bad.validate().is_ok());
    }
}
