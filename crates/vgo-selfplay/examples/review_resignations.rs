//! Were the resignations right? Reconstructs conceded games from a shard.
//!
//! A resignation records a real winner but no margin -- no area is scored, the
//! rule just names the conceding side the loser. So the shard alone cannot say
//! whether a concession was justified. This reads the game's last stored
//! position back, scores it by area, and reports what the board says next to
//! what the rule decided.
//!
//! Two caveats on reading the output, both of which understate the rule rather
//! than flatter it:
//!
//!   * The last stored position is the one the mover conceded *at*, not a
//!     played-out result. A game that is behind at ply 22 may still be winnable,
//!     so "board agrees" is necessary for a concession to be right, not
//!     sufficient.
//!   * Area score at ply 22 counts territory that is not yet settled. On a board
//!     this open it is a snapshot, not a final result.
//!
//!     review_resignations <shard-directory> [limit]

use std::{collections::BTreeMap, fs, path::PathBuf};

use vgo_core::{Analysis, Color, Position, Stone};

const HEADER: usize = 32;
const STONE: usize = 8 + 8 + 1;
const STONE_CAPACITY: usize = 128;
const POLICY_CAPACITY: usize = 64;

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn read_f64(bytes: &[u8], at: usize) -> f64 {
    f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Every stored position of every game, in ply order.
///
/// A shard stores positions, not moves. The move played at each ply is
/// recoverable by diffing consecutive positions: the stone that appears is the
/// placement, and a position whose stones are unchanged was a pass -- which is
/// also how a no-op self-capture now reads, since it leaves the board alone.
fn game_positions(shard: &PathBuf) -> BTreeMap<u64, Vec<(u32, Position)>> {
    let blob = fs::read(shard.join("dataset.vgo")).expect("read shard");
    let version = read_u32(&blob, 8);
    assert!(version == 5, "expected replay version 5, found {version}");
    let samples = read_u32(&blob, 12) as usize;

    let stones_at = 8 + 8 + 1 + 4 + 1 + 4;
    let count_at = 8 + 8 + 1 + 4 + 1;
    let stride = (blob.len() - HEADER) / samples;

    // Trailing scalars, in write order: value, selected_action, game, ply, seed.
    let policy_end = stones_at + STONE_CAPACITY * STONE + 4 + POLICY_CAPACITY * 20;
    let game_at = policy_end + 4 + 4;
    let ply_at = game_at + 8;

    let mut games: BTreeMap<u64, Vec<(u32, Position)>> = BTreeMap::new();
    for index in 0..samples {
        let base = HEADER + index * stride;
        let game = u64::from_le_bytes(blob[base + game_at..base + game_at + 8].try_into().unwrap());
        let ply = read_u32(&blob, base + ply_at);
        let radius = read_f64(&blob, base);
        let komi = read_f64(&blob, base + 8);
        let to_move = if blob[base + 16] == 0 { Color::Black } else { Color::White };
        let count = read_u32(&blob, base + count_at) as usize;
        let mut stones = Vec::with_capacity(count);
        for slot in 0..count {
            let at = base + stones_at + slot * STONE;
            stones.push(Stone::new(
                read_f64(&blob, at),
                read_f64(&blob, at + 8),
                if blob[at + 16] == 0 { Color::Black } else { Color::White },
            ));
        }
        games
            .entry(game)
            .or_default()
            .push((ply, Position::new(radius, stones, to_move).with_komi(komi)));
    }
    for plies in games.values_mut() {
        plies.sort_by_key(|(ply, _)| *ply);
    }
    games
}

/// An SGF for one game, with the moves recovered by diffing positions.
///
/// Diffing n stored positions yields only n-1 moves: the move played *from* the
/// last stored position has no successor to diff against, and a shard stores no
/// final board to supply it. The closing move is therefore genuinely absent.
///
/// That matters for reading parity. The concession happens at ply `plies - 1`,
/// which is one past the last move shown, so the side that appears to have
/// moved last is *not* the side that conceded -- it is its opponent. The
/// comment on the root node states the concession explicitly so the file cannot
/// be misread from move parity alone.
fn game_sgf(
    plies: &[(u32, Position)],
    komi: f64,
    radius: f64,
    result: &str,
    note: &str,
) -> String {
    let mut text = format!("(;FF[4]GM[VGO]SZ[1]RA[{radius}]KM[{komi}]PL[B]C[{note}]");
    for window in plies.windows(2) {
        let (before, after) = (&window[0].1, &window[1].1);
        // The mover is the side to play in the earlier position.
        let tag = match before.to_move() {
            Color::Black => "B",
            Color::White => "W",
        };
        let placed = after
            .stones()
            .iter()
            .find(|stone| {
                !before
                    .stones()
                    .iter()
                    .any(|prior| prior.x == stone.x && prior.y == stone.y)
            })
            .copied();
        match placed {
            Some(stone) => text.push_str(&format!(";{tag}[{},{}]", stone.x, stone.y)),
            None => text.push_str(&format!(";{tag}[]")),
        }
    }
    text.push_str(&format!("RE[{result}])"));
    text
}

fn main() {
    let mut args = std::env::args().skip(1);
    let shard = PathBuf::from(args.next().expect("shard directory"));
    let limit: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(12);

    let sidecar = fs::read_to_string(shard.join("games.jsonl")).expect("read games.jsonl");
    let positions = game_positions(&shard);
    // Outside the shard: the loop owns that directory and may retire it.
    let sgf_directory = PathBuf::from("diagnostics/resignation-review");
    fs::create_dir_all(&sgf_directory).expect("create review directory");

    let mut agree = 0_usize;
    let mut disagree = 0_usize;
    let mut shown = 0_usize;
    let mut margins: Vec<f64> = Vec::new();

    println!("{:>6} {:>5} {:>7} {:>7} {:>8} {:>8}  {}", "game", "plies", "komi", "winner", "board", "margin", "verdict");
    for line in sidecar.lines() {
        let field = |name: &str| -> Option<String> {
            let key = format!("\"{name}\":");
            let start = line.find(&key)? + key.len();
            let rest = &line[start..];
            let end = rest.find([',', '}'])?;
            Some(rest[..end].trim().to_owned())
        };
        if field("resigned").as_deref() != Some("true") {
            continue;
        }
        let game: u64 = field("game").unwrap().parse().unwrap();
        let plies: u32 = field("plies").unwrap().parse().unwrap();
        let komi: f64 = field("komi").unwrap().parse().unwrap();
        let utility: f64 = field("black_utility").unwrap().parse().unwrap();
        let Some(plies_stored) = positions.get(&game) else { continue };
        let Some((_, position)) = plies_stored.last() else { continue };

        let analysis = Analysis::new(position);
        let delta = analysis.score.black - analysis.score.white - position.komi();
        let recorded = if utility > 0.0 { "Black" } else { "White" };
        let board = if delta > 0.0 { "Black" } else { "White" };
        let matches = (delta > 0.0) == (utility > 0.0);
        if matches { agree += 1 } else { disagree += 1 }
        margins.push(delta.abs());

        if shown < limit {
            shown += 1;
            // Name the file with the verdict so the interesting ones are
            // findable without opening each: `ahead` means the side that
            // resigned was the one leading on the board.
            let verdict = if matches { "behind" } else { "ahead" };
            // The mover concedes, so the conceder is the side the recorded
            // winner is *not*.
            let conceder = if utility > 0.0 { "White" } else { "Black" };
            // The winner came from the resignation; the margin came from
            // scoring the board. Fusing them into `W+0.121` would assert that
            // White led by 0.121 when the board says the *conceder* did.
            // `RE[]` carries the recorded result, and the board's own verdict
            // goes in a comment beside it.
            let text = game_sgf(
                plies_stored,
                komi,
                position.radius(),
                if utility > 0.0 { "B+R" } else { "W+R" },
                &format!(
                    "{conceder} resigned at ply {}; {} was ahead by {:.3} \
                     (area, komi applied). NOTE: the closing move is not stored, \
                     so the move at ply {} by {} is missing from this file and \
                     the last move shown is two plies before the concession.",
                    plies - 1,
                    if delta > 0.0 { "Black" } else { "White" },
                    delta.abs(),
                    plies - 2,
                    if conceder == "Black" { "White" } else { "Black" },
                ),
            );
            fs::write(
                sgf_directory.join(format!(
                    "game{game:04}-ply{plies}-conceder-{verdict}-{:.3}.sgf",
                    delta.abs()
                )),
                text,
            )
            .expect("write sgf");
            println!(
                "{game:>6} {plies:>5} {komi:>+7.3} {recorded:>7} {board:>8} {:>8.3}  {}",
                delta.abs(),
                if matches { "board agrees" } else { "BOARD DISAGREES" }
            );
        }
    }

    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total = agree + disagree;
    println!("\nwrote {shown} sgfs to {}", sgf_directory.display());
    println!("resigned games scored: {total}");
    println!(
        "  board agrees with the concession: {agree} ({:.0}%)",
        100.0 * agree as f64 / total.max(1) as f64
    );
    println!(
        "  board disagrees:                  {disagree} ({:.0}%)",
        100.0 * disagree as f64 / total.max(1) as f64
    );
    if !margins.is_empty() {
        println!(
            "  area margin at concession: p25 {:.3}  median {:.3}  p75 {:.3}",
            margins[margins.len() / 4],
            margins[margins.len() / 2],
            margins[3 * margins.len() / 4],
        );
    }
}
