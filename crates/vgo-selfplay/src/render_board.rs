//! Render positions as SVG, matching the web client.
//!
//! Reads positions as JSON on stdin -- one object per line, or a single array --
//! and writes one SVG per position. The v4 replay reader lives in the training
//! package, so the split is deliberate: Python owns shard parsing and this owns
//! drawing, and neither reimplements the other.
//!
//!     {"radius": 0.0555, "stones": [{"x": 0.5, "y": 0.5, "color": "B"}, ...]}
//!
//! `diagnostics/render_game.py` produces exactly that from a shard.

use std::io::Read as _;
use std::path::PathBuf;
use std::process::ExitCode;

use vgo_core::{Color, Position, Stone};
use vgo_selfplay::render_svg::{RenderOptions, render};

fn parse_color(text: &str) -> Option<Color> {
    match text {
        "B" | "b" | "black" | "Black" => Some(Color::Black),
        "W" | "w" | "white" | "White" => Some(Color::White),
        _ => None,
    }
}

/// A minimal scanner for the object shape this tool documents.
///
/// Pulling in a JSON dependency for one fixed schema, in a binary that exists
/// to draw pictures, is not worth the build cost -- and the producer is a
/// script in this repository rather than an arbitrary caller.
fn parse_position(line: &str) -> Result<Position, String> {
    let radius = field(line, "\"radius\"")
        .ok_or("missing radius")?
        .parse::<f64>()
        .map_err(|error| format!("bad radius: {error}"))?;
    let mut stones = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("{\"x\"") {
        let chunk = &rest[start..];
        let end = chunk.find('}').ok_or("unterminated stone")?;
        let stone = &chunk[..=end];
        let x = field(stone, "\"x\"")
            .ok_or("stone missing x")?
            .parse::<f64>()
            .map_err(|error| format!("bad x: {error}"))?;
        let y = field(stone, "\"y\"")
            .ok_or("stone missing y")?
            .parse::<f64>()
            .map_err(|error| format!("bad y: {error}"))?;
        let color_text = quoted(stone, "\"color\"").ok_or("stone missing color")?;
        let color = parse_color(&color_text)
            .ok_or_else(|| format!("unknown colour {color_text:?}"))?;
        stones.push(Stone { x, y, color });
        rest = &chunk[end + 1..];
    }
    let to_move = quoted(line, "\"to_move\"")
        .and_then(|text| parse_color(&text))
        .unwrap_or(Color::Black);
    Ok(Position::new(radius, stones, to_move))
}

fn field(text: &str, key: &str) -> Option<String> {
    let start = text.find(key)? + key.len();
    let tail = text[start..].trim_start().strip_prefix(':')?.trim_start();
    let end = tail
        .find(|character: char| !matches!(character, '0'..='9' | '.' | '-' | 'e' | 'E' | '+'))
        .unwrap_or(tail.len());
    Some(tail[..end].to_owned())
}

fn quoted(text: &str, key: &str) -> Option<String> {
    let start = text.find(key)? + key.len();
    let tail = text[start..].trim_start().strip_prefix(':')?.trim_start();
    let tail = tail.strip_prefix('"')?;
    let end = tail.find('"')?;
    Some(tail[..end].to_owned())
}

fn main() -> ExitCode {
    let mut options = RenderOptions::default();
    let mut output = PathBuf::from(".");
    let mut prefix = "board".to_owned();

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--output" => output = PathBuf::from(arguments.next().unwrap_or_default()),
            "--prefix" => prefix = arguments.next().unwrap_or_default(),
            "--size" => {
                options.size = arguments
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(options.size);
            }
            "--settled" => options.settled = true,
            "--stone-ids" => options.stone_ids = true,
            "--no-legal-mask" => options.legal_mask = false,
            "--no-boundaries" => options.boundaries = false,
            "--no-regions" => options.regions = false,
            "--help" | "-h" => {
                eprintln!(
                    "usage: vgo-render-board [--output DIR] [--prefix NAME] [--size PX]\n\
                     \x20                      [--settled] [--stone-ids]\n\
                     \x20                      [--no-legal-mask] [--no-boundaries] [--no-regions]\n\
                     positions arrive as JSON on stdin, one per line"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mut input = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("reading stdin: {error}");
        return ExitCode::FAILURE;
    }
    if let Err(error) = std::fs::create_dir_all(&output) {
        eprintln!("creating {}: {error}", output.display());
        return ExitCode::FAILURE;
    }

    let mut written = 0_usize;
    for (index, line) in input.lines().filter(|line| line.contains("\"radius\"")).enumerate() {
        let position = match parse_position(line) {
            Ok(position) => position,
            Err(error) => {
                eprintln!("line {}: {error}", index + 1);
                return ExitCode::FAILURE;
            }
        };
        let svg = render(&position, options);
        let path = output.join(format!("{prefix}-{index:03}.svg"));
        if let Err(error) = std::fs::write(&path, svg) {
            eprintln!("writing {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
        written += 1;
    }
    println!("wrote {written} SVG(s) to {}", output.display());
    ExitCode::SUCCESS
}
