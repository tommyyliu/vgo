#![forbid(unsafe_code)]

//! Serve model moves over HTTP so the JS client can play against a checkpoint.
//!
//! The arena and the Elo pool answer "is this model stronger than that one",
//! which is what training needs but says nothing about whether the play looks
//! sensible to a person. This binary closes that gap: it wraps the same MCTS and
//! ONNX evaluator the arena uses, so the opponent in the browser is exactly the
//! model being rated -- not a reimplementation that could drift.
//!
//! The protocol is one route. POST /move with the position:
//!
//! ```json
//! {"radius": 0.0556, "toMove": "B", "komi": 0.18,
//!  "stones": [{"x": 0.5, "y": 0.5, "c": "B"}]}
//! ```
//!
//! `komi` is optional and defaults to zero. It is a fraction of the board, not
//! a stone count, and the model reads it as a raster channel -- so a caller
//! that scores at one komi and searches at another gets a confident move for a
//! different game.
//!
//! and the reply is the chosen move plus what the search thought of it:
//!
//! ```json
//! {"pass": false, "x": 0.42, "y": 0.31, "visits": 121, "value": 0.18,
//!  "candidates": [...]}
//! ```
//!
//! Hand-rolled over `std::net::TcpListener` rather than pulling an async web
//! stack into a workspace whose only shared dependency is clap. The surface is
//! one route with a small body, and this is a local development tool.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};

use clap::{ArgAction, Parser};
use vgo_core::{Color, Phase, Position, Stone};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::{Action, SearchConfig, search_with_evaluator};

#[derive(Debug, Parser)]
#[command(about = "Serve model moves over HTTP for the JS client")]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8181")]
    address: String,
    /// Simulations per move. Higher is stronger and slower.
    #[arg(long, default_value_t = 256)]
    simulations: u32,
    #[arg(long, default_value_t = 96)]
    resolution: usize,
    #[arg(long, default_value_t = 32)]
    policy_resolution: usize,
    /// Channel layout the model reads, fixed at its export. A model trained on
    /// `compact` fed the twelve semantic channels sees a different input than
    /// it learned from, so this must match the export or the net plays blind.
    #[arg(long, default_value = "semantic")]
    raster_kind: RasterKind,
    /// Fine cells per coarse sampling region, when drawing candidates from the
    /// policy map. Zero falls back to the quasi-random candidate sequence and
    /// stops the policy head guiding the search at all.
    ///
    /// Sixteen because that is what every recipe in `runs/` passes, for
    /// generation and for arenas alike, so it is how these models are searched
    /// everywhere their strength has been measured. The old default of 4 was
    /// not: no run has used it, so serving a model here meant searching it
    /// through a sampler nothing else uses.
    #[arg(long, default_value_t = 16)]
    coarse_pool: usize,
    #[arg(long, default_value_t = 4)]
    leaf_batch: usize,
    /// Coefficient on progressive widening: a node wants
    /// `coefficient * visits^0.5` candidates.
    ///
    /// 4.0 rather than `SearchConfig`'s 2.0, which is measurably far too narrow.
    /// Same model both seats, only this differing, 120 games per arm:
    ///
    ///     800 sims   coef 4.0 (114 cands) vs 2.0 (57)    +232 Elo
    ///    3200 sims   coef 4.0 (227 cands) vs 2.0 (114)   +183 Elo
    ///
    /// The whole gain arrives between 2.0 and 4.0 and then plateaus out to at
    /// least 32.0, so 4.0 is the cheap end of a wide basin rather than a peak
    /// to be hit precisely. Below it the fall is steep: 1.0 is -338 and 0.5 is
    /// -708.
    ///
    /// Serving does not write shards, so nothing here is bounded by the replay
    /// record's 128-cell capacity. Generation still runs at 2.0 for exactly
    /// that reason -- 227 draws touch ~152 cells and would be truncated on the
    /// way to disk.
    #[arg(long, default_value_t = 4.0)]
    widening_coefficient: f64,
    /// Ceiling on root candidates, which must clear
    /// `widening_coefficient * sqrt(simulations)` or the coefficient is inert.
    #[arg(long, default_value_t = 321)]
    maximum_candidates: usize,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
    #[arg(long, default_value_t = 900_001)]
    seed: u64,
}

/// The subset of JSON this protocol needs: objects of numbers, strings, and
/// arrays of objects. Written by hand because the workspace carries no serde and
/// the request shape is fixed and small.
#[derive(Debug)]
struct Request {
    radius: f64,
    komi: f64,
    to_move: Color,
    stones: Vec<Stone>,
}

fn parse_number(text: &str) -> Option<f64> {
    text.trim().trim_matches('"').parse::<f64>().ok()
}

fn parse_color(text: &str) -> Option<Color> {
    match text.trim().trim_matches('"') {
        "B" | "b" | "black" | "Black" => Some(Color::Black),
        "W" | "w" | "white" | "White" => Some(Color::White),
        _ => None,
    }
}

/// Pull `"key": value` out of a flat JSON fragment. Values are scalars only;
/// nested structures are handled separately by `parse_stones`.
fn field<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)? + needle.len();
    let rest = body[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']')
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn parse_stones(body: &str) -> Vec<Stone> {
    let Some(start) = body.find("\"stones\"") else {
        return Vec::new();
    };
    let Some(open) = body[start..].find('[') else {
        return Vec::new();
    };
    let Some(close) = body[start + open..].find(']') else {
        return Vec::new();
    };
    let array = &body[start + open + 1..start + open + close];
    let mut stones = Vec::new();
    for chunk in array.split('{').skip(1) {
        let object = chunk.split('}').next().unwrap_or("");
        let x = field(object, "x").and_then(parse_number);
        let y = field(object, "y").and_then(parse_number);
        // The client calls the colour `c`; accept `color` too.
        let color = field(object, "c")
            .or_else(|| field(object, "color"))
            .and_then(parse_color);
        if let (Some(x), Some(y), Some(color)) = (x, y, color) {
            stones.push(Stone::new(x, y, color));
        }
    }
    stones
}

fn parse_request(body: &str) -> Result<Request, String> {
    let radius = field(body, "radius")
        .and_then(parse_number)
        .ok_or("missing or malformed \"radius\"")?;
    if !radius.is_finite() || radius <= 0.0 || radius >= 0.5 {
        return Err("radius must be finite and between zero and one half".into());
    }
    // Optional, defaulting to zero: a client written before komi existed still
    // means komi zero, which is what those games were played at. A *malformed*
    // komi is rejected rather than defaulted, because silently searching at
    // zero when the caller asked for something else returns a confident move
    // for a game nobody is playing.
    let komi = match field(body, "komi") {
        None => 0.0,
        Some(text) => parse_number(text).ok_or("malformed \"komi\"")?,
    };
    if !komi.is_finite() {
        return Err("komi must be finite".into());
    }
    let to_move = field(body, "toMove")
        .or_else(|| field(body, "to_move"))
        .and_then(parse_color)
        .ok_or("missing or malformed \"toMove\"")?;
    Ok(Request {
        radius,
        komi,
        to_move,
        stones: parse_stones(body),
    })
}

/// Does this request line address the move route? Accepts an optional trailing
/// slash and any query string, because a mistyped URL would otherwise 404 and
/// the browser would report that as a CORS error rather than a wrong path.
fn is_move_request(request_line: &str) -> bool {
    let Some(target) = request_line
        .strip_prefix("POST ")
        .and_then(|rest| rest.split_whitespace().next())
    else {
        return false;
    };
    let path = target.split(['?', '#']).next().unwrap_or(target);
    path == "/move" || path == "/move/"
}

fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    // Permissive CORS so the client works when opened straight off the
    // filesystem, which is how the reference client is normally used. A
    // `file://` page sends `Origin: null`, which only `*` satisfies.
    //
    // `Allow-Headers` lists what a preflight may approve, so it has to name
    // every header the client might set -- a request the browser rejects here
    // surfaces only as an opaque "CORS error" with no server-side trace.
    // `Max-Age` lets the browser cache the approval instead of preflighting
    // every move.
    write!(
        stream,
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type, Accept, Origin, X-Requested-With\r\n\
         Access-Control-Allow-Methods: POST, OPTIONS\r\n\
         Access-Control-Max-Age: 86400\r\n\
         Vary: Origin\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

fn move_json(result: &vgo_search::SearchResult, to_move: Color) -> String {
    let mut children: Vec<_> = result.children.iter().collect();
    children.sort_by_key(|child| std::cmp::Reverse(child.visits));
    // black_value is from black's perspective throughout the search; report it
    // from the mover's, which is what a UI wants to show.
    let orient = |value: f64| match to_move {
        Color::Black => value,
        Color::White => -value,
    };
    let chosen = children
        .iter()
        .find(|child| child.action == result.action)
        .map(|child| (child.visits, orient(child.black_value)))
        .unwrap_or((0, 0.0));
    // Every candidate, not a top-slice. A caller drawing what the search
    // considered needs the whole set: the interesting part is which legal
    // ground got no candidate at all, and a truncated list makes explored
    // regions look unexplored. Around ninety entries in the opening, a few
    // kilobytes, on a local development route.
    let candidates = children
        .iter()
        .map(|child| match child.action {
            Action::Pass => format!(
                "{{\"pass\":true,\"visits\":{},\"value\":{:.4},\"prior\":{:.5}}}",
                child.visits,
                orient(child.black_value),
                child.prior
            ),
            Action::Place(point) => format!(
                "{{\"pass\":false,\"x\":{:.6},\"y\":{:.6},\"visits\":{},\"value\":{:.4},\"prior\":{:.5},\"proposals\":{}}}",
                point.x,
                point.y,
                child.visits,
                orient(child.black_value),
                child.prior,
                child.proposal_count
            ),
        })
        .collect::<Vec<_>>()
        .join(",");
    match result.action {
        Action::Pass => format!(
            "{{\"pass\":true,\"visits\":{},\"value\":{:.4},\"candidates\":[{candidates}]}}",
            chosen.0, chosen.1
        ),
        Action::Place(point) => format!(
            "{{\"pass\":false,\"x\":{:.6},\"y\":{:.6},\"visits\":{},\"value\":{:.4},\"candidates\":[{candidates}]}}",
            point.x, point.y, chosen.0, chosen.1
        ),
    }
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<(String, String)> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut length = 0_usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0_u8; length];
    if length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok((request_line, String::from_utf8_lossy(&body).into_owned()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.simulations == 0 || arguments.resolution == 0 {
        return Err("simulations and resolution must be positive".into());
    }
    if arguments.coarse_pool > arguments.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution".into());
    }

    let service = OnnxBatchService::load(&OnnxServiceConfig {
        policy: Some(RasterConfig::square(arguments.policy_resolution)),
        model: arguments.model.clone(),
        raster: RasterConfig::square_of(arguments.resolution, arguments.raster_kind),
        maximum_batch: arguments.leaf_batch.max(1),
        provider: arguments.provider,
        device_id: 0,
        fp16: arguments.fp16,
        cache_directory: arguments.cache_directory.clone(),
    })?;
    let evaluator = BatchedEvaluator::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(1),
            queue_capacity: arguments.leaf_batch.max(2) * 2,
        },
        service,
    )?;

    let mut config = SearchConfig::canary(arguments.simulations);
    config.coarse_pool = arguments.coarse_pool;
    config.leaf_batch = arguments.leaf_batch.max(1);
    config.widening_coefficient = arguments.widening_coefficient;
    // The cap has to clear what the coefficient asks for. `canary` caps at 96,
    // which 4.0 reaches by 576 simulations -- so at the default 1600, and at
    // every count a human would actually play against, the coefficient was
    // doing nothing and this server was quietly serving the narrow arm whose
    // cost the flag above documents as -232 Elo. 321 is what the arenas that
    // measured the gain use, and it clears 4.0 out to 6400 simulations.
    config.maximum_candidates = arguments.maximum_candidates;
    // A human opponent wants the search's best move, not a draw from it.
    config.temperature = 0.0;
    config.temperature_plies = 0;

    let listener = TcpListener::bind(&arguments.address)?;
    println!(
        "vgo-serve-move listening on http://{} ({} simulations, model {})",
        arguments.address,
        arguments.simulations,
        arguments.model.display()
    );
    println!("POST /move with {{radius, toMove, stones:[{{x,y,c}}], komi?}}");

    let mut request_index: u64 = 0;
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("connection failed: {error}");
                continue;
            }
        };
        let (request_line, body) = match read_request(&mut stream) {
            Ok(parts) => parts,
            Err(error) => {
                eprintln!("could not read request: {error}");
                continue;
            }
        };
        if request_line.starts_with("OPTIONS") {
            let _ = respond(&mut stream, "204 No Content", "");
            continue;
        }
        if !is_move_request(&request_line) {
            // A 404 reaches the browser as an opaque CORS failure rather than a
            // status code, so be liberal about the path: `/move`, `/move/`, and
            // `/move?x=1` all mean the same thing here.
            let _ = respond(
                &mut stream,
                "404 Not Found",
                "{\"error\":\"POST /move is the only route\"}",
            );
            continue;
        }
        let request = match parse_request(&body) {
            Ok(request) => request,
            Err(error) => {
                let _ = respond(
                    &mut stream,
                    "400 Bad Request",
                    &format!("{{\"error\":\"{}\"}}", escape(&error)),
                );
                continue;
            }
        };
        let position =
            Position::new(request.radius, request.stones, request.to_move).with_komi(request.komi);
        // A client can post a finished or self-inconsistent board; refuse both
        // here rather than let the search fail deep inside on it.
        if position.phase() != Phase::Playing {
            let _ = respond(
                &mut stream,
                "409 Conflict",
                "{\"error\":\"position is already finished\"}",
            );
            continue;
        }
        if !position.validate().is_playable() {
            let _ = respond(
                &mut stream,
                "400 Bad Request",
                "{\"error\":\"position is not playable (overlapping or out-of-bounds stones)\"}",
            );
            continue;
        }
        // Vary the seed per request so repeated identical positions do not
        // replay one search, while a single game stays reproducible per seed.
        request_index += 1;
        let result = match search_with_evaluator(
            &position,
            config,
            arguments.seed.wrapping_add(request_index),
            &evaluator,
        ) {
            Ok(result) => result,
            Err(error) => {
                let _ = respond(
                    &mut stream,
                    "500 Internal Server Error",
                    &format!("{{\"error\":\"{}\"}}", escape(&error.to_string())),
                );
                continue;
            }
        };
        let payload = move_json(&result, position.to_move());
        println!(
            "move {} for {:?}: {}",
            request_index,
            position.to_move(),
            payload.split(",\"candidates\"").next().unwrap_or(&payload)
        );
        let _ = respond(&mut stream, "200 OK", &payload);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{field, parse_request, parse_stones};
    use vgo_core::Color;

    const BODY: &str = r#"{"radius":0.0556,"toMove":"W","stones":[
        {"x":0.25,"y":0.5,"c":"B"},{"x":0.75,"y":0.5,"c":"W"}]}"#;

    #[test]
    fn parses_a_client_position() {
        let request = parse_request(BODY).expect("body should parse");
        assert!((request.radius - 0.0556).abs() < 1e-9);
        assert_eq!(request.to_move, Color::White);
        assert_eq!(request.stones.len(), 2);
        assert_eq!(request.stones[0].color, Color::Black);
        assert!((request.stones[1].x - 0.75).abs() < 1e-9);
    }

    /// Komi reaches the model as a raster channel its value head is trained
    /// against, so a request that carries one and a search that ignores it are
    /// playing different games -- and the reply looks identical either way,
    /// which is what makes the omission worth a test.
    #[test]
    fn komi_is_read_when_present_and_zero_when_absent() {
        let with = parse_request(
            r#"{"radius":0.05,"komi":0.18,"toMove":"B","stones":[]}"#,
        )
        .expect("should parse");
        assert!((with.komi - 0.18).abs() < 1e-9);

        // A client written before komi existed means komi zero.
        assert_eq!(parse_request(BODY).expect("should parse").komi, 0.0);

        // Negative komi spots Black and is a legitimate setting, not a typo.
        let negative =
            parse_request(r#"{"radius":0.05,"komi":-0.125,"toMove":"B","stones":[]}"#)
                .expect("should parse");
        assert!((negative.komi + 0.125).abs() < 1e-9);
    }

    #[test]
    fn a_malformed_komi_is_rejected_rather_than_defaulted() {
        // Defaulting here would search at zero while the caller scores at
        // something else, and return a confident move for a game nobody plays.
        assert!(parse_request(r#"{"radius":0.05,"komi":"wat","toMove":"B","stones":[]}"#).is_err());
        assert!(parse_request(r#"{"radius":0.05,"komi":,"toMove":"B","stones":[]}"#).is_err());
    }

    #[test]
    fn an_empty_board_is_a_valid_position() {
        let request =
            parse_request(r#"{"radius":0.05,"toMove":"B","stones":[]}"#).expect("should parse");
        assert!(request.stones.is_empty());
    }

    #[test]
    fn malformed_bodies_are_rejected_rather_than_defaulted() {
        // A missing radius must not silently become zero and produce a position
        // the search would then reject deep inside.
        assert!(parse_request(r#"{"toMove":"B","stones":[]}"#).is_err());
        assert!(parse_request(r#"{"radius":0.05,"stones":[]}"#).is_err());
        assert!(parse_request(r#"{"radius":9.0,"toMove":"B"}"#).is_err());
    }

    #[test]
    fn stones_missing_a_field_are_skipped_not_guessed() {
        let stones = parse_stones(r#"{"stones":[{"x":0.5,"c":"B"},{"x":0.1,"y":0.2,"c":"W"}]}"#);
        assert_eq!(stones.len(), 1);
        assert_eq!(stones[0].color, Color::White);
    }

    #[test]
    fn the_move_route_tolerates_slashes_and_queries() {
        use super::is_move_request;
        assert!(is_move_request("POST /move HTTP/1.1"));
        assert!(is_move_request("POST /move/ HTTP/1.1"));
        assert!(is_move_request("POST /move?seed=7 HTTP/1.1"));
        assert!(!is_move_request("POST /moves HTTP/1.1"));
        assert!(!is_move_request("GET /move HTTP/1.1"));
        assert!(!is_move_request("POST / HTTP/1.1"));
    }

    #[test]
    fn field_stops_at_the_value_boundary() {
        assert_eq!(field(BODY, "radius"), Some("0.0556"));
        assert_eq!(field(BODY, "toMove"), Some("\"W\""));
    }
}
