//! Did the search see the refutation, or only the evaluation after it landed?
//!
//! A value that holds near +1.0 and then collapses in a single ply means one of
//! two things, and they call for opposite fixes:
//!
//!   * the reply was never proposed -- a candidate-generation failure, where no
//!     amount of search depth helps because the move is not in the tree at all;
//!   * the reply was proposed and misjudged -- a value-head failure, where the
//!     move was searched and its consequence still read as fine.
//!
//! This replays one position, reports whether the actual reply was among the
//! proposals, how it ranked, and what the search thought the position was worth
//! before and after it.
//!
//!     probe_capture <position.json> <reply-x> <reply-y> <model.onnx> [simulations]

use std::{fs, path::PathBuf};

use vgo_core::{Analysis, Color, Point, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind};
use std::time::Duration;

use vgo_search::{Action, SearchConfig, search_at_ply};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};

fn field(text: &str, key: &str) -> Option<String> {
    let anchor = format!("\"{key}\":");
    let start = text.find(&anchor)? + anchor.len();
    let rest = text[start..].trim_start();
    let end = rest.find([',', '}', ']'])?;
    Some(rest[..end].trim().trim_matches('"').to_owned())
}

fn parse_position(text: &str) -> Position {
    let radius: f64 = field(text, "radius").expect("radius").parse().expect("f64");
    let to_move = match field(text, "toMove").as_deref() {
        Some("white") => Color::White,
        _ => Color::Black,
    };
    let mut stones = Vec::new();
    // Stones are flat objects in one array; walking the braces avoids a JSON
    // dependency for a diagnostic that reads one hand-made file.
    let body = &text[text.find("\"stones\"").expect("stones")..];
    for chunk in body.split('{').skip(1) {
        let x: f64 = match field(chunk, "x") {
            Some(value) => value.parse().expect("x"),
            None => continue,
        };
        let y: f64 = field(chunk, "y").expect("y").parse().expect("y");
        let colour = match field(chunk, "c").as_deref() {
            Some("w") => Color::White,
            _ => Color::Black,
        };
        stones.push(Stone::new(x, y, colour));
    }
    Position::new(radius, stones, to_move)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let position_path = PathBuf::from(args.next().expect("position json"));
    let reply_x: f64 = args.next().expect("reply x").parse().expect("f64");
    let reply_y: f64 = args.next().expect("reply y").parse().expect("f64");
    let model = PathBuf::from(args.next().expect("model path"));
    let simulations: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(2048);

    let position = parse_position(&fs::read_to_string(&position_path).expect("read position"));
    println!(
        "position: {} stones, {:?} to move",
        position.stones().len(),
        position.to_move()
    );

    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: RasterConfig::square_of(128, RasterKind::Compact),
        policy: Some(RasterConfig::square(128)),
        maximum_batch: 64,
        provider: OnnxProvider::TensorRt,
        device_id: 0,
        fp16: true,
        cache_directory: PathBuf::from("artifacts/onnx-cache"),
    })
    .expect("load model");
    let evaluator = BatchedEvaluator::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(1),
            queue_capacity: 128,
        },
        service,
    )
    .expect("spawn evaluator");

    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = 16;
    config.leaf_batch = 4;
    let result = search_at_ply(&position, config, 0, &evaluator, 0).expect("search");

    // Where the actual reply sits among what the search proposed.
    let reply = Point::new(reply_x, reply_y);
    let mut ranked: Vec<_> = result
        .children
        .iter()
        .filter_map(|child| match child.action {
            Action::Place(point) => Some((point, child.visits, child.black_value)),
            Action::Pass => None,
        })
        .collect();
    ranked.sort_by_key(|(_, visits, _)| std::cmp::Reverse(*visits));

    let found = ranked
        .iter()
        .enumerate()
        .min_by(|(_, (a, _, _)), (_, (b, _, _))| {
            a.distance(reply)
                .partial_cmp(&b.distance(reply))
                .expect("finite")
        });

    println!("\nsearch proposed {} distinct placements", ranked.len());
    match found {
        Some((rank, (point, visits, value))) => {
            let distance = point.distance(reply);
            println!(
                "nearest proposal to the played reply: ({:.6}, {:.6})\n  \
                 distance {distance:.6}  rank {} of {}  visits {visits}  value {value:+.4}",
                point.x,
                point.y,
                rank + 1,
                ranked.len(),
            );
            // The policy grid is 128x128 over the unit square, so a proposal
            // within half a cell is the same move; anything further is a
            // different point that merely happens to be closest.
            let cell = 1.0 / 128.0;
            if distance <= cell {
                println!("  -> the reply WAS in the tree");
            } else {
                println!("  -> the reply was NOT proposed (nearest is {:.1} cells away)", distance / cell);
            }
        }
        None => println!("search proposed nothing"),
    }

    println!("\ntop proposals by visits:");
    for (index, (point, visits, value)) in ranked.iter().take(8).enumerate() {
        println!(
            "  {:>2}. ({:.4}, {:.4})  visits {visits:>5}  value {value:+.4}",
            index + 1,
            point.x,
            point.y
        );
    }

    // What the reply actually does to the board, independent of the search.
    let mover = position.to_move();
    let before = Analysis::new(&position);
    let after_reply = Action::Place(reply).apply(&position);
    let after = Analysis::new(&after_reply.position);
    let before_area = before.score.for_color(mover);
    let after_area = after.score.for_color(mover);
    println!(
        "\nboard effect of the reply on {mover:?}'s area: {before_area:.4} -> {after_area:.4} \
         ({:+.4})",
        after_area - before_area
    );
    println!(
        "stones on board: {} -> {}",
        position.stones().len() + 1,
        after_reply.position.stones().len()
    );
}
