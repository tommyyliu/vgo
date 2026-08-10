//! How much policy weight did the model put on a specific move?
//!
//! Replays a fixed move list (coordinates only, in play order, alternating
//! colors starting Black) through the real game rules -- so captures are
//! resolved exactly as they were during play -- stops just before a chosen
//! ply, and reports the trained policy's probability at that point for the
//! move actually played there plus the highest-probability moves overall.
//!
//! Usage:
//!   cargo run --release -p vgo-inference --example inspect_policy -- \
//!     --model <candidate.onnx> --radius <r> --before-ply <n> \
//!     --moves x1,y1;x2,y2;...

use std::{env, path::PathBuf};

use vgo_core::{Color, Position, Stone, place};
use vgo_inference::{
    BatchService, InferenceInput, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind, action_pixel, rasterize};

fn argument(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .iter()
        .position(|a| a == name)
        .and_then(|i| arguments.get(i + 1))
        .cloned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().collect();
    let model = PathBuf::from(argument(&arguments, "--model").expect("--model required"));
    let radius: f64 = argument(&arguments, "--radius")
        .expect("--radius required")
        .parse()?;
    let resolution: usize = argument(&arguments, "--resolution")
        .unwrap_or_else(|| "128".to_string())
        .parse()?;
    let policy_resolution: usize = argument(&arguments, "--policy-resolution")
        .unwrap_or_else(|| "128".to_string())
        .parse()?;
    let before_ply: usize = argument(&arguments, "--before-ply")
        .expect("--before-ply required (1-indexed ply to stop before)")
        .parse()?;
    let moves_raw = argument(&arguments, "--moves").expect("--moves required");
    let top_n: usize = argument(&arguments, "--top")
        .unwrap_or_else(|| "15".to_string())
        .parse()?;
    let cache = PathBuf::from(
        argument(&arguments, "--cache-directory")
            .unwrap_or_else(|| "artifacts/onnx-cache".to_string()),
    );

    let moves: Vec<(f64, f64)> = moves_raw
        .split(';')
        .map(|pair| {
            let mut parts = pair.split(',');
            let x: f64 = parts.next().unwrap().parse().unwrap();
            let y: f64 = parts.next().unwrap().parse().unwrap();
            (x, y)
        })
        .collect();

    let mut position = Position::new(radius, Vec::<Stone>::new(), Color::Black);
    for (index, &(x, y)) in moves.iter().enumerate() {
        if index + 1 == before_ply {
            break;
        }
        let result = place(&position, x, y)
            .unwrap_or_else(|error| panic!("illegal move #{} at ({x}, {y}): {error:?}", index + 1));
        for event in &result.events {
            println!("  ply {:>3}: {:?}", index + 1, event);
        }
        position = result.position;
    }

    let mover = position.to_move();
    println!(
        "\nposition before ply {before_ply}: {} stones, {:?} to move",
        position.stones().len(),
        mover
    );

    let raster_config = RasterConfig::square_of(resolution, RasterKind::Compact);
    let policy_config = RasterConfig::square(policy_resolution);

    let mut service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: raster_config,
        policy: Some(policy_config),
        maximum_batch: 8,
        provider: OnnxProvider::TensorRt,
        device_id: 0,
        fp16: true,
        cache_directory: cache,
    })?;

    let raster = rasterize(&position, raster_config);
    let outputs = service.infer(&[InferenceInput::new(0, raster)])?;
    let output = &outputs[0];
    let logits = output.policy();

    // Softmax over the full grid (+ pass, the last entry) for real probabilities.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from(l - max_logit).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    let probability = |index: usize| exp[index] / sum;

    println!(
        "value estimate for {:?}: {:.4}",
        mover,
        output.current_value()
    );

    if before_ply <= moves.len() {
        let (mx, my) = moves[before_ply - 1];
        let index = action_pixel(mx, my, policy_config);
        let mut rank = 0usize;
        let mut sorted: Vec<usize> = (0..logits.len()).collect();
        sorted.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        for (position_in_order, &candidate_index) in sorted.iter().enumerate() {
            if candidate_index == index {
                rank = position_in_order;
                break;
            }
        }
        println!("\nmove actually played at ply {before_ply}: ({mx}, {my})  cell {index}");
        println!(
            "  probability: {:.6}  ({:.4}%)  rank {} of {}",
            probability(index),
            probability(index) * 100.0,
            rank + 1,
            logits.len()
        );

        // Local neighborhood: every legal cell within ~1 raster-pixel of the
        // played point, so "was the general area considered" is answerable
        // even if the exact snapped cell is not the top choice.
        let (width, height) = (policy_config.width, policy_config.height);
        let cell_x = ((mx * width as f64).floor() as isize).clamp(0, width as isize - 1);
        let cell_y = ((my * height as f64).floor() as isize).clamp(0, height as isize - 1);
        println!("\n  neighborhood (3x3 cells around the played point):");
        let mut neighborhood_mass = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = cell_x + dx;
                let ny = cell_y + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let neighbor_index = ny as usize * width + nx as usize;
                neighborhood_mass += probability(neighbor_index);
                println!(
                    "    cell ({nx:>3},{ny:>3})  logit {:>8.3}  prob {:.6}",
                    logits[neighbor_index],
                    probability(neighbor_index)
                );
            }
        }
        println!(
            "  total probability mass in 3x3 neighborhood: {:.6} ({:.4}%)",
            neighborhood_mass,
            neighborhood_mass * 100.0
        );
    }

    println!("\ntop {top_n} moves by policy probability:");
    let mut sorted: Vec<usize> = (0..logits.len()).collect();
    sorted.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let (width, _height) = (policy_config.width, policy_config.height);
    for &index in sorted.iter().take(top_n) {
        if index == logits.len() - 1 {
            println!("  PASS  prob {:.6}", probability(index));
            continue;
        }
        let row = index / width;
        let col = index % width;
        let x = (col as f64 + 0.5) / width as f64;
        let y = (row as f64 + 0.5) / width as f64;
        println!(
            "  cell ({col:>3},{row:>3}) ~ ({x:.4},{y:.4})  logit {:>8.3}  prob {:.6}",
            logits[index],
            probability(index)
        );
    }

    Ok(())
}
