//! Append-only generator reports. This module is owned by the continuous binary.

use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use vgo_inference::{BatchedEvaluatorPool, BrokerMetrics};
use vgo_selfplay::telemetry::{GenerationTelemetry, Snapshot, broker_json};

use super::Config;

pub struct Reporter {
    stop: mpsc::Sender<()>,
    join: JoinHandle<io::Result<()>>,
}

fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn git_output(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn memory() -> Value {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return Value::Null;
    };
    let kb = |name: &str| {
        status.lines().find_map(|line| {
            let value = line.strip_prefix(name)?.split_whitespace().next()?;
            value.parse::<u64>().ok()
        })
    };
    json!({ "rss_kib": kb("VmRSS:"), "peak_rss_kib": kb("VmHWM:") })
}

fn configuration(config: &Config) -> Value {
    json!({
        "output_root": config.output_root, "label": config.label,
        "stop_file": config.stop_file, "maximum_games": config.maximum_games,
        "blocking_sync": config.blocking_sync, "profile_output": config.profile_output,
        "profile_seconds": config.profile_seconds, "run_seconds": config.run_seconds,
        "first_game": config.first_game,
        "telemetry_seconds": config.telemetry_seconds,
        "resolution": config.resolution, "policy_resolution": config.policy_resolution,
        "raster_kind": config.raster_kind.as_str(),
        "simulations": config.simulations, "coarse_pool": config.coarse_pool,
        "full_search_fraction": config.full_search_fraction, "fast_simulations": config.fast_simulations,
        "leaf_batch": config.leaf_batch, "maximum_candidates": config.maximum_candidates,
        "widening_coefficient": config.widening_coefficient,
        "root_exploration_noise": config.root_exploration_noise,
        "temperature": config.temperature, "temperature_plies": config.temperature_plies,
        "maximum_plies": config.maximum_plies, "radius": config.radius,
        "board_mix": config.board_mix, "komi_area_coefficient": config.komi_area_coefficient,
        "komi_low": config.komi_low, "komi_high": config.komi_high,
        "ruleset": format!("{:?}", config.ruleset),
        "resign_threshold": config.resign_threshold, "resign_window": config.resign_window,
        "resign_minimum_ply": config.resign_minimum_ply,
        "resign_soft_simulations": config.resign_soft_simulations,
        "resign_disable_fraction": config.resign_disable_fraction,
        "seed": config.seed, "actors": config.actors, "model": config.model,
        "maximum_batch": config.maximum_batch, "delay_ms": config.delay_ms,
        "inference_slots": config.inference_slots, "provider": format!("{:?}", config.provider),
        "device_id": config.device_id, "fp16": config.fp16,
        "cache_directory": config.cache_directory,
    })
}

fn write_record(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writeln!(writer)?;
    writer.flush()
}

impl Reporter {
    pub fn start(
        config: &Config,
        root: &Path,
        model_sha256: Option<&str>,
        telemetry: Arc<GenerationTelemetry>,
        broker: Option<BatchedEvaluatorPool>,
        cancelled: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(root.join("performance.jsonl"))?,
        );
        let started = Instant::now();
        let pid = std::process::id();
        let run_id = format!("{pid}-{:.6}", unix_seconds());
        let cpu = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name")?
                    .split_once(':')
                    .map(|(_, value)| value.trim().to_owned())
            })
        });
        let contract = broker.as_ref().map(|pool| pool.lane_contracts()[0]);
        let gpu = config.model.as_ref().and_then(|_| {
            let output = Command::new("nvidia-smi")
                .arg(format!("--id={}", config.device_id.max(0)))
                .args([
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ])
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        });
        write_record(
            &mut writer,
            &json!({
                "schema": "vgo.performance.v1", "kind": "run", "run_id": run_id,
                "pid": pid, "unix_seconds": unix_seconds(),
                "model_sha256": model_sha256, "configuration": configuration(config),
                "arguments": std::env::args_os().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>(),
                "commit": git_output(&["rev-parse", "HEAD"]),
                "dirty": git_output(&["status", "--porcelain"]).map(|text| !text.is_empty()),
                "hostname": fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|text| text.trim().to_owned()),
                "cpu": cpu, "logical_cpus": thread::available_parallelism().ok().map(usize::from),
                "gpu_description": gpu,
                "evaluator": if contract.is_some() { "neural" } else { "naive" },
                "input_layout": contract.map(|contract| format!("{:?}", contract.input_layout)),
                "effective_batch_capacity": contract.map(|contract| contract.maximum_batch),
                "stage_definition": "thirds of each game's maximum plies",
                "duration_semantics": "summed search/inference wall times overlap; queue durations are position-weighted; none are CPU time",
            }),
        )?;
        let seconds = config.telemetry_seconds;
        let label = config.label.clone();
        let (stop, receiver) = mpsc::channel();
        let join = thread::Builder::new().name("vgo-progress".into()).spawn(move || {
            let result = (|| -> io::Result<()> {
                let mut previous = Snapshot::default();
                let mut previous_broker = BrokerMetrics::default();
                let mut previous_time = started;
                loop {
                    let final_report = if seconds == 0 {
                        let _ = receiver.recv();
                        true
                    } else {
                        match receiver.recv_timeout(Duration::from_secs(seconds)) {
                            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => true,
                            Err(mpsc::RecvTimeoutError::Timeout) => false,
                        }
                    };
                    let now = Instant::now();
                    let elapsed = now.duration_since(started).as_secs_f64();
                    let interval_seconds = now.duration_since(previous_time).as_secs_f64();
                    let snapshot = telemetry.snapshot();
                    let mut report = snapshot.interval_json(&previous, interval_seconds);
                    report["schema"] = json!("vgo.performance.v1");
                    report["kind"] = json!(if final_report { "final" } else { "interval" });
                    report["run_id"] = json!(run_id);
                    report["pid"] = json!(pid);
                    report["label"] = json!(label);
                    report["unix_seconds"] = json!(unix_seconds());
                    report["elapsed_seconds"] = json!(elapsed);
                    report["interval_seconds"] = json!(interval_seconds);
                    report["totals"] = snapshot.totals.json(elapsed);
                    report["memory"] = memory();
                    report["cancelled"] = json!(cancelled.load(Ordering::Acquire));
                    if let Some(pool) = &broker {
                        let metrics = pool.metrics();
                        let capacity = pool.lane_contracts()[0].maximum_batch;
                        report["broker_interval"] = broker_json(metrics.delta_since(previous_broker), capacity);
                        report["broker_totals"] = broker_json(metrics, capacity);
                        previous_broker = metrics;
                    }
                    write_record(&mut writer, &report)?;
                    let interval = &report["interval"];
                    eprintln!(
                        "[progress] {label} {:.1}s {:.2} searched plies/s {:.1} games/h {:.0} samples/h batch {:.1} rss {:.0} MiB{}",
                        elapsed,
                        interval["searched_plies_per_second"].as_f64().unwrap_or(0.0),
                        interval["published_games_per_hour"].as_f64().unwrap_or(0.0),
                        interval["published_samples_per_hour"].as_f64().unwrap_or(0.0),
                        report["broker_interval"]["average_batch_size"].as_f64().unwrap_or(0.0),
                        report["memory"]["rss_kib"].as_u64().unwrap_or(0) as f64 / 1024.0,
                        if final_report { " final" } else { "" },
                    );
                    previous = snapshot;
                    previous_time = now;
                    if final_report { return Ok(()) }
                }
            })();
            if result.is_err() { cancelled.store(true, Ordering::Release) }
            result
        })?;
        Ok(Self { stop, join })
    }

    pub fn finish(self) -> io::Result<()> {
        let _ = self.stop.send(());
        self.join
            .join()
            .map_err(|_| io::Error::other("progress reporter panicked"))?
    }
}
