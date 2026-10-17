use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use vgo_search::EvaluationError;

pub(crate) fn arm_fatal_watchdog() {
    static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if ARMED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    eprintln!(
        "[inference] fatal native failure: exit now; forced termination in 3s if cleanup blocks"
    );
    thread::spawn(|| {
        thread::sleep(Duration::from_secs(3));
        #[cfg(target_os = "linux")]
        rustix::runtime::exit_group(70);
        #[cfg(not(target_os = "linux"))]
        std::process::abort();
    });
}

/// Bound initialization that may be blocked inside a native library. A timeout
/// is fatal to the owning process: dropping a JoinHandle detaches the thread;
/// process exit is what releases its native resources.
pub(crate) fn startup<T: Send + 'static>(
    description: &str,
    timeout: Duration,
    fatal_native_failure: bool,
    load: impl FnOnce() -> Result<T, EvaluationError> + Send + 'static,
) -> Result<T, EvaluationError> {
    if timeout.is_zero() {
        return Err(EvaluationError::new(
            "inference startup timeout must be positive",
        ));
    }
    let started = Instant::now();
    let (sender, receiver) = mpsc::channel();
    let worker = thread::Builder::new()
        .name("vgo-onnx-startup".into())
        .spawn(move || {
            let _ = sender.send(load());
        })
        .map_err(|error| {
            EvaluationError::new(format!("start ONNX loader ({description}): {error}"))
        })?;
    let result = loop {
        let remaining = timeout.saturating_sub(started.elapsed());
        match receiver.recv_timeout(remaining.min(Duration::from_secs(30))) {
            Ok(result) => break result,
            Err(mpsc::RecvTimeoutError::Timeout) if started.elapsed() < timeout => {
                eprintln!(
                    "[inference] still initializing {description}; elapsed {:.1}s / {:.1}s startup limit; native session/engine construction has not returned",
                    started.elapsed().as_secs_f64(),
                    timeout.as_secs_f64()
                );
            }
            Err(error) => {
                if fatal_native_failure {
                    arm_fatal_watchdog();
                }
                return Err(EvaluationError::new(format!(
                    "ONNX startup failed after {:.3}s ({description}): {error}; native load may still be running, exit this process",
                    started.elapsed().as_secs_f64()
                )));
            }
        }
    };
    let _ = worker.join();
    if fatal_native_failure && result.is_err() {
        arm_fatal_watchdog();
    }
    result.map_err(|error| EvaluationError::new(format!("ONNX startup ({description}): {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn fatal_watchdog_child() {
        if std::env::var_os("VGO_TEST_FATAL_EXIT").is_none() {
            return;
        }
        arm_fatal_watchdog();
        loop {
            thread::park();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fatal_watchdog_terminates_a_child_that_cannot_finish_cleanup() {
        use std::process::{Command, Stdio};
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "deadline::tests::fatal_watchdog_child"])
            .env("VGO_TEST_FATAL_EXIT", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert_eq!(status.code(), Some(70));
                break;
            }
            if started.elapsed() > Duration::from_secs(8) {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("fatal watchdog did not terminate its process");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn startup_deadline_returns_without_joining_a_blocked_loader() {
        let (release, gate) = mpsc::channel();
        let started = Instant::now();
        let result = startup(
            "model=test provider=test",
            Duration::from_millis(20),
            false,
            move || {
                gate.recv().unwrap();
                Ok(())
            },
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("model=test provider=test"));
        assert!(started.elapsed() < Duration::from_secs(1));
        release.send(()).unwrap();
    }

    #[test]
    fn startup_success_error_and_panic_all_reach_the_owner() {
        assert_eq!(
            startup("ok", Duration::from_secs(1), false, || Ok(42)).unwrap(),
            42
        );
        assert!(
            startup::<()>("bad model", Duration::from_secs(1), false, || Err(
                EvaluationError::new("invalid graph")
            ))
            .unwrap_err()
            .to_string()
            .contains("bad model): invalid graph")
        );
        assert!(
            startup::<()>("missing library", Duration::from_secs(1), false, || panic!(
                "native startup panic"
            ))
            .unwrap_err()
            .to_string()
            .contains("missing library")
        );
    }
}
