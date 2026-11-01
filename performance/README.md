# Benchmarks

| What | Where |
|---|---|
| Run, compare and record CPU timings | `performance/run.py` |
| Timing and verification code | [`crates/vgo-bench`](../crates/vgo-bench) |
| Fixed positions | `positions.txt` |
| Timing history, one record per run | `history.jsonl` |
| Expected output digests | `golden.txt`, checked by `correctness/check.sh` |
| Browser inference benchmark | [`web-bot/bench`](../web-bot/bench), `web-bot/public/bench.html` |

Timings are wall-clock, so only compare runs from the same machine. Fixture paths such as
`raster_*` revisit positions and hit caches, so compare `search_*` across versions.

Run `vgo-verify` to compare engine output between builds. See
[`check.sh`](../correctness/check.sh) for how the golden digests are refreshed.
