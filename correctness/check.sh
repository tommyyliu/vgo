#!/usr/bin/env bash
# The one command that must pass before merging. CPU only.
#
#   correctness/check.sh           everything
#   correctness/check.sh --quick   skip the golden digests
#
# A golden mismatch means engine output changed. If that was intended, explain it
# in the commit and refresh with `correctness/check.sh --update-golden`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mode="${1:-}"

python3 correctness/check-spec.py
cargo fmt --all --check
cargo clippy --release --workspace --exclude vgo-wasm --all-targets --quiet -- -D warnings
cargo test --release -p vgo-core -p vgo-raster -p vgo-search -p vgo-inference -p vgo-selfplay -p vgo-bench -p vgo-wasm --quiet
cargo check --release -p vgo-wasm --target wasm32-unknown-unknown --quiet
cargo check --release -p vgo-selfplay -p vgo-bench --features profiling --all-targets --quiet
(cd training && PYTHONPATH=. uv run --extra tensorrt --with pytest python -m pytest tests --quiet -p no:cacheprovider)

(cd training && PYTHONPATH=.. uv run --extra tensorrt --with pytest python -m pytest ../dashboard/tests --quiet -p no:cacheprovider)

[[ "$mode" == --quick ]] && exit 0

cargo build --release -p vgo-bench --bin vgo-verify --quiet
cargo build --release -p vgo-selfplay --bins --quiet
digests() {
  target/release/vgo-verify
  echo "fuzz-200 $(target/release/vgo-verify --fuzz-games 200 | sha256sum | cut -d' ' -f1)"
  correctness/game-digest.sh
}
if [[ "$mode" == --update-golden ]]; then
  digests > correctness/golden.txt
else
  diff correctness/golden.txt <(digests)
fi
echo "check passed"
