#!/usr/bin/env bash
# The one command that must pass before merging. CPU only.
#
#   scripts/check.sh           everything
#   scripts/check.sh --quick   skip the golden digests
#
# A golden mismatch means engine output changed. If that was intended, explain it
# in the commit and refresh with `scripts/check.sh --update-golden`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mode="${1:-}"

python3 scripts/check-spec.py
cargo clippy --release --workspace --exclude vgo-wasm --quiet
cargo test --release -p vgo-core -p vgo-raster -p vgo-search -p vgo-selfplay --quiet
cargo check --release -p vgo-wasm --target wasm32-unknown-unknown --quiet
(cd training && PYTHONPATH=. uv run --with pytest python -m pytest tests --quiet -p no:cacheprovider)

[[ "$mode" == --quick ]] && exit 0

cargo build --release -p vgo-selfplay --bin vgo-bench --quiet
digests() {
  target/release/vgo-bench --digest --positions benchmarks/positions.txt
  echo "fuzz-200 $(target/release/vgo-bench --fuzz-games 200 | sha256sum | cut -d' ' -f1)"
}
if [[ "$mode" == --update-golden ]]; then
  digests > benchmarks/golden.txt
else
  diff benchmarks/golden.txt <(digests)
fi
echo "check passed"
