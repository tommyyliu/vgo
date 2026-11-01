# Dashboards

    python3 -m dashboard        # http://127.0.0.1:8766

Run it from the repository root. It needs only Python.

| Tab | Shows |
|---|---|
| Performance | speed of each path, timing history, generation and training throughput |
| Research | signal per compute, model strength, arena ledger, lemma status, ideas |

Metrics are read from `artifacts/`, `performance/history.jsonl` and `spec/LEMMAS.md`, and
refresh every 5 minutes. The loop panels use `artifacts/vgo-bulk` when it exists, and the
archived run otherwise. The work list, both experiments and engineering, is `ideas.json`,
which the Research tab edits.
