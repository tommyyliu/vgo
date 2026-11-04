# Dashboards

    python3 -m dashboard        # http://127.0.0.1:8766

Run it from the repository root. It needs only Python.

| Tab | Shows |
|---|---|
| Work | items for review, current tasks in priority order, the backlog by why it waits, ideas to explore |
| Performance | speed of each path, timing history, generation and training throughput |
| Research | signal per compute, model strength, arena ledger, lemma status |
| Net shapes | where the time goes, what each timed and trained net shape changes, its speed and its validation |
| Reports | written plans and ideas, from `dashboard/reports/` |

Metrics are read from `artifacts/`, `performance/`, and `spec/LEMMAS.md`, and
refresh every 5 minutes. The loop panels use `artifacts/vgo-bulk` when it exists, and the
archived run otherwise. The work list is `ideas.json`, which the Work tab edits. Each item has a `kind`
(`task` or `idea`) and a `status` (Review, Running, Next, Later, Parked, Done, Rejected); list order
is priority. Backlog items carry `held` (why they wait) and `revisit` (when to bring them back),
current ones `progress`, and review items `ask`.
