# Voronoi Go: project and idea outline

Current implementation reviewed on 2026-10-13. This is a conceptual map of the
system and the reasoning needed to understand it. Commands live in
[RUNNING.md](RUNNING.md); the next work lives in the
[restart roadmap](../todo/ROADMAP.md). Historical research notes may describe
drivers or options removed in the September prune.

## 1. The objective and the dependency chain

### 1.1 What the project is trying to do

Learn to play a continuous-placement territory game through self-play, then
make the learned opponent usable locally and in a browser. There is no human
move dataset supplying the main learning signal: search improves the current
model's decisions, games supply outcomes, and training distills those results
into the next model.

The useful end product is playing strength within a practical compute budget.
Lower training loss, more samples, and faster inference are intermediate
measurements. Each can improve while the actual player becomes weaker.

### 1.2 Which ideas depend on which

```text
precise game definition
  -> finite geometric tests for legality, settlement, and capture
    -> a reliable native simulator
      -> search over a sampled continuous action space
        -> policy targets that account for how actions were sampled
          -> self-play records with trustworthy outcomes
            -> a stable learner and measurable policy improvement
              -> efficient inference and generation
                -> sustained local training and a playable browser bot
```

There are feedback loops across this hierarchy. A better proposal policy makes
search more useful; search makes better policy targets. Faster geometry buys
more search or more independent games. Evaluation is what determines whether
either improvement actually helped.

### 1.3 Three representations of one game

1. **Rule state:** floating-point stone coordinates, colors, radius, side to
   move, pass count, phase, komi, and ruleset. This is what transactions consume.
2. **Derived geometry:** cells, shared edges, groups, legal placement features,
   survival witnesses, and areas. This is what computes consequences.
3. **Learning representation:** raster inputs, a spatial policy grid, and a
   value estimate. These help choose moves; they do not define the rules.

Keeping these roles distinct is essential. A pixel approximation cannot decide
a near-tangent capture, and a rendered board does not necessarily encode every
piece of rule state.

## 2. The game: continuous placement and changing territory

### 2.1 Board size is a geometric parameter

The board is always the unit square. Stone centers lie inside its radius-sized
inset, and centers must be at least two radii apart. Touching is legal. Smaller
radius means a larger effective board: radius 1/18 and 1/38 have capacities
roughly comparable to 9×9 and 19×19 Go.

This convention makes coordinates portable across rendering sizes. It also
means the learner must see the radius: an empty unit-square image alone cannot
tell it which effective board it is playing on.

### 2.2 Territory and connectivity come from Voronoi cells

Every board point belongs to its nearest stone. Intersecting a stone's distance
bisectors with the board produces its Voronoi cell. A color's score is the sum
of its cell areas, with komi subtracted from Black's lead.

Same-color stones form a group when their cells connect through shared edges
of positive length. Meeting at a single vertex is insufficient. Captures can
change both ownership and connectivity far from the newly placed stone.

### 2.3 Legal space determines contestability

The legal-placement set is the board inset minus the open exclusion disks of
radius `2r` around existing centers. A board point is contestable if some legal
new center could become strictly nearer than its existing owner.

A group is settled when none of its territory is contestable. Under this
repository's rules, settled opponent groups are removed during a placement,
followed by settled mover groups. Self-capture can affect a disconnected group,
not merely the stone just placed.

### 2.4 Transactions and ending

A placement has two simultaneous removal stages:

1. Insert the proposed stone and remove all settled opponent groups.
2. Reanalyze, then remove all settled mover groups and commit the result.

Reanalysis matters because removing an opponent can reopen legal space and
revive a friendly group. The mathematical results justify two stages without
an additional capture fixed-point loop.

Passing changes no stones; two passes end the game. A no-op placement also
counts as a pass. The Vgo rules permit repetition. The alternate `Official`
ruleset represents the website's different capture and self-capture behavior;
experiments must identify their ruleset explicitly.

Sources: [rules](../reference/RULES.md),
[derived properties](../reference/AXIOMS.md),
[official-rule differences](OFFICIAL_RULES.md).

## 3. The mathematical reduction that makes the engine possible

### 3.1 From infinitely many territory points to cell vertices

For an existing owner `s` and proposed center `p`, the difference
`||x-s||² - ||x-p||²` is affine in board point `x`. An affine function reaches
its maximum on a convex polygon at a vertex.

Therefore a placement takes positive area from a cell exactly when it is
strictly closer than the owner at some cell vertex. This turns an infinite
territory test into finitely many vertex queries. It is the central bridge from
the rules to computation.

### 3.2 From arbitrary legal centers to geometric features

At a cell vertex `v`, ask whether the distance to the legal-placement set is
less than the distance to the owner. Closest legal points can be found among
specific feature families: edge projections, exclusion-circle projections,
inset corners, circle/edge intersections, and circle/circle intersections.

Each candidate must satisfy the other placement constraints. Enumerating an
intersection does not automatically make it legal. Spatial buckets and early
rejection reduce the cost of these checks.

### 3.3 A survival witness is enough

One valid legal center challenging one cell vertex proves that the group is
not settled. The engine can stop searching for that group once it has such a
witness. A cached positive certificate remains useful only while its required
constraints remain valid.

Failure to find a cached witness is not proof of settlement: the engine must
fall back to the full test. This asymmetry is why cache eviction can safely
lose acceleration without changing captures.

### 3.4 Locality is narrower than whole-move locality

The research establishes an upper bound `(2 + 4√2)r`, approximately `7.65685r`,
for an individual stone's cell contestability under removal of farther stones,
with a constructed lower bound near `7.4918r`.

This does not bound group connectivity or an entire capture transaction.
Locality can help a geometric query without justifying a wholly local rules
engine. Confusing those statements would create an incorrect optimization.

### 3.5 Real-number theorems and floating-point implementations

Strict inequalities and exact tangencies matter. Rust's robust distance
comparison uses a fast floating-point error bound and an exact dyadic fallback
for ambiguous comparisons of represented coordinates. This does not make every
derived intersection exact real arithmetic: polygon construction and numeric
tolerances still require regression fixtures.

### 3.6 The settled overlay has its own geometric reduction

The settled set can be expressed as a union over stones of regions where
`||x-s|| <= dist(x,L)`. A stone that is not the nearest owner only makes the
inequality harder to satisfy, so this formulation needs no Voronoi partition.

Each per-stone region is star-shaped around that stone. Its boundary can be
described by one radius per direction, computed from the legal-feature families
in closed form. The reference renderer samples directions and fills the union
of these contours; overlapping constituent loops require a union fill rather
than treating each overlap as a hole.

This explains both the analytic reference overlay and the older radial raster
implementation. The faster distance-transform raster is another way to obtain
the indicator at model pixels. Capture uses the cell-vertex criterion, so none
of these drawing choices has rule authority.

Sources: [A3–A6](../reference/AXIOMS.md),
[numeric implementation](../crates/vgo-core/src/numeric.rs),
[locality upper bound](research/LOCAL_CONTESTABILITY_SHARP_BOUND.md),
[support certificates](research/SUPPORT_CERTIFICATES.md).

## 4. The native engine and its cost

### 4.1 Why Rust owns the hot loop

Rules, geometry, candidate generation, search, rasterization, inference
orchestration, arenas, and replay writing are native. These operations contain
branching, small allocations, and geometric predicates; they do not become a
GPU workload merely because the network runs on a GPU.

Python owns training and export, where PyTorch is useful and the expensive
operations are tensor computations. The durable interfaces are game datasets
and model artifacts.

### 4.2 What a transition still rebuilds

Voronoi construction clips cells against distance-ordered constraints with a
geometric stopping bound. Edge provenance identifies adjacency without
rediscovering which constraint produced an edge. Scratch buffers reduce
allocation. Legal-space construction and settlement have indexing and witness
shortcuts, but substantial geometry is still rebuilt after changes.

Incremental geometry would have to handle insertion, removals, group splits,
revival, and degeneracies. A fast result must preserve the outputs actually
needed by search and scoring; omitting work changes what a benchmark measures.

### 4.3 Production optimizations versus experiments

Production search shares immutable dense logits and packs legality masks.
Support caches are an opt-in backend. Compact undo history is a lab prototype;
it is not full reversible geometry or production tree traversal. Search still
owns positions and starts a new tree for each move.

The experiments establish useful possibilities, with mixed evidence on real
positions. Their synthetic speedups must not be treated as neural self-play
speedups.

Sources: [native simulator decision](adr/0001-native-simulator.md),
[iteration lab](research/BOARD_ITERATION_LAB.md),
[search memory](research/MCTS_MEMORY_LAB.md),
[support integration](research/SUPPORT_SEARCH_INTEGRATION.md).

## 5. Searching an action space that cannot be enumerated

### 5.1 Policy and value play different roles

The policy suggests where to search. The value estimates the result from a
position. MCTS combines prior preference, backed-up results, and an exploration
bonus to spend simulations on promising moves. Terminal transitions use actual
rule outcomes.

A good value does not compensate for never proposing the winning placement.
In this game, candidate generation is part of the learning algorithm.

### 5.2 Progressive widening limits the branching factor

There are continuously many legal centers. Search initially opens a few moves,
then increases the cumulative proposal budget approximately as a coefficient
times the square root of node visits, subject to a cap.

Too little widening misses alternatives; too much spreads simulations thinly.
This tradeoff has historically mattered more than many model changes. The
current bulk recipe widens at coefficient 6, with arenas commonly at 4.

### 5.3 The network proposes coarse-to-fine placements

The exported policy is a spatial logit map plus a pass logit. Search builds a
legal fine grid, pools regional scores, draws a coarse region, and draws a fine
placement within it. Boundary cells can map to legal points within the cell.
The policy grid is a discrete proposal representation of a continuous game.

The probability of a proposal is the product of its coarse and conditional
fine probabilities. Duplicate draws count toward proposal multiplicity; they
are not silently retried until a different move appears. Pass is handled
separately.

### 5.4 Exploration must reach policy blind spots

Uniform mass mixed into root proposals gives low-prior moves a chance to enter
search. Otherwise the policy can fail to propose a move, receive no evidence
that it is good, and reinforce that blind spot indefinitely.

Opening move temperature diversifies trajectories. Later deterministic
selection protects useful endgame decisions. These controls affect the
training distribution and belong to a run's identity.

### 5.5 Leaf batching changes both speed and search

A search can collect several pending leaves using virtual loss, evaluate them
together, and back up the results. This fills inference batches, but decisions
within the batch occur before all preceding results are known. Larger leaf
batches therefore change exploration and can reduce strength.

The stepped search driver exposes this process to the browser event loop. It
shares result assembly with native search so move selection semantics agree.

Sources: [search implementation](../crates/vgo-search/src/mcts.rs),
[coarse-to-fine sampling](../crates/vgo-search/src/coarse_fine.rs),
[stepped driver](../crates/vgo-search/src/stepped.rs).

### 5.6 Keeping useful work across actual moves

A search tree already contains predictions about later positions. Once an
actual move matches an evaluated child, its subtree can become the next root.
The reusable unit is the committed `Position`, including captures and pass
state; appending a stone to the old board is insufficient. The other player's
move must also match a searched branch before that seat can reuse work again.

Reuse depends on stable model weights, configuration and position identity.
The arena prototype retains each seat's own tree, invalidates mismatches, and
limits retained nodes/payload. Inherited visits influence play, while the new
simulation budget counts new work. Bookkeeping also costs elapsed time.

Training adds another dependency: inherited root proposals may have been drawn
without today's root noise, and inherited visits were collected under an older
search root. Defining how those affect proposal probabilities and targets is
necessary before enabling reuse in self-play. The current prototype therefore
supports zero-noise arena searches; the browser and self-play retain their
existing search behavior. See the [reuse investigation](research/TREE_REUSE_2026-10-14.md).

## 6. Turning sampled search into learnable policy targets

### 6.1 Why raw visits were insufficient

The original board-independent candidate sampler made target locations mostly
reflect sampling noise. A network could learn value while its policy received
an unstable signal. Proposing from the network's board-dependent map closed
that loop, but sampling still biases which actions become available.

### 6.2 What the replay must retain

For an explored placement, retain search visits `n`, proposal multiplicity `m`,
proposal probability `β`, and the cumulative placement draw count `K`.
Training uses unnormalized mass `n × m / (K × β)`, then normalizes the target.
Pass uses its visits directly because it was enumerated deterministically.

The denominator of the policy softmax covers the full legal grid. Unexplored
legal moves therefore receive negative learning pressure; the target support
is sparse, but the legal action set is not.

### 6.3 The correction has a limit

This is a target-side sampling correction. The current search does not apply
the preferred alternative correction to its PUCT priors. Progressive widening
also introduces candidates at different times, allowing earlier candidates to
accumulate visits before later candidates exist.

The resulting target is a practical estimator for this schedule, not a proof
of unbiased policy improvement. Storing the raw ingredients makes it possible
to audit or change the estimator without regenerating every game.

Source: [policy redesign and estimator assumptions](POLICY_REDESIGN.md).

## 7. Representation and the model

### 7.1 Seven planes encode the information the current model uses

`compact-radius` contains mover stones, opponent stones, Voronoi ridge,
settled territory, mover-relative komi, previous pass, and normalized diameter
`2r`. Raster pixels sample their centers. Stone/color features are relative to
the side to move, so the network uses one convention for both seats.

Komi tells the network what lead is required. Previous pass tells it whether
passing now ends the game. Radius distinguishes effective board sizes. Leaving
these out creates information that the model cannot recover from the stones
alone.

### 7.2 Settled territory is informative and costly

The settled plane conveys derived rule information. A distance transform and
legal-feature checks accelerate its construction, with exact query fallbacks
for unresolved pixels. This is an input-rendering technique; capture continues
to use geometric rule analysis.

Removing the plane might save substantial CPU time, but would force the model
to infer more of this relationship itself. That is an ablation question.

### 7.3 Detail and context need different resolutions

The DDRNet-inspired trunk keeps a detail branch at relatively high resolution
and a context branch at successively lower resolutions. Bilateral fusions
exchange local placement geometry and global board information. Context
attention supplements the convolutional path.

The current bulk recipe uses width 64, 16 blocks, eight GroupNorm groups, and
one attention block per context stage. Input raster resolution and policy
resolution are separate settings: changing one need not change the other.

### 7.4 Stable heads are as important as capacity

Value uses two categorical win/loss logits. Its inference utility is
`P(win) - P(loss)`. This replaced tanh plus squared error, whose gradients
vanished on confidently wrong predictions.

Training has normalized and plain twin heads. Normalized heads carry most of
the training loss, reducing incentives for trunk weight inflation; plain heads
are exported for inference. GroupNorm limits amplification through residual
blocks. fp16 stability is a requirement for the serving path.

The restart experiments expose one or two GroupNorms per convolutional block,
recorded in the checkpoint. They hold the data and training recipe fixed to
measure the speed/strength tradeoff. Missing metadata means the original two.

Ownership is an auxiliary head and currently has zero loss weight in the bulk
recipe. Final board and score records now provide material for further
auxiliary-target experiments, but their learning benefit is unestablished.

Sources: [raster contract](RASTER_REPRESENTATION.md),
[model architecture](MODEL_ARCHITECTURE.md),
[model code](../training/vgo_training/model.py).

## 8. Games, outcomes, and replay

### 8.1 Search effort and training eligibility are separate

The randomized-budget experiment distinguishes full-selected turns from cheap
turns. Full-selected turns supply both policy and value rows; cheap turns still
affect play, the final outcome, and resignation calibration. Every played ply
records its actual search budget and eligibility. Soft resignation is an
independent tail discount and can reduce even a full-selected turn's budget.
The first ply is always full, so a completed game has a replay row. A bounded
pilot produces more complete games; a full-replay follow-up has inconclusive
strength results, so fixed budgets remain the default. Tiny from-scratch pilot
corpora can fail to train a deployed head and need a health check before rating.

### 8.2 A game is the publication unit

An actor holds a fixed model, plays one game, writes that game's dataset and
sidecars, then takes another game. A staging directory is renamed into place
only after the data and manifest are written. Readers select published games.

This avoids the old shard tail, where almost all actors became idle waiting
for a few long games. At a model handoff, the old generator drains its games
while a new generator begins writing to its own generation directory.

### 8.3 Store positions instead of committing to rendered pixels

Replay stores stone coordinates and rule state along with sparse search
records. Rust renders model inputs when Python loads training data. A single
corpus can therefore support different input resolutions without another
self-play campaign.

Policy targets and legal masks are also packed in learner memory. Dense batch
tensors are expanded when needed. Once state images are compressed, policy
and mask storage become important memory costs in their own right.

### 8.4 Outcome labels include bounded-game adjudication

Natural pass/pass endings produce a rule result. Games reaching the ply cap
are scored on the actual final board, after the last move. Game sidecars retain
final areas, stones, passes, self-captures, and how the game ended.

**Known history:** allowing bots to continue on settled boards previously
produced stalling, rather than reliable natural endings. Soft resignation
reduces search after sustained losing evaluations, but continues the game to
an independently scored outcome. Thresholds are recalibrated from recorded
counterfactuals. Those counterfactuals are influenced by the cheaper continuation.

A high cap rate alone does not reopen this investigation. The useful remaining
question is whether capped scoring or cheap-tail sample weighting causes a
specific learning/evaluation problem.

Sources: [self-play architecture](SELFPLAY_ARCHITECTURE.md),
[position records](POSITION_SHARDS.md),
[shared generation](../crates/vgo-selfplay/src/generation.rs).

## 9. Learning and model replacement

### 9.1 The replay window determines the experience distribution

Training selects recent complete games by game number. The default recipe
mixes effective board sizes, uses a 150,000-sample window, and retrains after
15% turnover for eight epochs. These are script defaults, not immutable design
laws; historical runs often used other values.

The window trades recency against diversity. Many adjacent positions from one
game do not provide as many independent outcomes as positions from many games.
Game hashing keeps each game's rows together across training and validation,
with shard identity distinguishing locally overlapping game IDs.

### 9.2 Bulk retraining replaced an unstable incremental chain

Each update trains a fresh model on the latest window. This choice followed a
large observed strength gain over warm-started incremental training on the
same corpus. It prevents damaged weights from being inherited indefinitely,
although poor models can still affect later replay through their games.

The loop adopts each exported model without a promotion gate. Cheap arenas
were too uncertain and expensive to serve as a useful gate at every update.
The learner publishes the final epoch; best-epoch statistics remain diagnostic
because a small game-level validation set made checkpoint selection noisy.

### 9.3 What is optimized and what is measured

The learner trains policy cross-entropy against corrected search targets and
categorical value loss against game outcomes, with dihedral augmentation.
Adam is the retained optimizer. The ownership path is present but disabled in
the current recipe.

Loss screens an experiment and exposes failures. Playing fresh games decides
whether the resulting player improved. Compare both equal-search-budget
strength and equal-time strength when an experiment changes execution speed.

Sources: [bulk loop and defaults](../scripts/bulk-loop.sh),
[learner](../training/vgo_training/learner.py),
[supervision](../training/vgo_training/supervision.py).

## 10. Inference and the resource budget

### 10.1 Many CPU actors feed one batching broker

Actors rasterize pending leaves and submit grouped requests. A bounded broker
packs compatible groups across actors until a batch ceiling or delay deadline,
then dispatches to independent ONNX Runtime/TensorRT slots. IDs and output
counts are checked before evaluations return to search.

Two slots overlap host work and GPU execution. Higher concurrency is not
automatically better: more than two slots has a documented livelock. CUDA
blocking waits avoid spending entire CPU cores polling for GPU completion.

### 10.2 Packed inputs reduce transfer and staging work

Binary planes are bit-packed, constant planes become scalars, and other planes
use fp16. The graph expands this representation before the trunk. This saves
host/device traffic without changing the intended dense model input.

Model artifacts declare their raster shape, policy shape, and batch limit.
Engine and timing caches avoid rebuilding everything at every model load.
Changing precision, shape, or batch limits can require a new engine.

### 10.3 The full budget crosses CPU, GPU, and memory

For every move, search pays for many transitions, legal-grid operations,
renders, and evaluations. Training adds replay loading, rasterization, batch
expansion, and optimization. More actors can improve batching while increasing
tree memory and competing for CPU time. More simulations improve decisions
while reducing independent games per hour.

Queue durations are position-weighted; search and inference wall times overlap
across actors and slots. Their sums can exceed elapsed time. GPU utilization
alone cannot distinguish useful work from a stalled or spinning runtime.

Sources: [inference contract](INFERENCE_PROTOCOL.md),
[packed graph input](../training/vgo_training/packed_input.py),
[RAM measurements](research/RAM_SCALING.md).

## 11. Evaluation, verification, and evidence

### 11.1 Ratings require a connected comparison graph

Periodic arenas compare a model with sampled prior models. Bradley–Terry fits
the accumulated graph. Disconnected components cannot be assigned a meaningful
relative rating. Board size, komi, ruleset, simulations, and batching settings
must accompany a result.

Small matches have broad uncertainty. The big-board evaluation of one update
showing an improvement followed by another at parity is not, by itself, a
precise measure of the learning curve.

### 11.2 Verification has several independent jobs

1. **Mathematical justification:** explain why a finite test implements the rule.
2. **Rule fixtures:** catch tangencies, captures, revival, passes, and setup errors.
3. **Differential checks:** compare optimized transitions/search with baseline.
4. **Model parity:** check exported and packed inference against the intended model.
5. **Performance benchmarks:** establish costs on fixed, representative positions.
6. **Fresh arena games:** measure actual strength after a change.

Passing one layer does not establish the others. In particular, differential
parity with the Rust baseline cannot independently prove that both engines
agree with the mathematical rules or JavaScript reference.

### 11.3 Historical experiments need scope labels

Research records often distinguish synthetic policies, lattice positions,
saved played positions, and full neural generation. Preserve those distinctions
when interpreting a result. Experimental code removed from the current tree
can be recovered from `archive/pre-prune`; old reproduction commands may need
that branch.

Sources: [ratings fit](../training/vgo_training/bradley_terry.py),
[benchmark runner](../scripts/bench.py), [research index](research/README.md),
[rule regression backlog](../todo/004-rule-regression-suite.md).

## 12. Delivery, ownership, and the next frontier

### 12.1 The browser uses the same native rules and search

`vgo-wasm` exposes the Rust engine and stepped search. JavaScript loads an ONNX
model, runs WebGPU or a CPU fallback, and feeds evaluations back into the
search. The reference page also supports record loading, variations, undo,
move previews, and inspection of the rules.

Current client limitations include fresh trees per move, limited analysis
output, missing model identity in its public result, and unmeasured browser
memory at large search budgets. These are useful product work independent of
training a stronger model.

### 12.2 Code ownership map

| Layer | Main location | Responsibility |
| --- | --- | --- |
| Rule definition and reference | `reference/` | Rules, proofs, JavaScript oracle, reference UI/tests |
| Native state and geometry | `crates/vgo-core/` | Validation, cells, legal space, settlement, transactions, scoring |
| Move search | `crates/vgo-search/` | Proposals, widening, PUCT, backups, stepped search |
| Model inputs | `crates/vgo-raster/` | Raster planes, legal grids, packed inputs, shard helpers |
| Inference orchestration | `crates/vgo-inference/` | Model contract, batching, native sessions, timing counters |
| Games and datasets | `crates/vgo-selfplay/` | Continuous generation, arenas, replay publication, telemetry |
| Learning | `training/vgo_training/` | Replay preparation, targets, augmentation, model, optimizer, export |
| Run orchestration | `scripts/` | Bulk loop, evaluation, calibration, setup, benchmarks |
| Browser delivery | `crates/vgo-wasm/`, `client/` | Native engine in WASM, inference integration, bot API |
| Evidence and work order | `docs/research/`, `benchmarks/`, `todo/` | Results, reproducible measurements, prioritized work |

### 12.3 The next questions

The restart work now measures continuous generation and has tested spatial
pruning, input resolution, normalization and uneven search budgets. The bounded
tail audit found consistent final labels and a calibration-floor defect, now
fixed; it did not justify broad tail filtering. Arena tree reuse and native
inference deadlines now have implementations and bounded verification.

The next decisions concern promotion-grade strength evidence, shared Rust/JS
rule fixtures, and whether to extend useful arena reuse to the analysis client
or to self-play with well-defined training targets.

The governing principle is to preserve reliable rule consequences while
improving the decisions learned from a fixed amount of compute. The
[roadmap](../todo/ROADMAP.md) turns these questions into bounded work with
acceptance criteria.
