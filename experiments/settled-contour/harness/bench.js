(function (root) {
  "use strict";

  const registry = [];

  function register(name, note, compute) {
    registry.push({ name: name, note: note, compute: compute });
  }

  // Median of as many runs as fit in the budget, after one warm-up.
  function time(fn, budgetMs, maxRuns) {
    fn();
    const times = [];
    const start = performance.now();
    while (times.length < maxRuns && performance.now() - start < budgetMs) {
      const t0 = performance.now();
      fn();
      times.push(performance.now() - t0);
    }
    times.sort(function (a, b) { return a - b; });
    return { median: times[times.length >> 1], runs: times.length, best: times[0] };
  }

  function pad(value, width) { return String(value).padStart(width); }
  function fixed(value, digits, width) { return pad(Number(value).toFixed(digits), width); }

  function run(options) {
    const settings = options || {};
    const side = settings.side || 300;
    const budget = settings.budget || 900;
    const maxRuns = settings.maxRuns || 21;
    const onlyCandidates = settings.candidates;
    const onlyCases = settings.cases;

    const cases = root.CORPUS.build().filter(function (item) {
      return !onlyCases || onlyCases.includes(item.name);
    });
    const candidates = registry.filter(function (item) {
      return !onlyCandidates || onlyCandidates.includes(item.name);
    });

    const lines = [];
    const log = function (text) { lines.push(text); };

    log("SETTLED CONTOUR — candidate bench");
    log("oracle grid " + side + "x" + side + " exact samples per case; error = |g|/2 lower bound in board units");
    log("");

    const totals = {};
    candidates.forEach(function (candidate) { totals[candidate.name] = { time: 0, mismatches: 0, worst: 0, failed: 0 }; });

    cases.forEach(function (item) {
      const samples = root.ORACLE.sample(item.position, side);
      log("### " + item.name + "  (" + item.position.stones.length + " stones, r=" +
          item.position.radius.toFixed(5) + ") — " + item.note);
      log("    settled fraction (exact): " + (samples.settledCount / samples.count).toFixed(4));
      log("    candidate            time      runs   mismatch    false+   false-   worst error");
      log("    -------------------- --------- ----- ---------- -------- -------- -------------");

      candidates.forEach(function (candidate) {
        let pathData = null, failure = null;
        try {
          pathData = candidate.compute(item.position);
        } catch (error) {
          failure = error && error.message ? error.message : String(error);
        }
        if (failure) {
          totals[candidate.name].failed++;
          log("    " + candidate.name.padEnd(20) + " THREW: " + failure);
          return;
        }
        const timing = time(function () { candidate.compute(item.position); }, budget, maxRuns);
        const result = root.ORACLE.score(pathData, samples);
        totals[candidate.name].time += timing.median;
        totals[candidate.name].mismatches += result.mismatches;
        totals[candidate.name].worst = Math.max(totals[candidate.name].worst, result.worstError);
        log("    " + candidate.name.padEnd(20) +
            fixed(timing.median, 2, 8) + "ms" + pad(timing.runs, 6) +
            fixed(100 * result.mismatchRate, 4, 9) + "%" +
            pad(result.falseIn, 9) + pad(result.falseOut, 9) +
            "   " + result.worstError.toExponential(2));
      });
      log("");
    });

    log("### totals across " + cases.length + " cases");
    log("    candidate            total time   mismatches   worst error   threw");
    log("    -------------------- ---------- ------------ ------------- -------");
    candidates.forEach(function (candidate) {
      const total = totals[candidate.name];
      log("    " + candidate.name.padEnd(20) + fixed(total.time, 2, 9) + "ms" +
          pad(total.mismatches, 13) + "   " + total.worst.toExponential(2) + pad(total.failed, 8));
    });

    return lines.join("\n");
  }

  root.BENCH = Object.freeze({ register: register, run: run, registry: registry });
})(globalThis);
