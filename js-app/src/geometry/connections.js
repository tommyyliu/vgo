(function (root) {
  "use strict";

  /* Which same-colour connections can the opponent cut in one move?

     Two stones are connected while their cells share an edge, a piece AB of
     their perpendicular bisector. A new stone p takes a point q of that edge
     exactly when |q - p| < |q - i|, so it removes the whole edge -- cuts the
     link -- when |q - p| < |q - i| for every q on the edge. Expanding the
     squares, the |q|^2 terms cancel and the condition reads
     |p|^2 - |i|^2 < 2 (p - i) . q, which is linear in q: the edge points p
     takes form a half-plane's cut of the segment, so p takes the whole edge
     exactly when it takes both ends. The cut zone is therefore the open lens
     D(A, |A - i|) ∩ D(B, |B - i|), intersected over the edge's pieces when it
     has more than one, and a link is cuttable exactly when a legal placement
     lies strictly inside it.

     The legal set is the inset square minus the open 2r disks around stones;
     cutWitness below searches it exactly. A cut must reach MARGIN inside the
     lens, well above the geometry's own tolerances (1e-7 on coordinates): a
     stone that only just reaches the lens leaves a sliver of edge about 1e-7
     long that the Voronoi code still counts. So a cut zone thinner than MARGIN
     counts as uncuttable.

     One move only, and geometric: a cutting stone that would be captured at
     once still counts as a cut. */

  const MARGIN = 1e-5;          // how far inside the lens a cut must reach
  const LEGAL_SLACK = 1e-9;     // far below the engine's 1e-7, so a witness is surely legal

  function legal(position, x, y) {
    const r = position.radius, lo = r - LEGAL_SLACK, hi = 1 - r + LEGAL_SLACK;
    if (!(x >= lo && x <= hi && y >= lo && y <= hi)) return false;
    const minimum = (2 * r - LEGAL_SLACK) ** 2;
    for (const s of position.stones) {
      const dx = x - s.x, dy = y - s.y;
      if (dx * dx + dy * dy < minimum) return false;
    }
    return true;
  }

  /* Intervals on a parameter line, kept sorted and disjoint. Circles use the
     angle, unrolled onto [0, 2*pi). */
  const TAU = 2 * Math.PI;
  function intersect(a, b) {
    const out = [];
    for (const x of a) for (const y of b) {
      const lo = Math.max(x[0], y[0]), hi = Math.min(x[1], y[1]);
      if (hi > lo) out.push([lo, hi]);
    }
    return out.sort((p, q) => p[0] - q[0]);
  }
  function subtract(a, b) {
    let out = a;
    for (const y of b) {
      const next = [];
      for (const x of out) {
        if (y[1] <= x[0] || y[0] >= x[1]) { next.push(x); continue; }
        if (y[0] > x[0]) next.push([x[0], y[0]]);
        if (y[1] < x[1]) next.push([y[1], x[1]]);
      }
      out = next;
    }
    return out;
  }
  // Angles theta with cos(theta - phi) > k, as intervals on [0, 2*pi).
  function arc(phi, k) {
    if (k >= 1) return [];
    if (k < -1) return [[0, TAU]];
    const half = Math.acos(k);
    let lo = ((phi - half) % TAU + TAU) % TAU, hi = lo + 2 * half;
    return hi <= TAU ? [[lo, hi]] : [[lo, TAU], [0, hi - TAU]];
  }
  // Points of the circle (ox, oy, R) strictly inside the disk (cx, cy, rho).
  function circleInDisk(ox, oy, R, cx, cy, rho) {
    const dx = cx - ox, dy = cy - oy, d = Math.hypot(dx, dy);
    if (d < 1e-15) return R < rho ? [[0, TAU]] : [];
    return arc(Math.atan2(dy, dx), (R * R + d * d - rho * rho) / (2 * R * d));
  }
  // Points of the circle with n . p >= b (a half-plane), n a unit vector.
  function circleInHalfPlane(ox, oy, R, nx, ny, b) {
    return arc(Math.atan2(ny, nx), (b - (nx * ox + ny * oy)) / R - 1e-15);
  }
  // Parameter t on x = value (axis 0) or y = value (axis 1) inside the disk.
  function lineInDisk(axis, value, cx, cy, rho) {
    const offset = value - (axis === 0 ? cx : cy), h2 = rho * rho - offset * offset;
    if (h2 <= 0) return [];
    const centre = axis === 0 ? cy : cx, h = Math.sqrt(h2);
    return [[centre - h, centre + h]];
  }

  /* A legal point that cuts the link between stones i and j, or null.

     If a legal point lies inside the lens shrunk by MARGIN, slide it toward
     stone i. The lens corner at i is covered by i's own 2r disk, so the slide
     meets the legal set's boundary while still inside the lens (unless the lens
     is so thin that its shrunk tip reaches past 2r from i, under 1e-4 rad wide).
     So it is enough to search that boundary: each stone's 2r circle and each
     inset line, where "inside the lens", "inside the board" and "outside every
     other disk" are all intervals. Points legal only at a single point -- a
     placement touching two stones exactly, as snapping plays -- are the legal
     set's own corners, checked directly. */
  function cutWitness(position, geometry, i, j, legalVertices) {
    const stones = position.stones, r = position.radius, si = stones[i];
    const disks = [];
    for (const e of geometry.cells[i].edges) {
      if (e.j !== j) continue;
      for (const end of [e.A, e.B]) disks.push([end[0], end[1], Math.hypot(end[0] - si.x, end[1] - si.y) - MARGIN]);
    }
    if (!disks.length) return null;                        // not adjacent: nothing to cut
    const inside = (x, y) => disks.every(d => Math.hypot(x - d[0], y - d[1]) < d[2]);
    const found = (x, y) => inside(x, y) && legal(position, x, y) ? [x, y] : null;

    for (const v of legalVertices) { const w = found(v[0], v[1]); if (w) return w; }

    const D = 2 * r, lo = r, hi = 1 - r;
    // Only stones whose 2r circle reaches every lens disk can carry a witness,
    // and only stones within 4r of that one can cover part of its circle.
    const reaches = o => disks.every(d => Math.hypot(o.x - d[0], o.y - d[1]) < d[2] + D);
    for (let s = 0; s < stones.length; s++) {
      const o = stones[s];
      if (!reaches(o)) continue;
      let allowed = [[0, TAU]];
      for (const d of disks) allowed = intersect(allowed, circleInDisk(o.x, o.y, D, d[0], d[1], d[2]));
      if (!allowed.length) continue;
      for (const [nx, ny, b] of [[1, 0, lo], [-1, 0, -hi], [0, 1, lo], [0, -1, -hi]])
        allowed = intersect(allowed, circleInHalfPlane(o.x, o.y, D, nx, ny, b));
      for (let t = 0; t < stones.length && allowed.length; t++)
        if (t !== s && Math.abs(stones[t].x - o.x) < 2 * D && Math.abs(stones[t].y - o.y) < 2 * D)
          allowed = subtract(allowed, circleInDisk(o.x, o.y, D, stones[t].x, stones[t].y, D));
      for (const [a, b] of allowed) {
        const theta = (a + b) / 2, w = found(o.x + D * Math.cos(theta), o.y + D * Math.sin(theta));
        if (w) return w;
      }
    }
    for (const [axis, value] of [[0, lo], [0, hi], [1, lo], [1, hi]]) {
      let allowed = [[lo, hi]];
      for (const d of disks) allowed = intersect(allowed, lineInDisk(axis, value, d[0], d[1], d[2]));
      for (let t = 0; t < stones.length && allowed.length; t++)
        allowed = subtract(allowed, lineInDisk(axis, value, stones[t].x, stones[t].y, D));
      for (const [a, b] of allowed) {
        const m = (a + b) / 2, w = axis === 0 ? found(value, m) : found(m, value);
        if (w) return w;
      }
    }
    return null;
  }

  /* Every same-colour adjacency, split into the ones a single move can cut and
     the ones it cannot. Each entry is [i, j] with i < j. */
  function classify(position, geometry, legalVertices) {
    const stones = position.stones, solid = [], cuttable = [];
    for (let i = 0; i < stones.length; i++) for (const j of geometry.adj[i]) {
      if (j <= i || stones[i].c !== stones[j].c) continue;
      (cutWitness(position, geometry, i, j, legalVertices) ? cuttable : solid).push([i, j]);
    }
    return { solid: solid, cuttable: cuttable };
  }

  root.VGO.connections = Object.freeze({ classify: classify, cutWitness: cutWitness });
})(globalThis);
