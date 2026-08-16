// Direct per-pixel settled-region evaluation, one invocation per pixel.
//
// This is the "per-pixel, on a GPU" formulation from
// docs/SETTLED_REGION_PROBLEM.md: the direct form is 42x too slow on a CPU but
// is embarrassingly parallel. A pixel is settled when some stone is at least as
// close to it as the nearest legal placement:
//
//     settled(x) = exists stone s : ||x - s|| <= dist(x, L)
//
// `dist(x, L)` is a minimum over the candidate set that provably contains the
// nearest legal point (the query itself, each stone pushed off at one diameter,
// the four board-edge projections, and the legal-set vertices V). Each candidate
// is tested for membership in L, which is O(n), so one pixel costs
// O((n + |V|) * n). That is the cost the CPU contour solver exists to avoid and
// the one a GPU absorbs for free.
//
// Computed in f32 throughout: WGSL has no f64. The authoritative CPU path
// (`settled_mask` in lib.rs) works in f64, so the two can disagree on a pixel
// whose centre sits within f32 epsilon of the settled boundary. The validation
// test measures exactly that and requires it to be vanishingly rare, matching
// the existing f32-vs-f64 contract for `compact.wgsl`.

struct Params {
    width: u32,
    height: u32,
    stone_count: u32,
    vertex_count: u32,
    radius: f32,
    _pad: vec3<f32>,
};

// `contains` in legal_set.rs allows `2r - COORDINATE_EPSILON`, i.e. a placement
// may sit this far inside the exclusion disc and still read as legal. The CPU
// path applies it in f64; here it is f32.
const COORDINATE_EPSILON: f32 = 1.0e-7;
const NONE: f32 = 1.0e30;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stones: array<vec2<f32>>;
// Legal-set vertices, x/y interleaved.
@group(0) @binding(2) var<storage, read> vertices: array<vec2<f32>>;
// One u32 per pixel, 0/1.
@group(0) @binding(3) var<storage, read_write> settled: array<u32>;

fn in_inset(p: vec2<f32>) -> bool {
    let r = params.radius;
    return p.x >= r - COORDINATE_EPSILON && p.x <= 1.0 - r + COORDINATE_EPSILON
        && p.y >= r - COORDINATE_EPSILON && p.y <= 1.0 - r + COORDINATE_EPSILON;
}

// Membership in the legal set L: inside the inset and clear of every stone by
// at least 2r - COORDINATE_EPSILON.
fn contains(p: vec2<f32>) -> bool {
    if (!in_inset(p)) {
        return false;
    }
    let minimum = 2.0 * params.radius - COORDINATE_EPSILON;
    let minimum_squared = minimum * minimum;
    for (var i = 0u; i < params.stone_count; i = i + 1u) {
        let d = p - stones[i];
        if (dot(d, d) < minimum_squared) {
            return false;
        }
    }
    return true;
}

// `dist(x, L)`: the minimum distance from `x` to a legal point, over the same
// candidate families `visit_candidates` in legal_set.rs enumerates. Returns
// NONE when no candidate is legal (the legal set is empty).
fn distance_to_legal(x: vec2<f32>) -> f32 {
    var best = NONE;

    // 1. x itself.
    if (contains(x)) {
        return 0.0;
    }

    let diameter = 2.0 * params.radius;

    // 2. Each stone pushed off at exactly one diameter along the ray from x.
    for (var i = 0u; i < params.stone_count; i = i + 1u) {
        let s = stones[i];
        let d = x - s;
        let radial = length(d);
        if (radial < 1.0e-10) {
            // Query exactly on a stone centre: try the four axes.
            let axes = array<vec2<f32>, 4>(
                vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0),
                vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0),
            );
            for (var k = 0u; k < 4u; k = k + 1u) {
                let c = s + diameter * axes[k];
                if (contains(c)) {
                    best = min(best, length(c - x));
                }
            }
        } else {
            let c = s + diameter * (d / radial);
            if (contains(c)) {
                best = min(best, length(c - x));
            }
        }
    }

    // 3. The four board-edge projections.
    let r = params.radius;
    let edges = array<vec2<f32>, 4>(
        vec2<f32>(r, x.y), vec2<f32>(1.0 - r, x.y),
        vec2<f32>(x.x, r), vec2<f32>(x.x, 1.0 - r),
    );
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (contains(edges[k])) {
            best = min(best, length(edges[k] - x));
        }
    }

    // 4. The legal-set vertices.
    for (var i = 0u; i < params.vertex_count; i = i + 1u) {
        let v = vertices[i];
        if (contains(v)) {
            best = min(best, length(v - x));
        }
    }

    return best;
}

@compute @workgroup_size(8, 8, 1)
fn settled_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel = id.y * params.width + id.x;
    let x = vec2<f32>(
        (f32(id.x) + 0.5) / f32(params.width),
        (f32(id.y) + 0.5) / f32(params.height),
    );

    let d = distance_to_legal(x);
    // `dist(x, L) = +inf` for an empty legal set makes everything settled, so
    // NONE (the "no candidate" sentinel) reads as settled.
    if (d >= NONE) {
        settled[pixel] = 1u;
        return;
    }

    // Settled when some stone is at least as close as the nearest legal point.
    var result = 0u;
    for (var i = 0u; i < params.stone_count; i = i + 1u) {
        let s = stones[i];
        let ds = length(x - s);
        if (ds <= d) {
            result = 1u;
            break;
        }
    }
    settled[pixel] = result;
}
