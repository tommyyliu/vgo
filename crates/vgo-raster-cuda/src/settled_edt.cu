// Settled mask by separable distance transform, ported from
// `crates/vgo-raster/src/edt.rs`.
//
// `settled.cu` beside this evaluates the definition directly:
//
//     settled(x)  <=>  min_s ||x - s||  <=  dist(x, L)
//
// which costs O(stones^2) per pixel, because each of the O(stones) candidate
// legal points has to be tested for membership in L against every stone. At 28
// stones and 256x256 that is 62 million operations per mask. It is the
// formulation `vgo-core/src/settled.rs` exists to avoid, kept there on the
// theory that a GPU's parallelism would pay for the extra work. Measured, it
// does not: the whole card managed 3900 masks/s against 980 per CPU core, so
// 32 cores beat it eightfold.
//
// This is the same reformulation the CPU uses, which is O(stones) rather than
// O(stones^2) per pixel:
//
//     D_L_grid  -- distance to a *sampled* legal set, by exact Euclidean
//                  transform over the grid
//     D_S       -- distance to the nearest stone
//
//     D_S <= D_L_grid - slack  =>  settled     (soundly, since D_L_true <= D_L_grid)
//     D_S >  D_L_grid          =>  not settled (soundly, since sampling only misses)
//     otherwise                =>  undecided, and only then is the exact test run
//
// `slack` is a full grid diagonal, not half of one: half assumes every point of
// L has a cell *centre* within that distance, which fails along the set's
// boundary. That is `edt.rs`'s finding and the reason it says so at length.
//
// The undecided band is ~55 pixels of 16384, so the expensive path runs on
// about 0.3% of threads. That is warp divergence, but bounded, and vastly
// cheaper than paying it everywhere.
//
// Operation counts per mask at 28 stones, 256x256:
//
//     direct         62.4 M
//     this            4.0 M     legal set 1.8, transform 0.3, nearest 1.8
//
// The transform is separable -- a 1D lower-envelope pass down columns, then
// across rows -- and every line is independent, which is what makes it a GPU
// algorithm at all. Occupancy is only `width` threads per pass, low for this
// card, but 15x less work swamps that.

extern "C" {

#define COORDINATE_EPSILON 1.0e-7
#define EDGE_EPSILON 1.0e-10
#define ABSENT 1.0e20

#ifndef VGO_REAL
#define VGO_REAL double
#endif
typedef VGO_REAL real;

struct Stone { double x, y; };

__device__ inline bool in_inset(real x, real y, real radius) {
    return x >= radius - COORDINATE_EPSILON
        && x <= 1.0 - radius + COORDINATE_EPSILON
        && y >= radius - COORDINATE_EPSILON
        && y <= 1.0 - radius + COORDINATE_EPSILON;
}

// One cell of the sampled legal set. Mirrors `sampled_legal_set`, which starts
// from the inset rectangle and clears each stone's exclusion disc; gathering
// per cell rather than scattering per stone avoids write races and is the same
// predicate.
__global__ void sample_legal(
    const Stone* __restrict__ stones,
    const int* __restrict__ stone_offsets,
    const int* __restrict__ stone_counts,
    double radius_in,
    int width,
    int height,
    double* __restrict__ field
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int item = blockIdx.z;
    if (column >= width || row >= height) {
        return;
    }
    stones += stone_offsets[item];
    const int stone_count = stone_counts[item];
    field += (size_t) item * (size_t) width * (size_t) height;

    const real radius = (real) radius_in;
    const real x = ((real) column + (real) 0.5) / (real) width;
    const real y = ((real) row + (real) 0.5) / (real) height;

    bool legal = in_inset(x, y, radius);
    if (legal) {
        const real exclusion = (real) 2.0 * radius - (real) COORDINATE_EPSILON;
        const real exclusion_squared = exclusion * exclusion;
        for (int i = 0; i < stone_count; ++i) {
            const real dx = x - (real) stones[i].x;
            const real dy = y - (real) stones[i].y;
            if (dx * dx + dy * dy < exclusion_squared) {
                legal = false;
                break;
            }
        }
    }
    field[row * width + column] = legal ? 0.0 : ABSENT;
}

// Felzenszwalb & Huttenlocher's lower envelope, one line. A direct port of
// `transform_1d`, including the `k > 0` guard, which that function documents as
// protection against a degenerate row rather than an optimisation.
//
// Operates on contiguous lines; callers gather and scatter.
__device__ void transform_1d(
    double* __restrict__ f,
    double* __restrict__ d,
    int* __restrict__ v,
    double* __restrict__ z,
    int n
) {
    if (n == 0) {
        return;
    }
    int k = 0;
    v[0] = 0;
    z[0] = -ABSENT * ABSENT;
    z[1] = ABSENT * ABSENT;
    for (int q = 1; q < n; ++q) {
        const double fq = f[q];
        double s;
        for (;;) {
            const double fv = f[v[k]];
            const double qf = (double) q, vf = (double) v[k];
            s = ((fq + qf * qf) - (fv + vf * vf)) / (2.0 * qf - 2.0 * vf);
            if (k > 0 && s <= z[k]) {
                k -= 1;
                continue;
            }
            break;
        }
        if (k == 0 && s <= z[0]) {
            v[0] = q;
            z[1] = ABSENT * ABSENT;
            continue;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = ABSENT * ABSENT;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < (double) q) {
            k += 1;
        }
        const double offset = (double) q - (double) v[k];
        d[q] = offset * offset + f[v[k]];
    }
}

// Pass one: down each column. One thread per column per batch item.
//
// The line is gathered into contiguous scratch, transformed, and scattered
// back. Handing `transform_1d` strided pointers instead looks tidier and is
// wrong: its output write `d[q * stride]` runs to `(n-1) * stride`, so a
// per-line buffer sized `n` is overrun by a factor of `stride`. That is what
// the CPU's copy into `source`/`result` is quietly avoiding.
__global__ void edt_columns(
    double* __restrict__ field,
    double* __restrict__ source,
    double* __restrict__ result,
    int* __restrict__ vertices,
    double* __restrict__ boundaries,
    int width,
    int height
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int item = blockIdx.y;
    if (column >= width) {
        return;
    }
    const size_t plane = (size_t) width * (size_t) height;
    const int longest = width > height ? width : height;
    const size_t line = (size_t) item * (size_t) width + (size_t) column;
    double* f = source + line * (size_t) longest;
    double* d = result + line * (size_t) longest;
    double* base = field + (size_t) item * plane + column;

    for (int row = 0; row < height; ++row) {
        f[row] = base[(size_t) row * width];
    }
    transform_1d(f, d, vertices + line * (size_t) longest,
                 boundaries + line * (size_t) (longest + 1), height);
    for (int row = 0; row < height; ++row) {
        base[(size_t) row * width] = d[row];
    }
}

// Pass two: across each row.
__global__ void edt_rows(
    double* __restrict__ field,
    double* __restrict__ source,
    double* __restrict__ result,
    int* __restrict__ vertices,
    double* __restrict__ boundaries,
    int width,
    int height
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int item = blockIdx.y;
    if (row >= height) {
        return;
    }
    const size_t plane = (size_t) width * (size_t) height;
    const int longest = width > height ? width : height;
    const size_t line = (size_t) item * (size_t) height + (size_t) row;
    double* f = source + line * (size_t) longest;
    double* d = result + line * (size_t) longest;
    double* base = field + (size_t) item * plane + (size_t) row * width;

    for (int column = 0; column < width; ++column) {
        f[column] = base[column];
    }
    transform_1d(f, d, vertices + line * (size_t) longest,
                 boundaries + line * (size_t) (longest + 1), width);
    for (int column = 0; column < width; ++column) {
        base[column] = d[column];
    }
}

// Membership in L, for the exact test only. This is the O(stones) predicate the
// direct kernel pays at every candidate at every pixel; here it runs on the
// undecided band alone.
__device__ inline bool contains(
    real x, real y, real radius, const Stone* stones, int stone_count
) {
    if (!in_inset(x, y, radius)) {
        return false;
    }
    const real minimum = (real) 2.0 * radius - (real) COORDINATE_EPSILON;
    const real minimum_squared = minimum * minimum;
    for (int i = 0; i < stone_count; ++i) {
        const real dx = x - (real) stones[i].x;
        const real dy = y - (real) stones[i].y;
        if (dx * dx + dy * dy < minimum_squared) {
            return false;
        }
    }
    return true;
}

__device__ inline void consider(
    real cx, real cy, real px, real py,
    real radius, const Stone* stones, int stone_count, real* best
) {
    if (!contains(cx, cy, radius, stones, stone_count)) {
        return;
    }
    const real dx = px - cx;
    const real dy = py - cy;
    const real distance = sqrt(dx * dx + dy * dy);
    if (distance < *best) {
        *best = distance;
    }
}

// dist(x, L), exactly, over the candidate families that provably contain the
// nearest legal point. Same order as `legal_set::visit_candidates`.
__device__ real exact_distance_to_legal_set(
    real px, real py, real radius,
    const Stone* stones, int stone_count,
    const Stone* vertices, int vertex_count
) {
    real legal = ABSENT;
    consider(px, py, px, py, radius, stones, stone_count, &legal);

    const real diameter = (real) 2.0 * radius;
    for (int i = 0; i < stone_count; ++i) {
        const real dx = px - (real) stones[i].x;
        const real dy = py - (real) stones[i].y;
        const real radial = sqrt(dx * dx + dy * dy);
        if (radial < EDGE_EPSILON) {
            consider((real) stones[i].x + diameter, (real) stones[i].y, px, py, radius, stones, stone_count, &legal);
            consider((real) stones[i].x - diameter, (real) stones[i].y, px, py, radius, stones, stone_count, &legal);
            consider((real) stones[i].x, (real) stones[i].y + diameter, px, py, radius, stones, stone_count, &legal);
            consider((real) stones[i].x, (real) stones[i].y - diameter, px, py, radius, stones, stone_count, &legal);
        } else {
            consider(
                (real) stones[i].x + diameter * (dx / radial),
                (real) stones[i].y + diameter * (dy / radial),
                px, py, radius, stones, stone_count, &legal
            );
        }
    }

    consider(radius, py, px, py, radius, stones, stone_count, &legal);
    consider((real) 1.0 - radius, py, px, py, radius, stones, stone_count, &legal);
    consider(px, radius, px, py, radius, stones, stone_count, &legal);
    consider(px, (real) 1.0 - radius, px, py, radius, stones, stone_count, &legal);

    for (int i = 0; i < vertex_count; ++i) {
        const real dx = px - (real) vertices[i].x;
        const real dy = py - (real) vertices[i].y;
        const real distance = sqrt(dx * dx + dy * dy);
        if (distance < legal) {
            legal = distance;
        }
    }
    return legal;
}

// The decision, per pixel. `field` holds the squared grid distance to the
// sampled legal set, in cells, from the two transform passes.
__global__ void settled_from_field(
    const Stone* __restrict__ stones,
    const int* __restrict__ stone_offsets,
    const int* __restrict__ stone_counts,
    const Stone* __restrict__ vertices,
    const int* __restrict__ vertex_offsets,
    const int* __restrict__ vertex_counts,
    const double* __restrict__ field,
    double radius_in,
    int width,
    int height,
    unsigned char* __restrict__ out
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int item = blockIdx.z;
    if (column >= width || row >= height) {
        return;
    }
    stones += stone_offsets[item];
    const int stone_count = stone_counts[item];
    const size_t plane = (size_t) width * (size_t) height;
    field += (size_t) item * plane;
    out += (size_t) item * plane;

    const real radius = (real) radius_in;
    const real px = ((real) column + (real) 0.5) / (real) width;
    const real py = ((real) row + (real) 0.5) / (real) height;

    // An empty board settles nothing: no stone owns anything.
    if (stone_count == 0) {
        out[row * width + column] = 0;
        return;
    }

    real nearest = ABSENT;
    for (int i = 0; i < stone_count; ++i) {
        const real dx = px - (real) stones[i].x;
        const real dy = py - (real) stones[i].y;
        const real squared = dx * dx + dy * dy;
        if (squared < nearest) {
            nearest = squared;
        }
    }
    nearest = sqrt(nearest);

    const real spacing = (real) 1.0 / (real) width;
    const real slack = spacing * (real) 1.4142135623730951;
    const real sampled = sqrt(field[row * width + column]) * spacing;

    unsigned char result;
    if (nearest <= sampled - slack) {
        result = 1;
    } else if (nearest > sampled) {
        result = 0;
    } else {
        // The undecided band. ~0.3% of pixels, and the only place the
        // O(stones^2) work happens.
        vertices += vertex_offsets[item];
        const int vertex_count = vertex_counts[item];
        const real legal = exact_distance_to_legal_set(
            px, py, radius, stones, stone_count, vertices, vertex_count
        );
        result = (nearest <= legal) ? 1 : 0;
    }
    out[row * width + column] = result;
}

}

