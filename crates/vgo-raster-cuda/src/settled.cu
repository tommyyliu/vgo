// Settled mask, one thread per pixel, direct from the definition.
//
//     settled(x)  <=>  min_s ||x - s||  <=  dist(x, L)
//
// where L is the legal set. See docs/SETTLED_REGION_PROBLEM.md.
//
// This is the formulation crates/vgo-core/src/settled.rs exists to *avoid*: it
// measured 42x the cost of the whole rest of the raster on a CPU, which is why
// that file solves a per-stone radial equation instead. The reason it lost on a
// CPU -- brute-force work repeated at every pixel -- is the reason it is right
// here. 16384 pixels are independent and each is a bounded reduction.
//
// Being direct also makes it *exact*, where the CPU path approximates: that one
// walks a polygon contour at 1/128 tolerance and scanline-fills it, so pixels
// within about one cell of the boundary can fall either way. Disagreement
// between the two is expected there and is the CPU's approximation error.
//
// Compiled by NVRTC at startup, so no nvcc and no host compiler are involved.

extern "C" {

// Mirrors vgo_core::numeric.
#define COORDINATE_EPSILON 1.0e-7
#define EDGE_EPSILON 1.0e-10

#ifndef VGO_REAL
#define VGO_REAL double
#endif
typedef VGO_REAL real;

// NVRTC compiles without the standard headers, so INFINITY is not defined.
// The bit pattern is unambiguous and avoids depending on what NVRTC happens to
// pull in.
__device__ inline real infinity() {
    return (real) __longlong_as_double(0x7ff0000000000000LL);
}

// The arithmetic type is chosen by the host, which prepends a #define.
//
// f64 matches the Rust exactly, but this is a consumer card: fp64 runs at 1/64
// of fp32 on GeForce, and the kernel is arithmetic-bound (batching it plateaued
// at 4.3x, so launch overhead was never the limit). f32 is worth its own
// correctness measurement here because the comparison is a threshold against a
// minimum over many candidates, where a narrowing can flip a pixel outright --
// unlike the disc and ridge channels, where f32 changed nothing.
//
// Stone coordinates stay f64 on the wire so the host does not have to convert;
// they are narrowed on load.
struct Stone { double x, y; };

__device__ inline bool in_inset(real x, real y, real radius) {
    return x >= radius - COORDINATE_EPSILON
        && x <= 1.0 - radius + COORDINATE_EPSILON
        && y >= radius - COORDINATE_EPSILON
        && y <= 1.0 - radius + COORDINATE_EPSILON;
}

// Membership in L: inside the inset and at least 2r from every stone.
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

// The candidate families that provably contain the nearest legal point, in the
// same order as legal_set::visit_candidates. Order does not change a minimum,
// but keeping it makes the two readable side by side.
// One launch covers a whole batch: blockIdx.z selects the position, and the
// stone and vertex arrays are concatenated with per-position offsets. A search
// broker already batches 32 positions per inference, so launching per position
// spent more on launch overhead than on the work -- at 14 stones the per-call
// version measured no faster than the CPU purely for that reason.
__global__ void settled_mask(
    const Stone* __restrict__ stones,
    const int* __restrict__ stone_offsets,
    const int* __restrict__ stone_counts,
    const Stone* __restrict__ vertices,
    const int* __restrict__ vertex_offsets,
    const int* __restrict__ vertex_counts,
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
    vertices += vertex_offsets[item];
    const int stone_count = stone_counts[item];
    const int vertex_count = vertex_counts[item];
    out += (size_t) item * (size_t) width * (size_t) height;
    const real radius = (real) radius_in;
    const real px = ((real) column + (real) 0.5) / (real) width;
    const real py = ((real) row + (real) 0.5) / (real) height;

    // Nearest stone. An empty board has no stone, so nothing is settled.
    real nearest_stone = infinity();
    for (int i = 0; i < stone_count; ++i) {
        const real dx = px - (real) stones[i].x;
        const real dy = py - (real) stones[i].y;
        const real distance = sqrt(dx * dx + dy * dy);
        if (distance < nearest_stone) {
            nearest_stone = distance;
        }
    }

    // dist(x, L). Infinite when L is empty, which makes every point settled.
    real legal = infinity();

    // 1. the point itself
    consider(px, py, px, py, radius, stones, stone_count, &legal);

    // 2. for each stone, 2r away along the ray from the stone to the point
    const real diameter = (real) 2.0 * radius;
    for (int i = 0; i < stone_count; ++i) {
        const real dx = px - (real) stones[i].x;
        const real dy = py - (real) stones[i].y;
        const real radial = sqrt(dx * dx + dy * dy);
        if (radial < EDGE_EPSILON) {
            // The point is the stone centre: the ray is undefined, so take the
            // four axis directions, as the Rust does.
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

    // 3. the four inset projections
    consider(radius, py, px, py, radius, stones, stone_count, &legal);
    consider((real) 1.0 - radius, py, px, py, radius, stones, stone_count, &legal);
    consider(px, radius, px, py, radius, stones, stone_count, &legal);
    consider(px, (real) 1.0 - radius, px, py, radius, stones, stone_count, &legal);

    // 4. the legal-set vertices, precomputed on the host. They are already
    // filtered to members of L, so they skip the containment test.
    for (int i = 0; i < vertex_count; ++i) {
        const real dx = px - (real) vertices[i].x;
        const real dy = py - (real) vertices[i].y;
        const real distance = sqrt(dx * dx + dy * dy);
        if (distance < legal) {
            legal = distance;
        }
    }

    out[row * width + column] = (nearest_stone <= legal) ? 1 : 0;
}

}
