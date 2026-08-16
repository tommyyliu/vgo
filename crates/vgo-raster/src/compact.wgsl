// Compact raster, one invocation per pixel.
//
// Mirrors `rasterize_compact_into` in lib.rs, which stays authoritative. The
// Rust reference computes distances in f64 and WGSL has no f64, so this is a
// deliberate f32 port; `rasterize_compact_shader_reference_into` is the same
// arithmetic in Rust and exists so the precision loss can be measured against
// the f64 writer without a GPU. See docs/CLIENT_BOT.md.
//
// Channels, matching COMPACT_CHANNELS:
//   0 current_stones   pixel centre within radius of a current-player stone
//   1 opponent_stones  same, opponent
//   2 voronoi_ridge    clamp(1 - (d2 - d1) / radius, 0, 1) over *all* stones
//   3 settled          uploaded; per-stone contour work stays on the CPU
//   4 komi             constant plane, mover-relative
//
// Stones arrive mover-relative and pre-split: indices [0, current_count) are the
// current player's, then [current_count, current_count + opponent_count). That
// is what `relative_stones` already produces, so the CPU side does no extra work
// and the shader needs no per-stone colour lookup.

struct Params {
    width: u32,
    height: u32,
    current_count: u32,
    opponent_count: u32,
    radius: f32,
    komi: f32,
};

// Coordinates are normalised to [0, 1], so the largest possible squared
// distance is 2. Any sentinel well above that stands in for "no stone seen
// yet"; WGSL has no infinity literal.
const NONE: f32 = 1.0e30;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stones: array<vec2<f32>>;
// One u32 per pixel rather than a bitfield: 64 KB against 16 KB packed, both
// negligible beside the 327 KB tensor this shader exists to avoid uploading.
// Pack it only if the transfer ever measures.
@group(0) @binding(2) var<storage, read> settled: array<u32>;
@group(0) @binding(3) var<storage, read_write> planes: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn rasterize_compact(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixels = params.width * params.height;
    let pixel = id.y * params.width + id.x;
    let centre = vec2<f32>(
        (f32(id.x) + 0.5) / f32(params.width),
        (f32(id.y) + 0.5) / f32(params.height),
    );

    var current_square = NONE;
    var opponent_square = NONE;
    // Nearest and second-nearest over both colours, which is what the ridge
    // reads. Kept in the same update order as the Rust writer so the two agree
    // on which stone wins an exact tie.
    var nearest_square = NONE;
    var second_square = NONE;

    for (var i = 0u; i < params.current_count; i = i + 1u) {
        let offset = centre - stones[i];
        let square = dot(offset, offset);
        current_square = min(current_square, square);
        if (square < nearest_square) {
            second_square = nearest_square;
            nearest_square = square;
        } else if (square < second_square) {
            second_square = square;
        }
    }

    let total = params.current_count + params.opponent_count;
    for (var i = params.current_count; i < total; i = i + 1u) {
        let offset = centre - stones[i];
        let square = dot(offset, offset);
        opponent_square = min(opponent_square, square);
        if (square < nearest_square) {
            second_square = nearest_square;
            nearest_square = square;
        } else if (square < second_square) {
            second_square = square;
        }
    }

    // Squaring is monotonic for nonnegative distances, so the two disc planes
    // need no square root. The ridge needs the actual distances.
    let radius_square = params.radius * params.radius;
    planes[pixel] = select(0.0, 1.0, current_square <= radius_square);
    planes[pixels + pixel] = select(0.0, 1.0, opponent_square <= radius_square);

    var ridge = 0.0;
    if (second_square < NONE) {
        let spread = sqrt(second_square) - sqrt(nearest_square);
        ridge = clamp(1.0 - spread / params.radius, 0.0, 1.0);
    }
    planes[2u * pixels + pixel] = ridge;

    planes[3u * pixels + pixel] = f32(settled[pixel]);
    planes[4u * pixels + pixel] = params.komi;
}
