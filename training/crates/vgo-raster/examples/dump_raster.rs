// Dumps the Rust semantic raster for the positions in a v4 shard, so the Python
// port can be compared against it byte for byte.
fn main() {
    use std::io::{Read, Write};
    use vgo_core::{Color, Position, Stone};
    use vgo_raster::{RasterConfig, rasterize};
    let args: Vec<String> = std::env::args().collect();
    let mut raw = Vec::new();
    std::fs::File::open(&args[1]).unwrap().read_to_end(&mut raw).unwrap();
    let n = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(raw[20..24].try_into().unwrap()) as usize;
    let limit: usize = args[2].parse().unwrap();
    const SC: usize = 128; const PC: usize = 64;
    let rec = 8+1+4+1+4 + SC*17 + 4 + PC*20 + 28;
    let mut out = std::fs::File::create(&args[3]).unwrap();
    for i in 0..n.min(limit) {
        let o = 32 + i*rec;
        let radius = f64::from_le_bytes(raw[o..o+8].try_into().unwrap());
        let to_move = if raw[o+8] == 0 { Color::Black } else { Color::White };
        let count = u32::from_le_bytes(raw[o+13..o+17].try_into().unwrap()) as usize;
        let mut stones = Vec::with_capacity(count);
        for s in 0..count {
            let so = o + 17 + s*17;
            stones.push(Stone::new(
                f64::from_le_bytes(raw[so..so+8].try_into().unwrap()),
                f64::from_le_bytes(raw[so+8..so+16].try_into().unwrap()),
                if raw[so+16] == 0 { Color::Black } else { Color::White }));
        }
        let position = Position::new(radius, stones, to_move);
        let r = rasterize(&position, RasterConfig::square(h));
        for &v in r.data() { out.write_all(&v.to_le_bytes()).unwrap(); }
    }
    println!("dumped {} rasters", n.min(limit));
}
