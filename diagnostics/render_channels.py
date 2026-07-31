"""Render the 10 semantic channels the model reads, as a PNG contact sheet.

Written with zlib/struct rather than matplotlib so it adds no dependency to the
venv the running pipeline uses.
"""
import sys, glob, zlib, struct; sys.path.insert(0,"/home/tommy/PycharmProjects/vgo/training")
import torch, numpy as np
from vgo_training.model import DDRNetPolicyValueNet
from vgo_training import dataset as ds

NAMES = ['current_stones','opponent_stones','current_voronoi','opponent_voronoi',
         'current_distance','opponent_distance','voronoi_ridge','legal_clearance',
         'radius','previous_pass']
OUT = "/home/tommy/PycharmProjects/vgo/diagnostics/rasters"

def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))

def colorize(a):
    """magma-ish ramp on [0,1]."""
    a = np.clip(a, 0, 1)
    stops = np.array([[0,0,4],[40,11,84],[101,21,110],[159,42,99],
                      [212,72,66],[245,125,21],[252,193,56],[252,253,191]], float)
    pos = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(a, pos, stops[:, i]) for i in range(3)], -1).astype(np.uint8)

def label_bar(w, text, height=6):
    """A plain separator; tile identity is printed to stdout instead. A hand-
    rolled bitmap font was more trouble than the labels were worth."""
    return np.zeros((height, w, 3), np.uint8) + 30


d = ds.load_datasets(["artifacts/ddrnet-vs/replay/shard-000008/dataset.vgo"])
p = sorted(glob.glob("artifacts/ddrnet-vs/updates/update-*/candidate.pt"))[-1]
b = torch.load(p, map_location="cpu", weights_only=False)
m = DDRNetPolicyValueNet(channels=b["channels"], width=b["model_width"], blocks=b["blocks"],
    policy_resolution=b["policy_resolution"], variance_scaled=b.get("variance_scaled", False))
m.load_state_dict(b["state_dict"], strict=False); m.eval()

g, pl, lab = d.game_ids.numpy(), d.plies.numpy(), d.values.numpy()
lens = {int(x): int(pl[g == x].max()) for x in np.unique(g)}
gid = sorted(lens.items(), key=lambda kv: -kv[1])[0][0]
rows = np.where(g == gid)[0]; rows = rows[np.argsort(pl[rows])]
picks = rows[np.linspace(0, len(rows) - 1, 5).astype(int)]

with torch.no_grad():
    _, vals = m(d.states[picks].float())

S, PAD = 128, 4
W = 10 * (S + PAD) + PAD
H = len(picks) * (S + PAD + 6) + PAD
sheet = np.zeros((H, W, 3), np.uint8) + 18
for r, s in enumerate(picks):
    y = PAD + r * (S + PAD + 6)
    for c in range(10):
        x = PAD + c * (S + PAD)
        img = d.states[s, c].numpy()
        hi = float(img.max())
        sheet[y:y+S, x:x+S] = colorize(img / hi if hi > 1e-9 else img)
        tag = NAMES[c][:14] if r == 0 else (f"PLY {pl[s]} V{vals[r]:+.2f} L{lab[s]:+.0f}" if c == 0 else "")
        if tag: sheet[y+S:y+S+6, x:x+S] = label_bar(S, tag)
png(f"{OUT}/channels.png", sheet)
print(f"wrote channels.png  game {gid}, plies {[int(pl[s]) for s in picks]}")

# policy heat next to the board, same plies
W2 = 3 * (S + PAD) + PAD
H2 = 2 * (S + PAD + 6) + PAD + 8
sheet2 = np.zeros((H2, W2, 3), np.uint8) + 18
for i, s in enumerate(picks[:3]):
    with torch.no_grad():
        logits, v = m(d.states[s:s+1].float())
    prob = torch.softmax(logits[0], 0)
    place = prob[:-1].reshape(128, 128).numpy()
    x = PAD + i * (S + PAD)
    board = (d.states[s, 0].numpy() + 2 * d.states[s, 1].numpy()) / 2.0
    sheet2[PAD:PAD+S, x:x+S] = colorize(board)
    sheet2[PAD+S:PAD+S+6, x:x+S] = label_bar(S, f"PLY {pl[s]} STONES")
    y2 = PAD + S + PAD + 13
    sheet2[y2:y2+S, x:x+S] = colorize(place / max(place.max(), 1e-9))
    sheet2[y2+S:y2+S+6, x:x+S] = label_bar(S, f"POLICY PASS {prob[-1]:.2f}")
png(f"{OUT}/policy.png", sheet2)
print("wrote policy.png")
