#!/usr/bin/env python3
"""Packages a browser bot as a GitHub release of this repo.

A release is flat: every file is uploaded under its own name, and a site fetches
`https://github.com/tommyyliu/vgo/releases/download/<tag>/<name>`. This script
collects the files into `<out>/assets/`, checks their names, and writes what goes
with them:

  manifest.json   the version, names, sizes and sha256 of every asset, the model's
                  I/O contract, and where the bundle and model came from
  SHA256SUMS      the same hashes, for `sha256sum -c`
  NOTES.md        the release description, to paste into GitHub

The bundle and the model must carry their own hash in their names
(`vgo-bot-<sha8>.js`, `model-<sha8>.onnx`), so a site that caches them forever
never serves a stale file under a reused name. A release is never changed after
it is published: a new bot is a new tag.

The tag is a version, `bot-v<major>.<minor>`. A major release needs the site to
change code to use it: the bundle's API, the model's inputs or outputs (raster
kind, planes, policy shape), the onnxruntime files, or the rules the bot plays.
A minor release is a drop-in: a new model or search under the same contract,
taken by pointing the site at the new release.

  scripts/package-bot-release.py --tag bot-v1.0 \\
      --bundle .../vgo-bot-30786b16.js --model .../model-e5affd23.onnx \\
      --model-manifest .../update-000207-dense.onnx.json --ort-dir .../ort-1.20.1 \\
      --ort-version 1.20.1 --bundle-commit b71d6d70ff98 --out artifacts/bot-release-v1.0
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import sys

# The three files onnxruntime-web needs; it fetches the last two from beside the first.
ORT_FILES = ("ort.all.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm")
# What a site needs to know about the model, copied from the export's manifest. Paths are
# left out on purpose: they name the machine the model was trained on.
MODEL_KEYS = ("checkpoint_sha256", "opset", "input", "outputs", "model")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    sys.exit(f"package-bot-release: {message}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="release tag, bot-v<major>.<minor>")
    p.add_argument("--bundle", required=True, type=pathlib.Path, help="vgo-bot-<sha8>.js from web-bot")
    p.add_argument("--model", required=True, type=pathlib.Path, help="model-<sha8>.onnx")
    p.add_argument("--model-manifest", type=pathlib.Path, help="the export's .onnx.json, for the I/O contract")
    p.add_argument("--ort-dir", required=True, type=pathlib.Path, help="onnxruntime-web's dist directory, or a copy of the three files")
    p.add_argument("--ort-version", required=True)
    p.add_argument("--bundle-commit", help="vgo commit the bundle was built from, if known")
    p.add_argument("--note", default="", help="a paragraph for NOTES.md: what this bot is and why it was released")
    p.add_argument("--out", required=True, type=pathlib.Path)
    a = p.parse_args()

    files = [("bundle", a.bundle), ("model", a.model)] + [("onnxruntime", a.ort_dir / name) for name in ORT_FILES]
    for role, path in files:
        if not path.is_file():
            fail(f"{role} file {path} is missing")

    hashes = {path: sha256(path) for _, path in files}
    for role, path, prefix, suffix in (("bundle", a.bundle, "vgo-bot-", ".js"), ("model", a.model, "model-", ".onnx")):
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            fail(f"the {role} must be named {prefix}<sha8>{suffix}, not {name}")
        if name[len(prefix):-len(suffix)] != hashes[path][:8]:
            fail(f"{name} does not start with its own sha256 ({hashes[path][:8]})")
    version = re.fullmatch(r"bot-v(\d+)\.(\d+)", a.tag)
    if not version:
        fail(f"tag {a.tag} should be a version, bot-v<major>.<minor>")

    model_info = None
    if a.model_manifest:
        exported = json.loads(a.model_manifest.read_text())
        if exported.get("onnx_sha256") != hashes[a.model]:
            fail(f"{a.model_manifest} describes a different model ({exported.get('onnx_sha256', '?')[:8]})")
        model_info = {key: exported[key] for key in MODEL_KEYS if key in exported}

    assets = a.out / "assets"
    if a.out.exists():
        fail(f"{a.out} already exists; a release is packaged once")
    assets.mkdir(parents=True)
    for _, path in files:
        shutil.copy2(path, assets / path.name)

    entries = [{"name": path.name, "role": role, "size": path.stat().st_size, "sha256": hashes[path]} for role, path in files]
    manifest = {
        "schema": "vgo.bot-release.v1",
        "tag": a.tag,
        "version": f"{version[1]}.{version[2]}",
        # What a site checks before using the release: a model fed the wrong raster
        # kind plays badly without any error.
        "raster_kind": (model_info or {}).get("model", {}).get("raster_kind"),
        "download": f"https://github.com/tommyyliu/vgo/releases/download/{a.tag}/",
        "bundle_commit": a.bundle_commit,
        "onnxruntime_web": a.ort_version,
        "model": model_info,
        "files": entries,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.out / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['name']}\n" for e in entries))

    rows = "\n".join(f"| `{e['name']}` | {e['role']} | {e['size']:,} | `{e['sha256'][:16]}…` |" for e in entries)
    shape = model_info and model_info.get("model", {})
    model_line = (
        f"The model is a {shape.get('architecture', '?')} network, {shape.get('parameters', 0):,} parameters, "
        f"reading `{shape.get('raster_kind', '?')}` rasters."
        if shape else "The model's export manifest was not supplied."
    )
    bundle_line = f"The bundle was built from vgo `{a.bundle_commit}`." if a.bundle_commit else "The commit the bundle was built from was not recorded."
    title = f"Bot v{version[1]}.{version[2]} (model {hashes[a.model][:8]})"
    (a.out / "NOTES.md").write_text(
        f"# {title}\n\n{a.note}\n\n{model_line} {bundle_line} onnxruntime-web is {a.ort_version}, unmodified (MIT).\n\n"
        f"| file | role | bytes | sha256 |\n|---|---|---|---|\n{rows}\n\n"
        f"Every file is fetched from `{manifest['download']}<name>`. "
        "`manifest.json` has the full hashes and the model's inputs and outputs; `SHA256SUMS` checks a download with `sha256sum -c`.\n"
    )
    print(f"packaged {a.tag} ({title}) in {a.out}: upload everything in assets/ plus manifest.json and SHA256SUMS; NOTES.md is the description")


if __name__ == "__main__":
    main()
