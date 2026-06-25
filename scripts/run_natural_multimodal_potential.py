#!/usr/bin/env python3
"""Natural multimodal potential — TTS + LLM free fusion through Valhalla (no preset routing).

Hypothesis: after multi-round Fate adaptation with Hub/Tile/Stem running freely,
cross-modal structure emerges → evidence for **native multimodal** potential.

Usage:
  python tools/fair_benchmark/run_natural_multimodal_potential.py --phase smoke
  python tools/fair_benchmark/run_natural_multimodal_potential.py --phase test --bake

Exit 0 = potential_confirmed; exit 2 = partial/not yet (see verdict in JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAIR = Path(__file__).resolve().parent
ENCODE = REPO / "tools/valhalla_dl_front/encode.py"
FREE_BIN = REPO / "target/release/valhalla_free_multimodal"
BAKE = REPO / "tools/model_decompose/bake_model_decompose.py"
OUT_DIR = REPO / "reports/natural_multimodal_potential"

sys.path.insert(0, str(FAIR))
from run_fair_benchmark import load_suite, select_prompts  # noqa: E402
from run_strict_incubation_experiment import strict_user_text  # noqa: E402


def ensure_bin() -> None:
    if FREE_BIN.is_file():
        return
    subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "hub-f64",
            "--release",
            "--bin",
            "valhalla_free_multimodal",
        ],
        cwd=REPO,
        check=True,
        env={**os.environ, "RUSTFLAGS": "-L /opt/cuda/lib64"},
    )


def tts(text: str, wav: Path) -> None:
    subprocess.run(["espeak", "-w", str(wav), text], check=True, capture_output=True)


def encode_wav(wav: Path) -> list[float]:
    proc = subprocess.run(
        [sys.executable, str(ENCODE)],
        input=json.dumps({"modality": "audio_path", "path": str(wav)}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)["embedding"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke")
    ap.add_argument(
        "--bake",
        action="store_true",
        help="If potential confirmed, incubate+fuse corpus → new multimodal checkpoint",
    )
    ap.add_argument("--min-samples", type=int, default=3)
    args = ap.parse_args()

    ensure_bin()
    suite = load_suite()
    prompts = [
        p
        for p in select_prompts(suite, args.phase)
        if p.get("score_type") in ("open", "mcq")
    ]
    if len(prompts) < args.min_samples:
        sys.exit(f"Need >= {args.min_samples} open/mcq prompts in phase={args.phase}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for p in prompts:
            text = strict_user_text(p)
            wav = tmp / f"{p['id']}.wav"
            tts(text, wav)
            samples.append(
                {
                    "id": p["id"],
                    "text": text,
                    "audio_features": encode_wav(wav),
                }
            )

    payload = {"optimize_cycles": 2, "samples": samples}
    env = {
        **os.environ,
        "RUSTFLAGS": "-L /opt/cuda/lib64",
        "VALHALLA_FREE_MULTIMODAL": "1",
        "VALHALLA_FREE_MULTIMODAL_MIN_CYCLES": "5",
    }
    proc = subprocess.run(
        [str(FREE_BIN)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=env,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    data = json.loads(proc.stdout)
    out_path = OUT_DIR / f"potential_{args.phase}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    s = data["summary"]
    print(f"\n=== Natural Multimodal Potential ({args.phase}) ===")
    print(f"Samples: {s['sample_count']}")
    print(f"Emergence rate: {s['emergence_rate']:.1%}")
    print(f"Avg natural score: {s['avg_natural_multimodal_score']:.3f}")
    print(f"Avg Fate rounds: {s['avg_adaptation_rounds']:.1f}")
    print(f"Hub modality separation: {s['avg_hub_modality_separation']:.4f}")
    print(f"Tile superadditive: {s['tile_superadditive_rate']:.1%}")
    print(f"Verdict: {s['verdict']}")
    print(f"→ {out_path}")

    if args.bake and s.get("potential_confirmed"):
        bake_out = REPO / f"checkpoints/valhalla_free_multimodal/{args.phase}_fused_s0.08"
        print(f"\nBaking fused structure → {bake_out}", file=sys.stderr)
        bake_script = REPO / "tools/valhalla_model_bridge/bake_valhalla_model.py"
        bake_proc = subprocess.run(
            [
                sys.executable,
                str(bake_script),
                "--body",
                "triad",
                "--phase",
                args.phase,
                "--bake-mode",
                "full",
                "--strength",
                "0.08",
                "--compose-adapter",
                "--out",
                str(bake_out),
            ],
            cwd=REPO,
            env={**env, "VALHALLA_MULTIMODAL_FUSION": "1"},
        )
        if bake_proc.returncode != 0 and bake_out.is_dir() and (bake_out / "model.safetensors").is_file():
            sys.path.insert(0, str(REPO / "tools/valhalla_model_bridge"))
            from bake_valhalla_model import finalize_manifest  # noqa: E402

            finalize_manifest(
                bake_out,
                {"weight_delta_norm": 0.0},
                base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
                bake_mode="full",
                strength=0.08,
                corpus_lines=49,
            )
            print("  recovered manifest from valhalla_export.json", file=sys.stderr)
        elif bake_proc.returncode != 0:
            sys.exit(bake_proc.returncode)

        manifest_path = bake_out / "valhalla_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["product"] = "valhalla-free-multimodal-bake"
            manifest["free_multimodal"] = {
                "protocol": s.get("protocol"),
                "phase": args.phase,
                "sample_count": s.get("sample_count"),
                "emergence_rate": s.get("emergence_rate"),
                "avg_natural_multimodal_score": s.get("avg_natural_multimodal_score"),
                "avg_adaptation_rounds": s.get("avg_adaptation_rounds"),
                "potential_confirmed": s.get("potential_confirmed"),
                "verdict": s.get("verdict"),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(f"New model: {bake_out}", file=sys.stderr)

    if not s.get("potential_confirmed"):
        sys.exit(2)
    print("\nPASS: natural multimodal potential confirmed under free fusion")


if __name__ == "__main__":
    main()
