#!/usr/bin/env bash
# Reproduce natural multimodal fusion experiment (requires Valhalla checkout).
set -euo pipefail
VALHALLA_ROOT="${VALHALLA_ROOT:-$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd || echo "$HOME/AI/Valhalla")}"
cd "$VALHALLA_ROOT"

export RUSTFLAGS='-L /opt/cuda/lib64'
export VALHALLA_FREE_MULTIMODAL=1
export VALHALLA_FREE_MULTIMODAL_MIN_CYCLES=5

echo "=== Build ==="
cargo build -p hub-f64 --release --bin valhalla_free_multimodal --bin valhalla_fused_lm

echo "=== Potential experiment (test) ==="
python3 tools/fair_benchmark/run_natural_multimodal_potential.py --phase test --bake

echo "=== QA on baked model ==="
MODEL="$VALHALLA_ROOT/checkpoints/valhalla_free_multimodal/test_fused_s0.08"
target/release/valhalla_fused_lm --phase test --model "$MODEL" \
  --out reports/natural_multimodal_potential/fused_baked_full_test.json

echo "Done."
