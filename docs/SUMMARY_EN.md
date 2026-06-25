# Valhalla Natural Multimodal Fusion — Experiment Summary

**Protocol**: `valhalla-free-multimodal-fusion-v1`  
**Date**: 2026-06-25  
**Publisher**: Rogue Intelligence LNC.

---

## Objective

Test whether **TTS + LLM**, after decomposition, can **emerge native multimodal structure** when Valhalla bodies (Hub / Tile / Stem / Fate) run **freely** with multi-round adaptation — without preset routing, sidecar JSON, or concat-at-embed hybrids.

---

## Method

1. **Input**: LLM text ports + TTS mel ports (espeak → `valhalla_dl_front/encode.py`)
2. **Free pulse**: Each round alternates text / audio / mixed signals into **Hub, Tile, and Stem** (no Stem=text-only rule)
3. **Fate adapt**: 5+ rounds until marginal stability or convergence
4. **Metrics** (discovery-only):
   - Tile/stem **superadditivity** (fusion > max single-modality)
   - **Hub modality separation** (prefs shift under text vs audio)
   - **Mismatch binding** (matched fusion patch ≠ mismatched text+audio)
   - `natural_multimodal_score` → `emergence_likely`

---

## Structure-layer results

| Metric | Smoke (n=4) | **Test (n=107)** |
|--------|-------------|------------------|
| Emergence rate | 50% | **64.5%** |
| Avg natural score | 0.825 | **0.823** |
| Tile superadditivity | 100% | **99.1%** |
| Stem superadditivity | 100% | **100%** |
| Avg Fate rounds | 5 | **5** |
| **Verdict** | CONFIRMED | **POTENTIAL_CONFIRMED** |

**Conclusion (structure)**: Cross-modal structure **reproducibly emerges** under free fusion. Native multimodal **potential is confirmed** at the structure layer.

Artifact: `artifacts/potential_test_summary.json`

---

## Baked checkpoint & QA

After potential confirmation, a **new LM checkpoint** was baked:

- Local path: `checkpoints/valhalla_free_multimodal/test_fused_s0.08/` (~988MB, not in git)
- Manifest: `artifacts/checkpoint_manifest.json`

**Note**: Current bake uses corpus triad incubation + compose adapter; per-prompt free-fusion structure is **not yet aggregated** into weights.

### Fair benchmark test (145 prompts)

| Track | Total | Open | MCQ | Numeric |
|-------|-------|------|-----|---------|
| **This experiment (baked)** | **102/145** | **56/61** | 15/46 | 31/38 |
| Hybrid 2.0 (reference) | 117/145 | 57/61 | 28/46 | 32/38 |
| Prior baked | 115/145 | 56/61 | 26/46 | — |

Open matches prior baked; MCQ lags Hybrid. Structure potential ≠ full QA closure yet.

### Open + TTS compose (same checkpoint)

Open **56/61** · `structure_recompose_open/x*/structure_recompose_gen`  
Artifact: `artifacts/multimodal_compose_open_test.json`

---

## Related experiments

| Experiment | Decomposes | Output |
|------------|------------|--------|
| **Free fusion (this)** | TTS + LLM inputs | Emergence + new model |
| compose_fate | Text spans + TTS bands (preset shards) | Per-question runtime recompose |
| model_fate | Qwen layer quads | New model (weight decompose) |

---

## Next steps

1. Aggregate 107 free-fusion patches → rebake (not corpus-only incubation)
2. MCQ: structure → option affinity / lm_head
3. Mismatch ablation publish in artifacts

---

## Reproduce

Requires Valhalla monorepo with `hub-f64` built.

```bash
python3 scripts/run_natural_multimodal_potential.py --phase test --bake
valhalla_fused_lm --model checkpoints/valhalla_free_multimodal/test_fused_s0.08 --phase test
```

See `scripts/reproduce.sh`.
