//! Free multimodal potential experiment — LLM + TTS, Valhalla bodies run without preset routing.
//!
//! stdin JSON:
//! ```json
//! {
//!   "optimize_cycles": 2,
//!   "samples": [{"id":"p1","text":"...","audio_features":[...]}]
//! }
//! ```

use hub_f64::dl_front::{encode_modality, BodyFrontPorts, DlFrontConfig, ModalityInput};
use hub_f64::free_multimodal_fate::{
    run_free_multimodal_fate, FreeMultimodalOutcome, FreeMultimodalSample,
    FREE_MULTIMODAL_PROTOCOL,
};
use hub_f64::{BodyTarget, IngressBudget, TriadSession};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct ExperimentRequest {
    #[serde(default = "default_cycles")]
    optimize_cycles: usize,
    samples: Vec<FreeMultimodalSample>,
}

fn default_cycles() -> usize {
    2
}

#[derive(Debug, Serialize)]
struct SampleReport {
    id: String,
    text: String,
    outcome: FreeMultimodalOutcome,
    mismatch_binding: f64,
}

#[derive(Debug, Serialize)]
struct PotentialSummary {
    protocol: String,
    sample_count: usize,
    emergence_rate: f64,
    avg_natural_multimodal_score: f64,
    avg_adaptation_rounds: f64,
    avg_hub_modality_separation: f64,
    tile_superadditive_rate: f64,
    stem_superadditive_rate: f64,
    potential_confirmed: bool,
    verdict: String,
}

#[derive(Debug, Serialize)]
struct ExperimentResponse {
    samples: Vec<SampleReport>,
    summary: PotentialSummary,
}

fn ports_from_text(text: &str, front: &DlFrontConfig) -> BodyFrontPorts {
    encode_modality(
        &ModalityInput::Text {
            content: text.to_string(),
        },
        front,
    )
}

fn ports_from_audio(vals: &[f64], front: &DlFrontConfig) -> BodyFrontPorts {
    encode_modality(
        &ModalityInput::AudioFeatures {
            values: vals.to_vec(),
        },
        front,
    )
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-12 {
        0.0
    } else {
        dot / d
    }
}

fn run_mismatch_fusion(
    budget: &IngressBudget,
    text: &str,
    wrong_audio: &[f64],
) -> Vec<f64> {
    let front = DlFrontConfig::default();
    let llm = ports_from_text(text, &front);
    let tts = ports_from_audio(wrong_audio, &front);
    let mut session = TriadSession::with_body(budget.clone(), BodyTarget::Triad);
    let out = run_free_multimodal_fate(&mut session, &llm, &tts, "mismatch", None);
    out.report.model_patch.patch_vector
}

fn main() {
    let raw = std::io::read_to_string(std::io::stdin()).unwrap_or_default();
    let req: ExperimentRequest = serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("invalid JSON stdin: {e}");
        std::process::exit(1);
    });

    if req.samples.is_empty() {
        eprintln!("need at least 1 sample");
        std::process::exit(1);
    }

    let mut budget = IngressBudget::incubation();
    budget.max_items = 256;
    let front = DlFrontConfig::default();
    let n = req.samples.len();

    let mut reports = Vec::new();
    let mut emergence_count = 0usize;
    let mut natural_scores = Vec::new();
    let mut adapt_rounds = Vec::new();
    let mut hub_seps = Vec::new();
    let mut tile_super = 0usize;
    let mut stem_super = 0usize;

    for (i, sample) in req.samples.iter().enumerate() {
        let llm = ports_from_text(&sample.text, &front);
        let tts = ports_from_audio(&sample.audio_features, &front);

        let mismatch_j = (i + n / 2) % n;
        let mismatch_patch = if n >= 2 {
            Some(run_mismatch_fusion(
                &budget,
                &sample.text,
                &req.samples[mismatch_j].audio_features,
            ))
        } else {
            None
        };

        let mut session = TriadSession::with_body(budget.clone(), BodyTarget::Triad);
        let mut outcome = run_free_multimodal_fate(
            &mut session,
            &llm,
            &tts,
            "free_multimodal_fate",
            mismatch_patch.as_deref(),
        );

        let matched_patch = outcome.report.model_patch.patch_vector.clone();
        let mismatch_binding = if let Some(ref mp) = mismatch_patch {
            cosine(&matched_patch, mp)
        } else {
            0.0
        };
        if mismatch_patch.is_some() {
            let binding = (cosine(&matched_patch, &matched_patch) - mismatch_binding).max(0.005);
            outcome.emergence.cross_modal_binding = binding;
            outcome.emergence.natural_multimodal_score = (outcome.emergence.natural_multimodal_score
                + binding * 0.2)
                .min(1.0);
            outcome.emergence.emergence_likely = outcome.emergence.emergence_likely
                && binding > 0.005;
        }

        let e = &outcome.emergence;
        if e.emergence_likely {
            emergence_count += 1;
        }
        if e.tile_superadditive {
            tile_super += 1;
        }
        if e.stem_superadditive {
            stem_super += 1;
        }
        natural_scores.push(e.natural_multimodal_score);
        adapt_rounds.push(outcome.adaptation_rounds as f64);
        hub_seps.push(e.hub_modality_separation);

        eprintln!(
            "  [{}] rounds={} score={:.3} super=t{}/s{} emerge={}",
            sample.id,
            outcome.adaptation_rounds,
            e.natural_multimodal_score,
            e.tile_superadditive,
            e.stem_superadditive,
            e.emergence_likely,
        );

        reports.push(SampleReport {
            id: sample.id.clone(),
            text: sample.text.clone(),
            outcome,
            mismatch_binding,
        });
    }

    let nf = n as f64;
    let emergence_rate = emergence_count as f64 / nf;
    let avg_natural = natural_scores.iter().sum::<f64>() / nf;
    let avg_rounds = adapt_rounds.iter().sum::<f64>() / nf;
    let avg_hub_sep = hub_seps.iter().sum::<f64>() / nf;
    let tile_rate = tile_super as f64 / nf;
    let stem_rate = stem_super as f64 / nf;

    let potential_confirmed = emergence_rate >= 0.35
        && avg_natural >= 0.30
        && tile_rate >= 0.40
        && avg_hub_sep >= 0.02;

    let verdict = if potential_confirmed {
        "POTENTIAL_CONFIRMED: TTS+LLM multi-round free fusion shows emergent cross-modal structure"
    } else if emergence_rate >= 0.15 || avg_natural >= 0.20 {
        "PARTIAL_SIGNAL: some cross-modal emergence; extend corpus / rounds / TTS quality"
    } else {
        "NOT_YET: no reliable natural multimodal emergence under current settings"
    };

    let resp = ExperimentResponse {
        samples: reports,
        summary: PotentialSummary {
            protocol: FREE_MULTIMODAL_PROTOCOL.to_string(),
            sample_count: n,
            emergence_rate,
            avg_natural_multimodal_score: avg_natural,
            avg_adaptation_rounds: avg_rounds,
            avg_hub_modality_separation: avg_hub_sep,
            tile_superadditive_rate: tile_rate,
            stem_superadditive_rate: stem_rate,
            potential_confirmed,
            verdict: verdict.to_string(),
        },
    };

    println!("{}", serde_json::to_string_pretty(&resp).unwrap_or_default());
    if !potential_confirmed {
        std::process::exit(2);
    }
}
