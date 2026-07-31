# ROADMAP_VISION_STUDIO.md — Part 2 of 2 — aarambh-vision-studio

> Continues directly from Part 1 Phase 16. Read Part 1 first, and read
> `ARCHITECTURE_VISION_STUDIO_PART2.md` alongside this part.

---

## Phase 17 — Full Control Layer (`DrishtiRequest`)

**Duration:** 5–7 days | **Hardware:** i3

### Goal
Every capability built so far (generate, reference-prompt, edit,
inpaint, structural-condition, upscale) is reachable through one typed
request struct.

### Tasks

**`aarambh-vision-control`:**
```
[ ] src/request.rs — full DrishtiRequest (§23, ARCHITECTURE Part 2),
    including reference_mode (Edit vs. Prompt — disambiguating Phase 12
    vs. Phase 13's use of reference_image), structural_map, both strength fields
[ ] src/route.rs — dispatches DrishtiMode + reference_mode to the
    correct engine combination
[ ] src/validate.rs — rejects malformed requests (e.g. Edit mode with no
    reference_image, or reference_mode=Prompt with a mask set) before
    any compute happens
```

### Tests
```rust
#[test]
fn every_mode_and_reference_mode_combination_routes_correctly() {}

#[test]
fn invalid_combinations_are_rejected_with_clear_errors_not_panics() {
    // e.g. Inpaint with no mask; Prompt reference_mode with a mask set
}
```

### Milestone
A single `DrishtiRequest` value drives every engine combination through
one entry point. Tag: `v0.1.0-phase17`

---

## Phase 18 — Safety, Content Filtering & Provenance Watermarking

**Duration:** 7–10 days | **Hardware:** i3

### Tasks

**`aarambh-vision-safety`:**
```
[ ] src/filter.rs — prompt + output classifier built on Phase 4's
    understanding encoder, configurable policy
[ ] src/watermark.rs — spread-spectrum invisible watermark, survives
    moderate JPEG/WebP recompression
[ ] src/provenance.rs — sha2 content hash + generation metadata record
```

### Tests
```rust
#[test]
fn watermark_survives_typical_quality_recompression() {
    // Mark, re-encode through Phase 26's format pipeline at typical
    // quality settings, confirm recoverability
}

#[test]
fn filter_flags_fixed_policy_violating_test_prompts() {}
```

### Milestone
Every CLI-generated image carries a recoverable watermark and a recorded
content hash. Tag: `v0.1.0-phase18`

---

## Phase 19 — Quantisation Stack

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Tasks

**`aarambh-vision-quant`:**
```
[ ] src/int8.rs — calibration-based post-training quantisation
[ ] src/int4.rs — GGUF-style, per-block scales
[ ] src/calibrate.rs — calibration pass over a representative prompt/image set
[ ] Explicitly exclude the VAE from the INT4 path (§17, ARCHITECTURE Part 2)
```

### Tests
```rust
#[test]
fn vae_is_excluded_from_int4_quantisation_at_compile_time() {}

#[test]
fn quantised_model_quality_degradation_stays_within_threshold() {
    // Measured via Phase 23's eval harness once it exists; a manual
    // spot-check gates this phase in the meantime
}
```

### Milestone
INT4 checkpoint runs inference on i3 CPU at acceptable speed with
comparable output quality to FP32. Tag: `v0.1.0-phase19`

---

## Phase 20 — Fine-Tuning (LoRA/QLoRA/DoRA)

**Duration:** 7–10 days | **Hardware:** Kaggle

### Tasks

**`aarambh-vision-finetune`:**
```
[ ] src/lora.rs / qlora.rs / dora.rs — adapters over MMDiT attention + MLP weights
[ ] src/style_adapter.rs — small-image-set style adaptation recipe
[ ] src/subject_adapter.rs — DreamBooth-style subject adaptation recipe
```

### Tests
```rust
#[test]
fn trained_style_adapter_shifts_output_vs_base_checkpoint() {}

#[test]
fn base_checkpoint_weights_unchanged_after_adapter_training() {
    // Checksum comparison
}
```

### Milestone
CLI generates in a custom style from a 5–10 image reference set, base
weights untouched. Tag: `v0.1.0-phase20`

---

## Phase 21 — Alignment: GRPO + DPO

**Duration:** 10–14 days | **Hardware:** Kaggle

### Tasks

**`aarambh-vision-align`:**
```
[ ] src/grpo.rs — group-relative policy optimisation over sampled outputs
[ ] src/dpo.rs — direct preference optimisation over paired candidates
[ ] src/rewards.rs — wires CLIP-score, aesthetic score, edit-consistency
    (from aarambh-vision-eval, scaffolded now, full in Phase 23)
```

### Tests
```rust
#[test]
fn reward_computation_matches_eval_crate_output() {}

#[test]
fn post_alignment_checkpoint_scores_higher_than_pre_alignment() {
    // On a held-out prompt set, measured across all three reward metrics
}
```

### Milestone
Aligned checkpoint shows measurable aesthetic/CLIP-score improvement
over the Phase 20 checkpoint. Tag: `v0.1.0-phase21`

---

## Phase 22 — Self-Learning

**Duration:** 7–10 days | **Hardware:** i3

### Tasks

**`aarambh-vision-selflearn`:** (full breakdown in `SELF_LEARNING_VISION_STUDIO.md`)
```
[ ] src/memory.rs — associative memory bank of subject/style embeddings
[ ] src/orthogonalize.rs — gradient orthogonalisation vs. existing
    memory + base weights
[ ] src/gate.rs — confidence-gating against aarambh-vision-eval before commit
```

### Tests
```rust
#[test]
fn regressing_update_is_rejected_by_the_gate() {
    // Synthetic bad update, controlled test
}

#[test]
fn learning_new_subject_does_not_degrade_prior_subject_quality() {
    // Anti-forgetting check across two committed memory entries
}
```

### Milestone
The system learns a new subject from 3–5 reference images at inference
time, and a previously-learned subject remains intact afterward.
Tag: `v0.1.0-phase22`

---

## Phase 23 — Evaluation Harness + Named External Baselines

**Duration:** 8–11 days | **Hardware:** i3 + Kaggle

### Goal
FID, CLIP-score, aesthetic score, and edit-consistency, all computable,
plus comparison against the two named external baselines from §22.2,
ARCHITECTURE Part 2 — not a vague "some open-weight checkpoint."

### Tasks

**`aarambh-vision-eval`:**
```
[ ] src/fid.rs
[ ] src/clip_score.rs — uses Phase 4's understanding encoder
[ ] src/aesthetic.rs — learned predictor over human-preference-labelled pairs
[ ] src/edit_consistency.rs
[ ] src/baseline.rs — downloads and runs inference-only against:
      - Sana (NVIDIA) — primary, peer-scale comparison
      - FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0) — aspirational
        stretch comparison
      Both used READ-ONLY for scoring — never as a training
      initialisation, distillation teacher, or weight-merge source
      (explicit boundary, §22.2 ARCHITECTURE Part 2)
```

### Tests
```rust
#[test]
fn each_metric_is_stable_across_repeated_runs_on_a_fixed_image_set() {}

#[test]
fn baseline_comparison_report_generates_for_a_fixed_prompt_set() {
    // Report includes current checkpoint, Sana, and FLUX.2 [klein] scores
}

#[test]
fn baseline_models_are_never_referenced_by_any_training_code_path() {
    // A workspace-wide check: -train, -finetune, -align, -selflearn never
    // import or load Sana/FLUX.2 weights
}
```

### Milestone
A single command produces a full metrics report (all four metrics) for
the current checkpoint vs. Sana vs. FLUX.2 [klein] 4B.
Tag: `v0.1.0-phase23`

---

## Phase 24 — Few-Step Distillation: Reflow + Consistency + Adversarial Correction

**Duration:** 12–16 days | **Hardware:** Kaggle | ⚠ heavy, three-stage phase

### Goal
A distilled checkpoint sampling in 1–4 steps, using the named three-stage
recipe from §14, ARCHITECTURE Part 2 — not an unspecified "distillation."

### Tasks

**`aarambh-vision-train`:**
```
[ ] src/distill/reflow.rs
      Generate many (noise, image) pairs from the Phase 21 (post-
      alignment) checkpoint via full multi-step sampling. Retrain the
      SAME architecture from scratch on these straightened pairs.
      One reflow round is the v1 target; a second round is a stretch
      goal if time/compute allow.
[ ] src/distill/consistency.rs
      On top of the reflowed model, train a student to map any point on
      a trajectory directly to its endpoint in one function evaluation,
      using the reflowed model as teacher for intermediate targets
      (self-consistency training, Latent-Consistency-Model style).
[ ] src/distill/adversarial_correct.rs
      Add a patch-GAN discriminator loss on the STUDENT's few-step
      output specifically — this is what corrects the two documented
      failure modes (blurry one-step, oversaturated few-step) rather
      than accepting them.
```

### Tests
```rust
#[test]
fn reflowed_model_trajectories_are_measurably_straighter() {
    // Curvature metric on a fixed validation (noise, image) pair set,
    // before vs. after reflow
}

#[test]
fn consistency_student_matches_teacher_endpoint_within_tolerance() {
    // For several intermediate trajectory points, one-step student
    // prediction vs. multi-step teacher endpoint
}

#[test]
fn adversarial_correction_reduces_blur_at_one_step() {
    // Sharpness metric (e.g. Laplacian variance) before/after Stage C,
    // at 1-step sampling
}

#[test]
fn adversarial_correction_reduces_oversaturation_at_few_step() {
    // Color-histogram-based saturation metric, before/after Stage C, at 4 steps
}
```

### Milestone
CLI `--solver distilled --steps 4` produces output within an agreed
quality tolerance of the full 20-step baseline, measured via Phase 23's
harness against the Phase 21 teacher AND against Sana/FLUX.2 [klein].
Tag: `v0.1.0-phase24`

---

## Phase 25 — GPU Scale-Up: Large Model

**Duration:** 7–10 days | **Hardware:** Kaggle

### Tasks
```
[ ] Train Large scale with gradient checkpointing (§26 memory estimates)
[ ] Re-run Phase 20 (fine-tuning), Phase 21 (alignment), Phase 24
    (distillation) at Large scale
```

### Tests
```rust
#[test]
fn large_checkpoint_loads_and_infers_without_oom_on_target_gpu() {}

#[test]
fn large_scale_beats_medium_on_all_four_eval_metrics() {}
```

### Milestone
Large-scale checkpoint is the new flagship, outperforming Medium on all
four eval metrics and narrowing the gap to Sana/FLUX.2 [klein] versus
Medium's gap. Tag: `v0.1.0-phase25`

---

## Phase 26 — Inference Server + Output Formats

**Duration:** 7–10 days | **Hardware:** i3

### Tasks

**`aarambh-vision-serve`:**
```
[ ] src/routes.rs — axum handlers for generate/reference-prompt/edit/
    inpaint/structural-condition/upscale
[ ] src/formats.rs — PNG (default), JPEG, WebP (feature-gated), latent
    safetensors export
[ ] Load-test with concurrent requests on i3
```

### Tests
```rust
#[test]
fn each_output_format_roundtrips_in_a_standard_image_library() {}

#[test]
fn concurrent_requests_do_not_leak_state() {
    // Fixed-seed reproducibility check under concurrency
}
```

### Milestone
Server responds to a generate request over HTTP and returns a valid
image in each of PNG/JPEG/WebP on request. Tag: `v0.1.0-phase26`

---

## Phase 27 — Production Release v1.0

**Duration:** 7–10 days | **Hardware:** all

### Tasks
```
[ ] Full README.md (donation info, no contact section, per convention)
[ ] CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
[ ] Final pass over ARCHITECTURE_VISION_STUDIO_PART1/2.md, this roadmap,
    and SELF_LEARNING_VISION_STUDIO.md for drift vs. actual shipped code
[ ] Tag v1.0.0
```

### Milestone
`v1.0.0` tagged, source-only, `publish = false` on every crate, README
complete. Tag: `v1.0.0`

---

## Post-v1.0 (explicitly not in this roadmap)

See §30 (ARCHITECTURE Part 2) — video generation, 3D/multi-view
consistency, depth-map/pose-skeleton structural conditioning, real-person
likeness cloning, live human-preference collection infrastructure, and
multi-GPU training are all deliberately deferred past v1.
