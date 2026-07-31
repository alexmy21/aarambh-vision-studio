# ROADMAP_VISION_STUDIO.md — Part 1 of 2 — aarambh-vision-studio

> Final v1 step-by-step build plan, revised to match the depth of
> `ROADMAP_V4.md` — tasks are grouped by crate with the reasoning behind
> each one inline, tests are real function stubs, and every phase that
> touches training data has an explicit Data Setup step. This revision
> also adds two phases (Reference-Image Prompting, Structural
> Conditioning) that the first draft's architecture didn't cover, and
> rewrites Phases 1, 4, 7, 10, 13, and 24 to close the seven gaps raised
> after the first draft. Read alongside `ARCHITECTURE_VISION_STUDIO_
> PART1/2.md` and `SELF_LEARNING_VISION_STUDIO.md`.
>
> Source/engineering release: no pretrained checkpoints, style adapters,
> or subject adapters are released as part of this roadmap.

---

## How to Read This Roadmap

Each phase has:
- **Goal** — exactly what you will have when this phase is done
- **Tasks** — the checklist to follow, in order, grouped by crate
- **Tests** — what you write to prove it works
- **Milestone** — how you know you are done, with the git tag to cut

---

## Phase Map (Quick Reference — Final v1, 28 Phases)

```
Phase 0  →  Workspace + core types                          (1–2 days)    [i3]
Phase 1  →  Image VAE (tokenizer), staged w/ fallback gates  (16–20 days) [i3+Kaggle] ⚠
Phase 2  →  Text prep + prompt tokenizer                     (3–5 days)   [i3]
Phase 3  →  Data pipeline + resolution-bucketing scaffold     (7–10 days)  [i3+Kaggle]
Phase 4  →  Image understanding — staged fine-tune schedule   (9–12 days)  [Kaggle]
Phase 5  →  NN primitives — MMDiT, AdaLN-Zero, bucket-2D-RoPE (8–11 days)  [i3]
Phase 6  →  CPU SIMD kernels + CUDA prep                       (5–7 days)   [i3+Kaggle prep]
Phase 7  →  Text encoder integration + hidden-layer sweep      (4–6 days)   [i3+Kaggle]
Phase 8  →  Text-to-image baseline — Tiny trains!              (14–21 days) [Kaggle] ⚠ heaviest
Phase 9  →  Sampler (ODE/CFG) + inference engine + CLI          (5–7 days)   [i3]
Phase 10 →  Multi-resolution / aspect-ratio bucket training      (7–10 days)  [Kaggle]
Phase 11 →  Scale-up to Small/Medium                              (10–14 days) [Kaggle]
Phase 12 →  Reference-image prompting (IP-Adapter-equivalent)     (10–14 days) [Kaggle] — NEW
Phase 13 →  Image editing — self-bootstrapped reference conditioning (14–18 days) [Kaggle] ⚠
Phase 14 →  Masked / inpainting editing extension                  (5–7 days)   [Kaggle]
Phase 15 →  Structural conditioning — edge maps (ControlNet-equiv)  (8–11 days)  [Kaggle] — NEW
Phase 16 →  Super-resolution / upscale refinement                   (7–10 days)  [Kaggle]

── continues in ROADMAP_VISION_STUDIO_PART2.md ──

Phase 17 →  Full control layer (DrishtiRequest)                     (5–7 days)   [i3]
Phase 18 →  Safety, content filtering, watermarking                 (7–10 days)  [i3]
Phase 19 →  Quantisation stack                                       (7–10 days)  [i3+Kaggle]
Phase 20 →  Fine-tuning (LoRA/QLoRA/DoRA)                             (7–10 days)  [Kaggle]
Phase 21 →  Alignment — GRPO + DPO                                     (10–14 days) [Kaggle]
Phase 22 →  Self-learning                                               (7–10 days)  [i3]
Phase 23 →  Evaluation harness + named baselines (Sana, FLUX.2 klein)   (8–11 days)  [i3+Kaggle]
Phase 24 →  Few-step distillation — reflow + consistency + adversarial  (12–16 days) [Kaggle] ⚠
Phase 25 →  GPU scale-up — Large model                                    (7–10 days)  [Kaggle]
Phase 26 →  Inference server + output formats                             (7–10 days)  [i3]
Phase 27 →  Production release v1.0                                         (7–10 days)  [all]
```

**Total realistic estimate: 217–303 days (~7.2–10.1 months) part-time** —
grew from the first draft's 187–264 days because two real production
features (Phases 12 and 15) were missing entirely and several phases
(1, 4) now carry explicit staging that the first draft glossed over.

---

## Why This Order

1. **0–1** — workspace, then the Image VAE, staged with explicit go/no-go
   gates (§6.5, ARCHITECTURE Part 1) instead of a single training recipe
   that either works or silently doesn't.
2. **2–4** — text prep, data pipeline, understanding. Phase 4's captioner
   must exist before Phase 8's generation training (it auto-labels the
   training pool), and now has an explicit freeze/unfreeze schedule
   instead of an unspecified "fine-tune further."
3. **5–7** — MMDiT-specific primitives, then kernels, then the text
   encoder — deliberately split out as its own small phase now, because
   "load `aarambh-ai`'s checkpoint and sweep 2–3 candidate hidden layers"
   is a discrete, testable unit of work in its own right, not a detail
   buried inside Phase 8.
4. **8–9** — baseline generation, then sampler + CLI. Unchanged in
   position from the first draft — still the highest-risk phase, still
   where the architecture gets proven end-to-end.
5. **10–11** — multi-resolution (now with a concrete bucket table,
   §8.5 ARCHITECTURE Part 1), then scale-up. Unchanged in position.
6. **12 (reference-image prompting) before 13 (editing)** — deliberately
   sequenced this way because both features share the same underlying
   idea (encode a reference image, feed it into the MMDiT as
   conditioning), and reference-prompting is the simpler version of that
   idea (a single cross-attention branch, no loss-masking, no synthetic
   data pipeline). Building the simpler version first means Phase 13
   reuses proven reference-encoding plumbing instead of inventing it
   under the pressure of also solving instruction-editing's data problem
   at the same time.
7. **13–14 (editing, then masked editing)** — unchanged reasoning from
   the first draft: editing depends on a working generation backbone,
   and masked editing is split out after global editing works.
8. **15 (structural conditioning) after editing** — placed here rather
   than earlier because the zero-convolution conditioning-branch pattern
   (§12.2, ARCHITECTURE Part 2) is easiest to validate against a model
   that already handles other forms of conditioning correctly (reference
   images, masks) — if something breaks, it's easier to isolate "is this
   the new branch or an existing one" with more of the system proven.
9. **16 (upscale)** — deliberately last in this block; an optional,
   additive pass over whatever the base model already produces.
10. **17–22 (control layer, safety, quant, fine-tune, alignment,
    self-learning)** — same ordering logic as the first draft: alignment
    (21) comes after fine-tuning (20), self-learning (22) comes last in
    this block.
11. **23 (evaluation + named baselines)** — now explicitly scores against
    Sana and FLUX.2 [klein] 4B (§22.2, ARCHITECTURE Part 2), not a vague
    "some baseline."
12. **24 (distillation) after evaluation, not before** — this is a
    deliberate change from the first draft's ordering. Distillation's
    three stages (reflow → consistency → adversarial correction, §14
    ARCHITECTURE Part 2) need the eval harness's metrics to measure the
    1/2/4-step quality tradeoff (§14.3) — running distillation before the
    harness exists would mean judging it by eye only.
13. **25–27 (GPU scale-up, server + formats, release)** — unchanged
    reasoning from the first draft.

---

## Workspace `Cargo.toml`

```toml
[workspace]
members = [
    "crates/aarambh-vision-core",
    "crates/aarambh-vision-tokenizer",
    "crates/aarambh-vision-textprep",
    "crates/aarambh-vision-data",
    "crates/aarambh-vision-understand",
    "crates/aarambh-vision-nn",
    "crates/aarambh-vision-kernel",
    "crates/aarambh-vision-textencoder",
    "crates/aarambh-vision-model",
    "crates/aarambh-vision-weights",
    "crates/aarambh-vision-train",
    "crates/aarambh-vision-quant",
    "crates/aarambh-vision-finetune",
    "crates/aarambh-vision-align",
    "crates/aarambh-vision-selflearn",
    "crates/aarambh-vision-edit",
    "crates/aarambh-vision-structure",
    "crates/aarambh-vision-refprompt",
    "crates/aarambh-vision-upscale",
    "crates/aarambh-vision-safety",
    "crates/aarambh-vision-eval",
    "crates/aarambh-vision-control",
    "crates/aarambh-vision-inference",
    "crates/aarambh-vision-serve",
    "aarambh-vision-studio",
]
resolver = "2"

[workspace.dependencies]
candle-core         = { version = "0.11" }
candle-nn           = { version = "0.11" }
candle-transformers = { version = "0.11" }
image               = { version = "0.25", features = ["png", "jpeg", "bmp", "tiff"] }
webp                = "0.3"
fast_image_resize   = "5"
imageproc           = "0.25"
tokenizers          = "0.22"
anyhow              = "1"
thiserror           = "2"
serde               = { version = "1", features = ["derive"] }
serde_json          = "1"
toml                = "0.9"
tokio               = { version = "1", features = ["full"] }
clap                = { version = "4", features = ["derive"] }
tracing             = "0.1"
tracing-subscriber  = "0.3"
safetensors         = "0.7"
rayon               = "1.7"
cc                  = "1"
which               = "6"
criterion           = "0.8"
sha2                = "0.10"
axum                = "0.8"
```

---

## Phase 0 — Workspace + Core Types

**Duration:** 1–2 days | **Hardware:** i3

### Goal
A compilable Cargo workspace, zero warnings, `aarambh-vision-core` 100%
complete, all other crates scaffolded (25 crates total).

### Tasks

**Workspace setup:**
```
[ ] Root Cargo.toml — 24 lib crates + 1 bin, workspace.dependencies
[ ] cargo new --lib for all 24 library crates (see full list above)
[ ] cargo new --bin aarambh-vision-studio
```

**`aarambh-vision-core`:**
```
[ ] src/config.rs
      ModelConfig { d_model, n_dual_layers, n_single_layers, n_heads,
                    latent_channels, patch_size }
      impl ModelConfig { fn tiny()/small()/medium()/large() -> Self }
      Rationale: one config type shared by every crate that needs to
      know the active scale, so a scale change never means touching
      more than one file.
[ ] src/request.rs
      DrishtiRequest (full fields per ARCHITECTURE Part 2 §23), stubbed
      here, fully wired by Phase 17.
[ ] src/error.rs
      AarambhVisionError via thiserror, one variant per crate's failure
      mode — added to incrementally as each crate is built, not written
      exhaustively up front.
```

### Tests
```rust
#[test]
fn tiny_config_has_expected_dims() {
    let cfg = ModelConfig::tiny();
    assert_eq!(cfg.d_model, 384);
    assert_eq!(cfg.n_dual_layers, 4);
    assert_eq!(cfg.n_single_layers, 8);
}

#[test]
fn config_roundtrips_through_json() {
    // serde_json::to_string then from_str reproduces the original struct exactly
}
```

### Additional
```
[ ] .gitignore, .github/ (CI, release workflow, issue/PR templates)
[ ] CHANGELOG.md
```

### Milestone
`cargo check --workspace` passes, zero warnings, all 25 crates present.
Tag: `v0.1.0-phase0`

---

## Phase 1 — Image VAE (Tokenizer), Staged With Fallback Gates

**Duration:** 16–20 days | **Hardware:** i3 + Kaggle | ⚠ highest-risk phase

### Goal
A frozen, trustworthy image ⇄ latent tokenizer, built through the
explicit four-stage fallback path in ARCHITECTURE Part 1 §6.5 — not a
single training recipe with an implicit hope that adversarial training
converges.

### Tasks

**`aarambh-vision-tokenizer` — shared architecture (build once, used by every stage below):**
```
[ ] src/encoder.rs — strided conv encoder, ×8 downsample, μ/log-σ² head
[ ] src/decoder.rs — transposed conv decoder, mirror of encoder
[ ] src/reparam.rs — z = μ + σ·ε sampling
[ ] src/patchify.rs — 2×2 patch → token sequence, and inverse
[ ] src/discriminator.rs — patch-GAN (70×70 patches), hinge loss
[ ] src/losses.rs — L1 + LPIPS reconstruction, KL, adversarial (§6.2)
```

**Stage 0a — Reconstruction-only (i3, `aarambh-vision-train`):**
```
[ ] Training loop: L = L1 + LPIPS only, no KL, no adversarial
[ ] Run on a small curated 64×64 image sample
[ ] Rationale: isolate encoder/decoder/patchify correctness from
    adversarial-balance risk entirely — if this doesn't converge, the
    bug is architectural, not a GAN stability problem
[ ] Go/no-go: LPIPS below a fixed threshold within a bounded step budget
```

**Stage 0b — Add KL (i3):**
```
[ ] Enable λ_kl · L_kl on top of Stage 0a's converged checkpoint
[ ] Go/no-go: LPIPS regression from Stage 0a stays within an agreed
    tolerance band; latent distribution's mean/variance approach N(0,1)
    on a held-out batch (checked directly, not assumed)
```

**Stage 0c — Add adversarial term (i3, then Kaggle if unstable on CPU):**
```
[ ] Enable discriminator + λ_adv · L_adversarial on top of Stage 0b
[ ] Stabilisation tricks available if training diverges: lower
    discriminator LR, R1 gradient penalty, delayed discriminator start
[ ] Go/no-go: if stable within a bounded number of retries, proceed with
    the full L_vae checkpoint. If NOT stable after retries: fall back to
    shipping the Stage 0b checkpoint as the frozen VAE — log this
    decision explicitly in the phase's milestone notes, don't silently
    retry indefinitely
```

**Stage 1 — Scale to full resolution range (Kaggle):**
```
[ ] Retrain whichever of 0b/0c passed its gate, on the full curated
    image set, up to 512×512
[ ] Freeze checkpoint
```

### Tests
```rust
#[test]
fn encoder_output_shape_matches_downsample_factor() {
    // For several (H, W), assert encoder output is (H/8, W/8, latent_channels)
}

#[test]
fn kl_matches_hand_computed_value() {
    // μ=0.4, σ²=0.81 → KL ≈ 0.0904 (§6.2 worked example 1)
    // μ=0.02, σ²=1.02 → KL ≈ 0.0003 (§6.2 worked example 2)
}

#[test]
fn patchify_unpatchify_roundtrips_exactly() {
    // Random tensor → patchify → unpatchify reproduces the input exactly
}

#[test]
fn stage_0b_latent_distribution_approaches_unit_gaussian() {
    // Sample latents on a held-out batch; assert mean/var within tolerance of (0, 1)
}

#[test]
fn frozen_checkpoint_reconstruction_stable_across_reruns() {
    // Same input, same frozen checkpoint, 3 reruns → LPIPS variance near zero
}
```

### Milestone
A CLI command encodes a real image to latent and decodes it back
recognisably. The phase's milestone notes explicitly record which stage
(0b or 0c) the frozen checkpoint came from and why. Tag: `v0.1.0-phase1`

---

## Phase 2 — Text Prep + Prompt Tokenizer

**Duration:** 3–5 days | **Hardware:** i3

### Goal
Prompt text reliably becomes the token IDs `aarambh-ai`'s decoder-only
checkpoint expects (Phase 7 loads the checkpoint itself; this phase only
prepares text going into it).

### Tasks

**`aarambh-vision-textprep`:**
```
[ ] src/tokenize.rs — wraps tokenizers crate, loads aarambh-ai's EXISTING
    trained BPE vocabulary directly (no new tokenizer training)
[ ] src/normalize.rs — casing, whitespace, basic prompt-syntax handling
[ ] src/null_condition.rs — fixed, reproducible null/empty token sequence
    for CFG training (Part 1 §8.3)
```

### Tests
```rust
#[test]
fn roundtrip_preserves_prompt_content() {
    // text -> tokens -> text preserves a fixed sample prompt set
}

#[test]
fn null_conditioning_is_deterministic() {
    // Calling null_condition() twice produces byte-identical token sequences
}
```

### Milestone
`cargo test -p aarambh-vision-textprep` passes. Tag: `v0.1.0-phase2`

---

## Phase 3 — Data Pipeline + Resolution-Bucketing Scaffold

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
A dataset loader serving (image, caption) pairs at scale, with the
resolution-bucketing scaffold (§8.5, ARCHITECTURE Part 1) in place from
the start, even though it isn't exercised until Phase 10.

### Tasks

**`aarambh-vision-data`:**
```
[ ] src/loader.rs — batched image+caption loading, resizing via
    fast_image_resize
[ ] src/bucket.rs — the 5-bucket table (square/landscape/wide/portrait/
    tall, §8.5), assignment by closest log-aspect-ratio, center-crop
    (never stretch) to the assigned bucket's resolution, drop images
    too small for their bucket
[ ] src/augment.rs — crop/flip/colour-jitter, applied per-bucket so
    augmentation never changes an image's assigned bucket
[ ] src/caption_bootstrap.rs — routes uncaptioned images to Phase 4's
    captioner once it exists (stub interface defined now)
[ ] Ingest a permissively-licensed curated image-caption sample set
```

### Tests
```rust
#[test]
fn bucket_assignment_matches_closest_aspect_ratio() {
    // A fixed set of (width, height) pairs each map to the expected bucket
}

#[test]
fn small_images_are_dropped_not_upscaled() {
    // An image smaller than its assigned bucket's resolution is excluded
    // from the batch, never upsampled
}

#[test]
fn augmentation_is_deterministic_under_fixed_seed() {}
```

### Milestone
A training-shaped, bucket-tagged batch can be pulled from the pipeline
end-to-end. Tag: `v0.1.0-phase3`

---

## Phase 4 — Image Understanding: CLIP-Style Encoder + Captioner

**Duration:** 9–12 days | **Hardware:** Kaggle

### Goal
A working contrastive image-text encoder and captioning head, built
through the explicit four-step freeze/unfreeze schedule in ARCHITECTURE
Part 1 §9.5 — not a single unspecified "fine-tune further" instruction.

### Tasks

**`aarambh-vision-understand`:**
```
[ ] src/vision_encoder.rs — ViT-style patch encoder
[ ] src/text_projection.rs — contrastive text-side projection head
[ ] src/caption_decoder.rs — small autoregressive captioning transformer,
    cross-attending into the vision encoder's tokens
[ ] src/losses.rs — InfoNCE contrastive (§9.2) + captioning cross-entropy
```

**Step 1 (fixed step budget A) — heads only:**
```
[ ] Load aarambh-ai's frozen CLIP-B/32 weights into vision_encoder
[ ] Freeze the entire vision encoder
[ ] Train ONLY text_projection + caption_decoder (randomly initialised)
    against the frozen encoder
[ ] Rationale: the loaded encoder already has a useful embedding space;
    new heads just need to learn to read it before anything else changes
```

**Step 2 (fixed step budget B) — partial unfreeze:**
```
[ ] Unfreeze the top N transformer blocks of vision_encoder (a config
    constant, tuned during this phase — start at N=2 for CLIP-B/32's
    depth and adjust based on Step 2's own validation curve)
[ ] Continue contrastive-only training at a reduced learning rate
[ ] Rationale: lets the encoder adapt to this project's image
    distribution without catastrophically forgetting aarambh-ai's CLIP-B/32
```

**Step 3 (fixed step budget C) — enable captioning:**
```
[ ] Turn on λ_cap · L_caption on top of Step 2's checkpoint
[ ] Train jointly (contrastive + captioning) for the remainder of this phase
[ ] Rationale: captioning needs a stable embedding space to attend into;
    enabling it earlier wastes steps on a moving target
```

**Step 4 — freeze and bootstrap:**
```
[ ] Freeze the final checkpoint
[ ] Run the captioner once over the full uncaptioned image pool from
    Phase 3, feeding results back into aarambh-vision-data's caption_bootstrap
```

### Tests
```rust
#[test]
fn contrastive_loss_decreases_over_step_budget_a() {}

#[test]
fn step_2_unfreeze_does_not_regress_step_1_clip_score_beyond_tolerance() {
    // Confirms partial unfreeze adapts without catastrophic forgetting
}

#[test]
fn captioner_produces_non_degenerate_captions() {
    // No repeated-token degeneracy on a fixed held-out image set
}
```

### Milestone
The captioner produces a plausible caption for a held-out image, and
auto-captioned output is fed back into Phase 3's pipeline. The phase's
milestone notes record the actual step counts used for budgets A/B/C and
the chosen N for Step 2. Tag: `v0.1.0-phase4`

---

## Phase 5 — NN Primitives: MMDiT, AdaLN-Zero, Bucket-Aware 2D-RoPE

**Duration:** 8–11 days | **Hardware:** i3

### Goal
The MMDiT block, unit-tested in isolation, including the bucket-aware
2D-RoPE and NTK-style extrapolation reused from `aarambh-ai`'s long-
context work (§8.5, ARCHITECTURE Part 1).

### Tasks

**`aarambh-vision-nn`:**
```
[ ] src/patchify.rs — shared patch-token framing with the tokenizer
[ ] src/rope2d.rs — 2D rotary embeddings over row/column coordinates
[ ] src/rope2d_ntk.rs — NTK-style frequency rescaling for grid sizes
    beyond Tiny-scale training range, ported from aarambh-ai's YaRN/NTK
    long-context extrapolation, extended to two dimensions
[ ] src/adaln_zero.rs — conditioning MLP → scale/shift/gate, zero-init gates
[ ] src/dual_stream.rs — separate image/text MLP, joint attention
[ ] src/single_stream.rs — concatenated-sequence standard transformer block
[ ] src/mmdit.rs — full backbone assembly per ModelConfig scale
```

### Tests
```rust
#[test]
fn adaln_zero_block_is_identity_at_init() {
    // Gate values are exactly zero at initialisation; block output == input
}

#[test]
fn rope2d_matches_hand_computed_rotation() {
    // A known Q/K pair rotated by rope2d matches manual computation
}

#[test]
fn rope2d_ntk_extrapolates_beyond_training_grid_size() {
    // A grid larger than any Tiny-training bucket still produces
    // well-conditioned (non-NaN, bounded-magnitude) attention scores
}

#[test]
fn dual_stream_block_preserves_separate_token_shapes() {}

#[test]
fn single_stream_block_reunifies_dual_stream_output() {}
```

### Milestone
A randomly-initialised Tiny-scale MMDiT runs a forward pass on dummy
image + text tokens at two different bucket grid sizes without shape
errors. Tag: `v0.1.0-phase5`

---

## Phase 6 — CPU SIMD Kernels + CUDA Prep

**Duration:** 5–7 days | **Hardware:** i3 + Kaggle prep

### Goal
Fused kernels for patchify/unpatchify, 2D-RoPE, and AdaLN-Zero
modulation (§16, ARCHITECTURE Part 2).

### Tasks

**`aarambh-vision-kernel`:**
```
[ ] src/simd_patchify.rs
[ ] src/simd_rope2d.rs
[ ] src/simd_adaln.rs
[ ] src/cuda_prep/ — feature-gated stubs, exercised once Kaggle GPU is available
[ ] Benchmark fused vs. naive candle-op versions with criterion
```

### Tests
```rust
#[test]
fn fused_patchify_matches_naive_within_f32_tolerance() {}

#[test]
fn fused_rope2d_matches_naive_within_f32_tolerance() {}
```

### Milestone
Criterion benchmark shows fused kernels meaningfully faster than the
naive path on i3 CPU. Tag: `v0.1.0-phase6`

---

## Phase 7 — Text Encoder Integration + Hidden-Layer Sweep

**Duration:** 4–6 days | **Hardware:** i3 + Kaggle | *(new, split out from Phase 8)*

### Goal
`aarambh-ai`'s decoder-only checkpoint loaded and wired as the text
encoder, with the intermediate hidden-layer choice settled by a small,
real experiment rather than a guess (§7.2, ARCHITECTURE Part 1).

### Tasks

**`aarambh-vision-textencoder`:**
```
[ ] src/load.rs — load aarambh-ai's existing decoder-only checkpoint via
    the shared weights format, no architecture changes
[ ] src/hidden_states.rs — expose per-token hidden states at a
    configurable hidden_layer_index, plus mean-pooling for the AdaLN-Zero
    path (§7.2)
[ ] src/freeze.rs — freeze/unfreeze toggle (frozen by default; Phase 8
    decides whether to unfreeze end-to-end once the MMDiT baseline is stable)
```

**Hidden-layer sweep (against Phase 5's Tiny MMDiT, using a throwaway
short training run — not the real Phase 8 baseline):**
```
[ ] Train 3 short, small-scale MMDiT runs, identical except for which
    text-encoder layer feeds conditioning (e.g. layer index 8, 16, and
    the final layer, for a checkpoint of aarambh-ai's given depth)
[ ] Score each on CLIP-score (once Phase 23's harness exists, this is
    re-validated retroactively; in the meantime, a fixed small validation
    prompt set + Phase 4's contrastive encoder gives an interim score)
[ ] Record the winning hidden_layer_index as the project default
```

### Tests
```rust
#[test]
fn hidden_states_shape_matches_expected_seq_len_and_dim() {}

#[test]
fn frozen_text_encoder_produces_identical_output_across_calls() {}
```

### Milestone
`hidden_layer_index` is set to a specific, experimentally-chosen value
(not the final layer by default, per §7.2's rationale), recorded in this
phase's milestone notes with the sweep's scores. Tag: `v0.1.0-phase7`

---

## Phase 8 — Text-to-Image Baseline (Tiny Trains!)

**Duration:** 14–21 days | **Hardware:** Kaggle | ⚠ heaviest phase in Part 1

### Goal
Tiny-scale MMDiT + Phase 7's text encoder + frozen VAE, trained
end-to-end with the rectified flow objective, producing recognisable
images from a text prompt.

### Tasks

**`aarambh-vision-model`:**
```
[ ] src/generation.rs — assembles VAE (frozen) + text encoder (frozen,
    per Phase 7) + MMDiT into one forward/training interface
```

**`aarambh-vision-train`:**
```
[ ] src/flow_matching.rs — L_flow loss (§8.2), CFG null-conditioning
    dropout at ~10% (§8.3)
[ ] src/loop.rs — sample (x0, x1, t), compute x_t, predict velocity, backprop
[ ] Stage 0: Tiny, single bucket (square, 64×64-equivalent), small curated set
[ ] Once Stage 0 converges: optionally unfreeze the text encoder
    end-to-end at a reduced learning rate, re-validate it doesn't regress
    the Phase 7 sweep's winning configuration
```

### Data Setup
```bash
# Small curated 64x64 image-caption set, single resolution bucket,
# produced by Phase 3's pipeline + Phase 4's auto-captioner
scripts/phase8_prepare_baseline_data.sh data/tiny_baseline
```

### Tests
```rust
#[test]
fn flow_loss_matches_worked_examples() {
    // x0=2.0, x1=-1.0, t=0.3 → L=0.25 (§8.2 worked example 1)
    // x0=2.0, x1=-1.0, t=0.8 → L=0.04 (§8.2 worked example 2)
}

#[test]
fn cfg_null_conditioning_applied_at_expected_rate() {
    // Over a large batch sample, ~10% of examples receive the null
    // embedding (statistical check, not an exact count)
}

#[test]
fn end_to_end_forward_pass_produces_correctly_shaped_image() {
    // prompt -> tokens -> text embedding -> MMDiT -> VAE decode
}
```

### Milestone
`cargo run --bin aarambh-vision-studio -- generate --prompt "a red apple on a table"`
produces a 64×64 image where a red, apple-like shape is visible.
Tag: `v0.1.0-phase8`

---

## Phase 9 — Sampler (ODE/CFG) + Inference Engine + CLI

**Duration:** 5–7 days | **Hardware:** i3

### Tasks

**`aarambh-vision-inference`:**
```
[ ] src/euler.rs
[ ] src/heun.rs
[ ] src/cfg.rs — v_cfg = v_uncond + guidance_scale · (v_cond − v_uncond)
[ ] src/sampler.rs — step-count/solver-agnostic sampling loop
```

**`aarambh-vision-studio` (bin):**
```
[ ] CLI: generate --prompt --steps --guidance-scale --solver --seed --output
```

### Tests
```rust
#[test]
fn same_seed_produces_byte_identical_output() {}

#[test]
fn cfg_formula_matches_hand_computed_value() {}

#[test]
fn heun_has_lower_error_than_euler_at_matched_step_count() {
    // On a fixed validation prompt, measured against a long-step reference
}
```

### Milestone
CLI generates a reproducible image with a chosen solver and guidance
scale. Tag: `v0.1.0-phase9`

---

## Phase 10 — Multi-Resolution / Aspect-Ratio Bucket Training

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
The model trains and samples correctly across all 5 buckets from Phase
3's bucket table (§8.5, ARCHITECTURE Part 1), not just the single square
bucket Phase 8 proved out.

### Tasks
```
[ ] Enable all 5 buckets in aarambh-vision-data's batching (bucketed —
    never mixed within one batch, per §8.5)
[ ] Retrain the Tiny checkpoint across the full bucket set
[ ] Verify rope2d_ntk (Phase 5) engages correctly for any bucket whose
    grid exceeds Phase 8's single-bucket training range
```

### Tests
```rust
#[test]
fn each_bucket_produces_correctly_shaped_output_at_inference() {}

#[test]
fn cross_bucket_batches_are_never_constructed() {
    // A batching invariant check, not a training-quality check
}
```

### Milestone
The same checkpoint generates coherent images at two different bucket
aspect ratios from the CLI. Tag: `v0.1.0-phase10`

---

## Phase 11 — Scale-Up to Small/Medium

**Duration:** 10–14 days | **Hardware:** Kaggle

### Tasks
```
[ ] Train Small scale on the full curated dataset, all buckets
[ ] Train Medium scale, gradient checkpointing enabled (§26 memory estimates)
[ ] Manual quality spot-check vs. Tiny on a fixed comparison prompt set
    (retroactive FID/CLIP-score comparison once Phase 23 exists)
```

### Tests
```rust
#[test]
fn medium_checkpoint_loads_and_infers_without_oom_on_target_gpu() {}
```

### Milestone
Medium-scale checkpoint visibly outperforms Tiny on prompt-following for
a fixed comparison prompt set. Tag: `v0.1.0-phase11`

---

## Phase 12 — Reference-Image Prompting (IP-Adapter-Equivalent)

**Duration:** 10–14 days | **Hardware:** Kaggle | *(new)*

### Goal
Zero-shot reference-image conditioning: a reference image influences
generation as an additional prompt, no per-subject training required
(§11, ARCHITECTURE Part 2).

### Tasks

**`aarambh-vision-refprompt`:**
```
[ ] src/projection.rs — small trainable network: Phase 4's understanding-
    encoder image embedding → K/V tokens matching text cross-attention's
    dimensionality
[ ] src/decoupled_attn.rs — a SECOND cross-attention operation per MMDiT
    block (separate from the existing text cross-attention), output
    summed with the text cross-attention's output, scaled by
    reference_strength
[ ] Freeze every existing MMDiT weight; train ONLY projection.rs +
    decoupled_attn.rs's new weights
```

### Data Setup
```bash
# Self-referential: reuses Phase 3/4's existing (image, caption) pairs
# directly — each image is its own reference target during training
scripts/phase12_prepare_refprompt_data.sh data/refprompt_selfref
```

### Tests
```rust
#[test]
fn frozen_mmdit_weights_are_unchanged_after_refprompt_training() {
    // Checksum comparison, pre/post training
}

#[test]
fn reference_strength_zero_reproduces_text_only_generation() {
    // A DrishtiRequest with reference_strength=0 matches Phase 11's
    // text-only output for the same seed/prompt
}

#[test]
fn reference_strength_one_measurably_shifts_output_toward_reference() {
    // CLIP-image-similarity to the reference increases vs. strength=0
}
```

### Milestone
CLI generates a new image "inspired by" a reference image plus a text
prompt, with no adapter-training step in between. Tag: `v0.1.0-phase12`

---

## Phase 13 — Image Editing: Self-Bootstrapped Reference Conditioning

**Duration:** 14–18 days | **Hardware:** Kaggle | ⚠ heavy phase

### Goal
Global instruction-based editing, using the self-bootstrapped
prompt-to-prompt data construction method (§10.3, ARCHITECTURE Part 2)
— not a dependency on an external editing dataset.

### Tasks

**`aarambh-vision-edit`:**
```
[ ] src/reference_condition.rs — reference latent tokens concatenated
    in-context with noisy target tokens (reuses Phase 12's reference-
    encoding plumbing where applicable)
[ ] src/loss_mask.rs — restricts L_flow to target-token positions only
[ ] src/prompt_to_prompt.rs — the self-bootstrap data engine:
      1. generate image A from caption A, cache its cross-attention maps
      2. generate image B from caption B using the SAME initial noise,
         injecting caption A's cross-attention maps at every token
         except the ones that differ between A and B
      3. derive the edit instruction from the caption diff (template for
         simple swaps; text-encoder-assisted phrasing for complex diffs)
[ ] src/data_synth.rs — orchestrates prompt_to_prompt.rs over a large
    caption-pair set, plus imageproc-based mask generation (feeds Phase 14)
```

### Data Setup
```bash
# Runs prompt-to-prompt generation against your OWN Phase 11 checkpoint —
# no external editing dataset or external text-to-image model involved
scripts/phase13_generate_edit_pairs.sh --model checkpoints/medium_v11 \
    --caption-pairs data/caption_diff_pairs.jsonl \
    --out data/synthetic_edit_pairs/
```

### Tests
```rust
#[test]
fn prompt_to_prompt_preserves_shared_tokens_attention_maps() {
    // Attention maps for non-differing tokens match between image A/B generation
}

#[test]
fn loss_mask_excludes_reference_token_positions_exactly() {
    // Index-level check, not a statistical one
}

#[test]
fn instruction_template_correctly_derives_simple_attribute_swaps() {
    // "red car" / "blue car" -> "change the car color to blue"
}
```

### Milestone
CLI edits a real photo per a text instruction ("make the sky orange")
and non-sky regions remain visually unchanged. Tag: `v0.1.0-phase13`

---

## Phase 14 — Masked / Inpainting Editing Extension

**Duration:** 5–7 days | **Hardware:** Kaggle

### Tasks
```
[ ] aarambh-vision-edit: src/mask_channel.rs — binary mask patchified
    alongside reference latent
[ ] Mask dataset generation via imageproc (§10.3 strategy 2, ARCHITECTURE Part 2)
[ ] Extend the Phase 13 checkpoint with masked training examples
```

### Tests
```rust
#[test]
fn masked_out_region_is_measurably_preserved() {
    // Pixel-diff below threshold on a held-out test image
}

#[test]
fn masked_in_region_changes_per_instruction() {}
```

### Milestone
CLI performs a masked edit and the unmasked region is pixel-close to the
source. Tag: `v0.1.0-phase14`

---

## Phase 15 — Structural Conditioning: Edge Maps (ControlNet-Equivalent)

**Duration:** 8–11 days | **Hardware:** Kaggle | *(new)*

### Goal
Precise layout control via edge-map conditioning (§12, ARCHITECTURE Part
2), scoped to edge maps only for v1 (depth/pose deferred — §30).

### Tasks

**`aarambh-vision-structure`:**
```
[ ] src/edge_extract.rs — classical Canny-equivalent edge detection via
    imageproc (deterministic, no training required to produce this data)
[ ] src/branch.rs — small conv/patch encoder for the edge map, matching
    image-token shape
[ ] src/zero_conv.rs — zero-initialised output projection, so the branch
    starts as an exact no-op (same principle as AdaLN-Zero's zero gates)
[ ] Wire the branch's output as a residual add into dual-stream blocks'
    image-token path
```

### Data Setup
```bash
# Fully self-supervised: extract edge maps from the existing image pool,
# no new data collection
scripts/phase15_extract_edge_maps.sh data/images/ data/edge_maps/
```

### Tests
```rust
#[test]
fn zero_conv_branch_output_is_exactly_zero_at_init() {
    // Confirms the branch is a true no-op before any training
}

#[test]
fn structural_conditioning_does_not_regress_unconditioned_generation() {
    // A DrishtiRequest with no structural_map produces output
    // statistically indistinguishable from pre-Phase-15 generation
}

#[test]
fn output_edges_correlate_with_input_edge_map() {
    // Re-extract edges from generated output, compare to the conditioning map
}
```

### Milestone
CLI generates an image whose composition follows a supplied edge map,
with the accompanying text prompt controlling content/style.
Tag: `v0.1.0-phase15`

---

## Phase 16 — Super-Resolution / Upscale Refinement

**Duration:** 7–10 days | **Hardware:** Kaggle

### Tasks

**`aarambh-vision-upscale`:**
```
[ ] src/refine_model.rs — small patch-based refinement transformer (§15.1, ARCHITECTURE Part 2)
[ ] src/losses.rs — reconstruction + adversarial (§15.2)
[ ] Train on downsampled-then-upsampled pairs derived from existing data
```

### Tests
```rust
#[test]
fn upscaled_output_has_higher_sharpness_than_naive_bicubic() {
    // Laplacian variance comparison on the same input
}
```

### Milestone
CLI `--upscale` flag visibly sharpens a base-resolution output.
Tag: `v0.1.0-phase16`

---

*(Continued in `ROADMAP_VISION_STUDIO_PART2.md` — Phase 17 onward.)*
