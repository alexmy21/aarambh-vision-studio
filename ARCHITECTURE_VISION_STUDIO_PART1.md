# ARCHITECTURE_VISION_STUDIO.md — Part 1 of 2 — aarambh-vision-studio

> Final v1 architecture, revised. This revision closes seven specific
> gaps identified after the first draft (VAE fallback path, text-encoder
> bootstrap specifics, Phase 4 fine-tuning schedule, editing dataset
> source, distillation method, external baseline model, multi-resolution
> bucketing) and adds two production features that were missing from the
> first draft entirely: structural/spatial conditioning (a ControlNet-
> equivalent) and reference-image prompting (an IP-Adapter-equivalent).
> Read this together with Part 2 and SELF_LEARNING_VISION_STUDIO.md
> before writing any code.

**Companion documents:**
- `ARCHITECTURE_VISION_STUDIO_PART2.md` — editing, structural
  conditioning, reference-image prompting, sampling/CFG, few-step
  distillation, kernels, quantisation, fine-tuning, alignment (GRPO/DPO),
  safety, eval + named external baselines, crate reference, data flow,
  memory estimates, hardware strategy, image output formats, relationship
  to `aarambh-studio` / `aarambh-voice-studio`, out-of-scope.
- `SELF_LEARNING_VISION_STUDIO.md` — the self-learning subsystem in full.
- `ROADMAP_VISION_STUDIO_PART1.md` / `PART2.md` — 27-phase step-by-step
  build plan, same Goal/Tasks/Tests/Milestone depth as `ROADMAP_V4.md`.

---

## Table of Contents (Part 1)

1. [Project Overview](#1-project-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Dependency Versions & Toolchain](#3-dependency-versions--toolchain)
4. [Complete Workspace — 25 Crates](#4-complete-workspace--25-crates)
5. [Model Scales](#5-model-scales)
6. [Image VAE (Tokenizer) — Training In Detail](#6-image-vae-tokenizer--training-in-detail)
7. [Text Encoder — Training In Detail](#7-text-encoder--training-in-detail)
8. [MMDiT Backbone (Text-to-Image) — Training In Detail](#8-mmdit-backbone-text-to-image--training-in-detail)
9. [Image Understanding / Captioning — Training In Detail](#9-image-understanding--captioning--training-in-detail)

*(Sections 10–29 continue in Part 2.)*

---

## 1. Project Overview

**aarambh-vision-studio** is a ground-up image generation, editing, and
understanding system written entirely in Rust, on `candle`. Every
tokenizer, transformer block, and training loop is implemented from
scratch — no bindings to PyTorch, no vendored checkpoints.

### What changed in this revision

| # | Gap raised | Fix, in one line | Where |
|---|---|---|---|
| 1 | VAE fallback path missing | Staged de-risking: reconstruction-only Stage 0a before adversarial/KL are added, with explicit go/no-go thresholds | §6.5 |
| 2 | Text encoder bootstrap underspecified | Reuse `aarambh-studio`'s **decoder-only** checkpoint directly (not a new bidirectional variant), extract hidden states from an intermediate layer, not the final one | §7 |
| 3 | Phase 4 fine-tuning unclear | Explicit freeze/unfreeze schedule, step counts, and curriculum order | §9.5 |
| 4 | Editing dataset source unclear | Self-bootstrapped synthetic pairs via prompt-to-prompt cross-attention injection on your own Phase 10 checkpoint, not a dependency on an external editing dataset | Part 2, §10.3 |
| 5 | Distillation too vague | Named recipe: reflow + adversarial correction (the same fix published work uses to solve one-step blur and few-step oversaturation) | Part 2, §12 |
| 6 | No external baseline named | Two named permissively-licensed models: **Sana** (peer-scale) and **FLUX.2 [klein] 4B**, Apache-2.0 (aspirational) | Part 2, §22 |
| 7 | Multi-resolution bucketing unclear | Concrete bucket table + assignment rule + how 2D-RoPE handles variable grids | §8.5 |

### What it is (v1, final — full build, no mini variant)

- **Image VAE (Tokenizer)** — image ⇄ continuous latent patches, trained
  once, frozen for every downstream stage.
- **Text-to-Image Engine** — an **MMDiT** trained with **Rectified Flow
  Matching**.
- **Reference-Image Prompting** (ControlNet/IP-Adapter family — the
  IP-Adapter half) — use a reference image's subject or style as an
  additional prompt input, zero-shot, no per-subject training required.
- **Structural/Spatial Conditioning** (the ControlNet half) — condition
  generation on an edge map, depth map, or pose skeleton for precise
  layout control.
- **Image Editing Engine** — instruction-based and masked/inpainting
  editing, on the same MMDiT backbone via in-context reference-image
  conditioning.
- **Image Understanding** — a CLIP-style contrastive encoder plus a
  captioning head, bootstrapped from `aarambh-studio`'s frozen CLIP-B/32.
- **Upscale / Refinement** — an optional super-resolution pass.
- **Few-Step Distillation** — a named, concrete recipe (Part 2 §12), not
  a placeholder.
- **Alignment** — GRPO + DPO, rewards sourced from your own eval harness.
- **Self-learning** — online subject/style adaptation with anti-forgetting
  (`SELF_LEARNING_VISION_STUDIO.md`).
- **Full Control Layer** — one typed `DrishtiRequest`.

### Why MMDiT + Rectified Flow, not U-Net + DDPM

| Old paradigm (SD 1.5/2.1) | Current paradigm (SD3/FLUX/Qwen-Image/Sana/HiDream) | Why it matters here |
|---|---|---|
| Convolutional U-Net | Transformer backbone on patch tokens (DiT/MMDiT) | Reuses your `aarambh-studio` attention/transformer-block code |
| Discrete-step DDPM, ~1000 steps | Continuous-time rectified flow, straight-line paths, few sampling steps | Simpler loss, simpler sampler, clean distillation path |
| CLIP-only conditioning | Large text encoder, deep token-level interaction | Your own decoder-only transformer is a legitimate text encoder (§7) |
| Editing = separate model | Editing = same backbone, in-context reference tokens | One model, one training loop |

### The subsystems, one foundation

```
                    ┌────────────────────────────┐
                    │     Full control layer       │  ← DrishtiRequest
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      Inference runtime         │
                    │  (sampler, CFG, mode routing)   │
                    └─┬─────┬──────┬──────┬──────┬──┘
                      │     │      │      │      │
                 ┌────┴─┐┌──┴───┐┌─┴────┐┌┴─────┐┌┴──────┐
                 │Text- ││Ref-  ││Edit /││Struct-││Upscale│
                 │to-   ││Image ││Inpaint││ural  ││refine │
                 │Image ││Prompt││       ││Cond.  ││       │
                 └────┬─┘└──┬───┘└─┬────┘└┬──────┘└┬──────┘
                      └─────┴──────┴──────┴─────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │   Shared foundation                │
                    │   VAE tokenizer + MMDiT core       │
                    │   + text encoder + understanding   │
                    │   + alignment (GRPO/DPO)           │
                    │   + self-learning                  │
                    └─────────────────────────────────────┘
```

---

## 2. Design Philosophy

| Goal | Decision |
|---|---|
| Reuse, don't rebuild | Transformer block ported from `aarambh-studio`; text encoder is `aarambh-studio`'s checkpoint used **as-is** (§7), not reimplemented in a new configuration |
| Understanding before generation | The captioning/CLIP-style encoder (§9) trains before the generator |
| One model, not a pipeline | Generation, editing, structural conditioning, and reference-prompting share one MMDiT backbone |
| Latest architecture, not legacy | MMDiT + rectified flow from day one |
| Nothing shipped as a placeholder | Every subsystem that was previously "TBD" (VAE risk, distillation, baseline) now has a named, concrete method |
| One control surface | Every knob is a typed field on `DrishtiRequest` |
| No opaque generation | Every image carries an invisible provenance watermark (Part 2 §20) |
| Source-first releases | `publish = false`; no bundled checkpoints |
| i3 + free Kaggle GPU only | No paid compute assumed anywhere |
| Reward-aligned | Every generative subsystem has a matching eval metric, and that metric becomes a GRPO/DPO reward |
| Learn after ship, safely | Self-learning updates are confidence-gated before commit |
| De-risk the riskiest part first, explicitly | The VAE — the single point of failure for everything downstream — gets a named fallback, not just a hope that Stage 1 works (§6.5) |

---

## 3. Dependency Versions & Toolchain

> Versions current as of mid-2026 — pin exact patch versions with
> `cargo add` at Phase 0 time.

```toml
[workspace.dependencies]
candle-core         = { version = "0.11" }
candle-nn           = { version = "0.11" }
candle-transformers = { version = "0.11" }

image               = { version = "0.25", features = ["png", "jpeg", "bmp", "tiff"] }
webp                = "0.3"
fast_image_resize   = "5"
imageproc           = "0.25"          # masks, drawing, edge/depth-map derivation for structural conditioning

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

> **VAE:** implemented from scratch in `aarambh-vision-tokenizer` — a
> continuous latent autoencoder (KL-regularised, not VQ). See §6.
>
> **Text encoder:** no new crate-level dependency — it is `aarambh-studio`'s
> own decoder-only checkpoint, loaded via the existing weights format.
> See §7.
>
> **Output encoders:** `image` stays in the default feature set; `webp`
> is feature-gated (`webp` cargo feature), mirroring `mp3lame-encoder`'s
> gating in `aarambh-voice-studio`. Full rationale in Part 2 §23.

---

## 4. Complete Workspace — 25 Crates

```
aarambh-vision-studio/
├── Cargo.toml
├── crates/
│   ├── aarambh-vision-core/            # L0 — config, request/response types, errors
│   ├── aarambh-vision-tokenizer/       # L1 — image VAE (+ fallback reconstruction-only mode)
│   ├── aarambh-vision-textprep/        # L1 — prompt tokenization + normalisation
│   ├── aarambh-vision-data/            # L1 — dataset loaders, auto-captioning, resolution bucketing
│   ├── aarambh-vision-understand/      # L1 — CLIP-style contrastive encoder + captioner
│   ├── aarambh-vision-nn/              # L2 — MMDiT block, AdaLN-Zero, 2D-RoPE (bucket-aware)
│   ├── aarambh-vision-kernel/          # L2 — CPU SIMD kernels, CUDA prep, fused patchify
│   ├── aarambh-vision-textencoder/     # L3 — thin wrapper loading aarambh-studio's decoder-only checkpoint
│   ├── aarambh-vision-model/           # L3 — full MMDiT assembly (gen + edit + structure + refprompt heads)
│   ├── aarambh-vision-weights/         # L3 — SafeTensors save/load, checkpoint conversion
│   ├── aarambh-vision-train/           # L4 — pretraining loops, rectified flow objective, distillation
│   ├── aarambh-vision-quant/           # L4 — INT8 / INT4 / GGUF-style quantisation
│   ├── aarambh-vision-finetune/        # L5 — LoRA / QLoRA / DoRA adapters
│   ├── aarambh-vision-align/           # L5 — GRPO + DPO alignment
│   ├── aarambh-vision-selflearn/       # L5 — online self-learning
│   ├── aarambh-vision-edit/            # L6 — reference-image conditioning, inpainting masks
│   ├── aarambh-vision-structure/       # L6 — NEW — structural/spatial conditioning (edge/depth/pose)
│   ├── aarambh-vision-refprompt/       # L6 — NEW — reference-image prompting (decoupled cross-attention)
│   ├── aarambh-vision-upscale/         # L7 — super-resolution refinement pass
│   ├── aarambh-vision-safety/          # L9 — content filtering, provenance watermarking
│   ├── aarambh-vision-eval/            # L9 — FID/CLIP-score/aesthetic/edit-consistency + named baselines
│   ├── aarambh-vision-control/         # L9 — DrishtiRequest DSL
│   ├── aarambh-vision-inference/       # L9 — sampler (ODE solver), CFG, shared runtime
│   └── aarambh-vision-serve/           # L10 — HTTP inference server
└── aarambh-vision-studio/              # L11 — bin (CLI)
```

24 library crates + 1 binary = **25 crates total** (was 23 — the two new
crates, `-structure` and `-refprompt`, are the two production features
this revision adds; see Part 2 §10.4–10.5).

---

## 5. Model Scales

| Scale | d_model | Dual-stream layers | Single-stream layers | n_heads | Params (MMDiT) | Native training resolution |
|---|---|---|---|---|---|---|
| Tiny   | 384  | 4  | 8  | 6  | ~40M  | 64×64   |
| Small  | 512  | 6  | 12 | 8  | ~120M | 128×128 |
| Medium | 768  | 8  | 16 | 12 | ~350M | 256×256 |
| Large  | 1024 | 12 | 24 | 16 | ~900M | 512×512 |

The **Image VAE is scale-independent** — one VAE, ~35–45M params, trained
once, frozen for all downstream engine training (§6).

```rust
impl ModelConfig {
    pub fn tiny() -> Self { /* d_model=384, dual=4, single=8, heads=6, ... */ }
    pub fn small() -> Self { /* d_model=512, dual=6, single=12, heads=8, ... */ }
    pub fn medium() -> Self { /* d_model=768, dual=8, single=16, heads=12, ... */ }
    pub fn large() -> Self { /* d_model=1024, dual=12, single=24, heads=16, ... */ }
}
```

---

## 6. Image VAE (Tokenizer) — Training In Detail

The single riskiest piece of infrastructure — everything downstream
depends on it, so it gets a fallback path, not just a training recipe.

### 6.1 Architecture

```
image (H×W×3, pixel space)
   │
   ▼
Conv encoder (strided downsample, ×8 total)
   │
   ▼
Latent distribution head → μ, log σ²   (continuous, KL-regularised)
   │
   ▼
Sample z = μ + σ·ε   (reparameterisation trick, ε ~ N(0,1))
   │
   ▼                      latent grid: (H/8)×(W/8)×C_latent, C_latent = 16
Patchify (2×2 patches) → token sequence for the MMDiT
   │
   ▼  (decoder path, mirror of encoder)
Conv decoder (transposed, ×8 upsample)
   │
   ▼
image (reconstructed, H×W×3)
```

Continuous latents (not discrete codebooks) avoid the codebook-collapse
issues that plagued VQ-VAE image tokenizers, and match what a rectified-
flow objective is defined over (§8.2).

### 6.2 Loss function (Stage 0 VAE training)

```
L_vae = λ_recon · L_reconstruction    (L1 pixel loss + LPIPS perceptual loss)
      + λ_adv   · L_adversarial       (patch-GAN discriminator, hinge loss)
      + λ_kl    · L_kl                (KL divergence: q(z|x) vs. N(0,1) prior)
```

Typical starting weights: `λ_recon=1.0, λ_adv=0.5, λ_kl=1e-6`.

**Worked numeric example — KL term for one latent channel:**
`μ=0.4, σ²=0.81`:
```
KL = 0.5 · (μ² + σ² − log(σ²) − 1)
   = 0.5 · (0.16 + 0.81 − (−0.2107) − 1) = 0.5 · 0.1807 = 0.0904
```
A latent closer to the prior, `μ=0.02, σ²=1.02`:
```
KL = 0.5 · (0.0004 + 1.02 − 0.0198 − 1) = 0.5 · 0.0006 = 0.0003
```
A latent already near N(0,1) contributes almost nothing — the network is
only penalised for latents that drift far from the prior.

### 6.3 Discriminator

One patch-GAN discriminator (70×70 patches, real-vs-reconstructed hinge
loss), same alternating one-step-per-side schedule as
`aarambh-voice-studio`'s codec discriminators.

### 6.4 Data

- Stage 0: a curated permissively-licensed image sample, 64×64, i3.
- Stage 1 (Kaggle T4/P100): full resolution range up to 512×512.
- Once frozen, never fine-tuned again in v1.

### 6.5 Fallback Path (De-Risking) — **new in this revision**

Training a continuous VAE with adversarial + KL loss simultaneously is
the step most likely to destabilise (discriminator/generator balance is
notoriously fragile at small scale on limited compute). Rather than
discovering this mid-Phase-1 with no plan, the phase is explicitly
staged with a go/no-go gate at each step:

```
Stage 0a — Reconstruction-only (no adversarial, no KL)
  L = L1 + LPIPS only.
  Goal: prove the encoder/decoder/patchify/unpatchify loop itself is
  correct — shapes, gradients, and a recognisable (if blurry) reconstruction.
  Go/no-go: LPIPS below a fixed threshold on a held-out set within a
  bounded number of training steps. If this alone doesn't converge, the
  bug is in the encoder/decoder architecture, not in adversarial balance
  — fix here before adding any more loss terms.

Stage 0b — Add KL only (still no adversarial)
  L = L1 + LPIPS + λ_kl · KL.
  Goal: confirm the latent space becomes well-behaved (roughly unit-
  Gaussian) without destabilising reconstruction quality from Stage 0a.
  Go/no-go: LPIPS regression from Stage 0a stays within an agreed
  tolerance band.

Stage 0c — Add the adversarial term (full L_vae, §6.2)
  Goal: sharpen reconstruction detail beyond what L1+LPIPS alone achieve.
  Go/no-go: if the discriminator destabilises training (loss diverges,
  or generator collapses) after a bounded number of retries with
  standard stabilisation tricks (lower discriminator LR, R1 gradient
  penalty, delayed discriminator start), **fall back to shipping the
  Stage 0b (non-adversarial) checkpoint** rather than blocking the whole
  roadmap on GAN stability. A non-adversarial VAE is measurably blurrier
  but fully functional — every downstream phase (generation, editing,
  upscaling) still works against it; only final-detail sharpness is
  reduced, and the optional upscale pass (Part 2 §11) partially
  compensates for exactly this.

Stage 1 — Scale to full resolution range (Kaggle), using whichever of
  0b/0c passed its go/no-go gate.
```

This turns "the VAE might not train well" from an unstated risk into an
explicit decision tree with a working fallback at every branch — nothing
downstream is blocked on adversarial training succeeding.

---

## 7. Text Encoder — Training In Detail

### 7.1 What changed here

The first draft proposed configuring your transformer stack as a new
**bidirectional** encoder for this project. Current production practice
(SD3, FLUX, Qwen-Image, Lumina-Next, SeFi-Image) has moved decisively
toward **decoder-only LLM text encoders** used directly — Lumina-Next
uses Gemma-2B, SeFi-Image uses Qwen3-VL, and multiple published
comparisons find decoder-only LLMs match or beat T5/CLIP on
compositional prompt understanding. That means the better move — and
the one that costs less work — is to **reuse `aarambh-studio`'s existing
decoder-only checkpoint exactly as it already is**, not build a new
bidirectional variant.

### 7.2 Architecture

```
prompt text
   │
   ▼
BPE tokenizer (aarambh-studio's existing trained tokenizer — reused directly)
   │
   ▼
aarambh-studio's decoder-only transformer, loaded via aarambh-vision-textencoder
(causal masking, exactly as trained — no architecture change)
   │
   ├── hidden states from an intermediate layer (not the final layer —
   │     published comparisons find final-layer-only embeddings
   │     underperform for this use case) → per-token conditioning,
   │     fed into MMDiT cross-stream attention (§8)
   │
   └── mean-pooled hidden state from the same layer → AdaLN-Zero
         conditioning (§8.2)
```

**Layer selection is a concrete, testable choice, not an afterthought:**
`aarambh-vision-textencoder` exposes a `hidden_layer_index` config field.
Phase 5's task list (ROADMAP Part 1) includes running a small sweep over
2–3 candidate layers on the Tiny baseline and picking the one with the
best CLIP-score on a fixed validation prompt set — this is a cheap,
one-time experiment, not a guess.

### 7.3 Why this is reuse, not new training

Because `aarambh-studio`'s checkpoint is used **as-is** — frozen at first,
with the option to unfreeze and fine-tune end-to-end once the MMDiT
baseline (Phase 7) is stable — this component needs **no dedicated
pretraining phase** in the roadmap at all. This is "reuse, don't rebuild"
(§2) taken as literally as possible: not "retrain a similar-shaped
encoder," but "load the exact checkpoint that already exists."

### 7.4 Data

No text-only data is consumed directly by this project — that
pretraining already happened as part of `aarambh-studio`. This component
only ever sees the (image, caption) pairs that Phase 3/4's pipeline
produces.

---

## 8. MMDiT Backbone (Text-to-Image) — Training In Detail

### 8.1 Architecture

```
Image latent tokens (from frozen VAE, §6, patchified)
Text tokens (from text encoder, §7 — intermediate-layer hidden states)
Timestep t (scalar, sinusoidal-embedded)
   │
   ▼
┌─────────────────────────────────────────────┐
│  Dual-stream blocks (×N_dual)                  │
│  image tokens and text tokens each keep their  │
│  own MLP + norm, share one joint attention op  │
└──────────────────────┬──────────────────────┘
                        ▼
┌─────────────────────────────────────────────┐
│  Single-stream blocks (×N_single)              │
│  image + text tokens concatenated into one     │
│  sequence, standard transformer blocks         │
└──────────────────────┬──────────────────────┘
                        ▼
              Predicted velocity field v_θ
                        │
                        ▼
              Unpatchify → latent-space update
```

**2D RoPE** gives the model a native notion of image spatial layout.
**AdaLN-Zero conditioning:** a small MLP takes the timestep embedding
(and the pooled text embedding, §7.2) and emits scale/shift/gate values
per block; gates initialise to zero so every block starts as an identity
function.

### 8.2 Loss function — Rectified Flow Matching

```
x_t = (1 − t)·x₀ + t·x₁,   t ∈ [0, 1]      (x₀ = data latent, x₁ = noise)
L_flow = E[ ‖ v_θ(x_t, t, text) − (x₁ − x₀) ‖² ]
```

**Worked numeric example 1:** `x₀=2.0, x₁=−1.0, t=0.3`:
```
x_t = 0.7·2.0 + 0.3·(−1.0) = 1.1
target v = x₁ − x₀ = −3.0
if v_θ predicts −2.5:  L_flow = (−2.5 + 3.0)² = 0.25
```

**Worked numeric example 2:** same pair, `t=0.8`:
```
x_t = 0.2·2.0 + 0.8·(−1.0) = −0.4
target v = −3.0   (unchanged — the target never depends on t)
if v_θ predicts −2.8:  L_flow = (−2.8 + 3.0)² = 0.04
```
`x_t` moves a lot between the two examples; the training target does not
move at all — that's the "rectified"/straight-line property.

### 8.3 Classifier-Free Guidance training

Text conditioning is randomly dropped (null embedding) on ~10% of
examples, enabling CFG at inference time (Part 2 §11) with no extra
training machinery.

### 8.4 Data

- Stage 0 (Tiny, i3): small curated 64×64 set.
- Stage 1+ (Kaggle): scale per §5, captions from §9's pipeline.

### 8.5 Multi-Resolution / Aspect-Ratio Bucketing — **clarified in this revision**

**Concrete bucket table** (v1, at Medium/Large native resolution — Tiny/
Small use a proportionally smaller version of the same table):

| Bucket | Aspect ratio | Resolution (Medium) |
|---|---|---|
| square | 1:1 | 256×256 |
| landscape | 4:3 | 296×224 |
| wide | 16:9 | 344×192 |
| portrait | 3:4 | 224×296 |
| tall | 9:16 | 192×344 |

**Assignment rule:** each training image's native aspect ratio is
compared to the five bucket ratios; it's assigned to the closest one by
absolute log-ratio difference, then center-cropped (never stretched) to
that bucket's exact resolution. Images too small for their assigned
bucket are dropped from training rather than upscaled (upscaling training
data would teach the model on blurred-then-relearned detail).

**How 2D-RoPE handles variable grids:** each bucket produces a different
patch-grid shape, but 2D-RoPE is computed from the row/column coordinate
of each patch, not from a fixed sequence length — so no architecture
change is needed per bucket. For grid sizes larger than anything seen at
Tiny-scale training time, the same NTK-style RoPE frequency rescaling
`aarambh-studio` already uses for long-context extrapolation (v2 phase,
YaRN/NTK) is reused here, extended to two dimensions — this is a direct,
concrete port of existing work, not a new mechanism.

**Batching:** images are grouped into batches by bucket (never mixed
across buckets in one batch), so every tensor in a batch has the same
shape — this is standard aspect-ratio-bucketed batching, not a novel
scheme.

---

## 9. Image Understanding / Captioning — Training In Detail

### 9.1 Architecture

```
image
   │
   ▼
Vision transformer encoder (patch-based, ViT-style, bootstrapped from
aarambh-studio's frozen CLIP-B/32)
   │
   ├── Contrastive projection head → image embedding (CLIP-style dual encoder)
   │
   └── Captioning head → small autoregressive transformer decoder,
         cross-attending into the vision encoder's tokens
```

### 9.2 Loss function

```
L_understand = L_contrastive   (InfoNCE, image↔text batch)
             + λ_cap · L_caption  (cross-entropy, next-token captioning)
```
`λ_cap = 1.0` once the contrastive stage has stabilised.

### 9.3 Why this comes before generation

The captioner auto-labels the image-only training pool for §8; the
contrastive embedding powers CLIP-score in the eval harness (Part 2
§22) and later becomes a GRPO/DPO reward (Part 2 §17).

### 9.4 Data

Any image set with even partial captions can bootstrap the contrastive
stage; once trained, the captioner runs once over the full image pool.

### 9.5 Fine-Tuning Schedule — **clarified in this revision**

The first draft said "fine-tuned further" without specifying what stays
frozen, for how long, or in what order. Concretely:

```
Step 1 — Load aarambh-studio's frozen CLIP-B/32 weights into the vision
         encoder. Everything frozen except the two new projection heads
         (contrastive text-projection, captioning decoder) — these are
         randomly initialised and trained first, for a fixed number of
         steps, against the loaded (frozen) vision encoder.
         Rationale: the loaded encoder already produces a useful
         embedding space; the new heads just need to learn to read it.

Step 2 — Unfreeze the top N transformer blocks of the vision encoder
         (not all of it) and continue contrastive-only training at a
         reduced learning rate for a second fixed number of steps.
         Rationale: lets the encoder adapt slightly to this project's
         specific image distribution without catastrophically forgetting
         what CLIP-B/32 already learned in aarambh-studio.

Step 3 — Enable the captioning loss (λ_cap turned on) on top of Step 2's
         checkpoint, train jointly for a third fixed number of steps.
         Rationale: captioning is trained last because it depends on
         having a stable, meaningful image embedding space to attend
         into — turning it on too early wastes steps on a moving target.

Step 4 — Freeze. Run the captioner once over the full uncaptioned image
         pool (§3.3 data pipeline) to produce training captions for §8.
```

Exact step counts are tuned during Phase 4 itself against the eval
harness (Part 2 §22) rather than fixed in advance — the schedule
(frozen → top-N unfrozen → captioning enabled → freeze) is the part that
doesn't change; the exact step counts are a Phase 4 tuning output,
recorded in that phase's milestone notes.

---

*(Continued in `ARCHITECTURE_VISION_STUDIO_PART2.md` §10 — Image Editing,
Structural Conditioning, and Reference-Image Prompting.)*
