# ARCHITECTURE_VISION_STUDIO.md — Part 2 of 2 — aarambh-vision-studio

> Continues directly from Part 1 §9. Read Part 1 first.

## Table of Contents (Part 2)

10. [Image Editing (Instruction-Based + Masked)](#10-image-editing-instruction-based--masked--training-in-detail)
11. [Reference-Image Prompting (IP-Adapter-Equivalent)](#11-reference-image-prompting-ip-adapter-equivalent--training-in-detail) — **new**
12. [Structural / Spatial Conditioning (ControlNet-Equivalent)](#12-structural--spatial-conditioning-controlnet-equivalent--training-in-detail) — **new**
13. [Sampling — ODE Solvers & Classifier-Free Guidance](#13-sampling--ode-solvers--classifier-free-guidance)
14. [Few-Step Distillation — Named Method](#14-few-step-distillation--training-in-detail)
15. [Super-Resolution / Upscale Refinement](#15-super-resolution--upscale-refinement--training-in-detail)
16. [Custom Kernels](#16-custom-kernels-aarambh-vision-kernel)
17. [Quantisation](#17-quantisation-aarambh-vision-quant)
18. [Fine-Tuning (LoRA/QLoRA/DoRA)](#18-fine-tuning-aarambh-vision-finetune)
19. [Alignment — GRPO + DPO](#19-alignment--grpo--dpo-aarambh-vision-align)
20. [Self-Learning (Summary)](#20-self-learning-summary-aarambh-vision-selflearn)
21. [Safety Layer & Provenance Watermarking](#21-safety-layer--provenance-watermarking-aarambh-vision-safety)
22. [Evaluation Harness + Named External Baselines](#22-evaluation-harness--named-external-baselines-aarambh-vision-eval)
23. [Full Control Layer](#23-full-control-layer-aarambh-vision-control)
24. [Crate-by-Crate Reference — 25 Crates](#24-crate-by-crate-reference--25-crates)
25. [Data Flow Across the Workspace](#25-data-flow-across-the-workspace)
26. [Memory & Compute Estimates](#26-memory--compute-estimates)
27. [Hardware Strategy](#27-hardware-strategy)
28. [Image Output Formats](#28-image-output-formats)
29. [Relationship to `aarambh-ai` and `aarambh-voice-studio`](#29-relationship-to-aarambh-ai-and-aarambh-voice-studio)
30. [What's Explicitly Out of Scope (v1)](#30-whats-explicitly-out-of-scope-v1)

---

## 10. Image Editing (Instruction-Based + Masked) — Training In Detail

### 10.1 Architecture

No separate editing model — the MMDiT backbone (Part 1 §8) gets a
**reference-image conditioning path**:

```
Source image ──► frozen VAE (Part 1 §6) ──► reference latent tokens ──┐
                                                                        │
Edit instruction ──► text encoder (Part 1 §7) ─────────────────────────┤
                                                                        ▼
Noisy target latent tokens ──────────────────────────► joint MMDiT
                                                          self-attention
                                                                        │
                                                                        ▼
                                              velocity prediction, loss-masked
                                              over TARGET tokens only (§10.2)
```

For **masked/local editing**, a binary mask channel is patchified
alongside the reference latent, marking which reference tokens must be
preserved exactly vs. regenerated. Global instruction edits omit this
channel.

### 10.2 Loss function

Identical rectified-flow loss as Part 1 §8.2, computed only over target-
image token positions:
```
L_edit = E[ ‖ v_θ(x_t, t, text, ref_tokens) − (x₁ − x₀) ‖² ]     (target tokens only)
```

### 10.3 Data construction — **clarified in this revision**

The first draft said "synthetic pair generation" without saying *how*.
Concretely, the primary data source is a **self-bootstrapped
prompt-to-prompt technique, run against your own Phase 10 checkpoint —
not a dependency on an external editing dataset or an external
text-to-image model:**

```
1. Take two related captions that differ in one controlled way
   (e.g. "a photo of a red car parked on a street" /
         "a photo of a blue car parked on a street").
2. Generate both images from the SAME initial noise sample and the SAME
   sampling trajectory, but inject the second caption's cross-attention
   maps at the tokens that differ, while keeping every other token's
   attention maps identical to the first generation (prompt-to-prompt
   cross-attention injection — this is what keeps composition, layout,
   and everything-but-the-edited-concept consistent between the pair,
   without needing any pixel-level supervision).
3. The first image becomes the "source," the second becomes the
   "target," and the instruction is derived directly from the difference
   between the two captions ("change the car color to blue") via a
   template for simple attribute/object swaps, or via the text encoder's
   own language understanding for more complex phrasing.
4. This runs entirely against your own trained model (post-Phase 10) —
   there is no dependency on downloading a third-party editing dataset,
   and no external text-to-image model is used to generate the pairs.
```

This is supplemented, not replaced, by two smaller sources:
- **Mask-based local edits** — `imageproc`-generated masks paired with
  "restore this region as X" instructions (unchanged from the first draft).
- **A small curated real-edit set** — a hand-verified set of genuine
  before/after edits, weighted higher late in training to correct for any
  synthetic-data artifacts (e.g. the model learning "prompt-to-prompt's
  specific visual signature" as if it were "what editing looks like").

### 10.4 Identity preservation

Because reference tokens are attended to directly (not compressed into a
single vector), the model has direct access to exact source
pixels-in-latent-form at every layer — the more of the source image
survives as literal tokens to copy from, the more faithful non-edited
regions are.

---

## 11. Reference-Image Prompting (IP-Adapter-Equivalent) — Training In Detail

**New in this revision.** This is different from editing (§10): editing
transforms a specific source image per an instruction; reference-image
prompting uses an image purely as an additional *prompt* — "generate
something new inspired by this image's subject/style" — with no
pixel-level correspondence expected between input and output, and
critically, **zero-shot at inference time** — no per-subject training
job required (that's what self-learning, `SELF_LEARNING_VISION_STUDIO.md`,
is for when you *do* want a trained, persistent adaptation).

### 11.1 Architecture — decoupled cross-attention

```
Reference image ──► frozen VAE-adjacent understanding encoder (Part 1 §9)
                     ──► image embedding
                                  │
                                  ▼
                     Small trainable projection network
                     (image embedding → K/V tokens, same dimensionality
                      as text cross-attention tokens)
                                  │
Text prompt ──► text encoder ──► text K/V tokens
                                  │
                                  ▼
          Each MMDiT block gets a SECOND cross-attention operation
          (the "decoupled" part — a separate attention op from the
          text one, not a shared one), whose output is added to the
          text cross-attention's output, scaled by a configurable
          `reference_strength` weight (0 = ignore reference entirely,
          1 = full influence)
```

The rest of the MMDiT — self-attention, AdaLN-Zero, the VAE — is
completely unchanged and stays frozen during this component's training.

### 11.2 Loss function and training strategy

Only the new projection network and the new decoupled cross-attention
weights are trained; every other MMDiT weight is frozen:
```
L_refprompt = E[ ‖ v_θ(x_t, t, text, ref_embedding) − (x₁ − x₀) ‖² ]
```
using the **same rectified-flow loss** as Part 1 §8.2 — nothing new
mathematically, only a new, small, trainable conditioning path.

**Training data needs no new collection:** the standard trick (used by
the published IP-Adapter work) is self-referential training — for each
(image, caption) pair already in the pipeline, use the image's own
embedding as the "reference" and its own caption as the "text," and
train the model to reconstruct that same image. Text conditioning is
randomly dropped during this training too, so at inference time the
reference image can be used with or without an accompanying text prompt.

### 11.3 Data

Reuses Part 1 §9's existing (image, caption) pipeline directly — no new
dataset.

---

## 12. Structural / Spatial Conditioning (ControlNet-Equivalent) — Training In Detail

**New in this revision.** Lets generation follow a precise spatial
layout — an edge map or a coarse structural sketch — rather than relying
on the text prompt alone to describe composition.

### 12.1 Scope decision for v1 (explicit, not assumed)

Structural conditioning is commonly extended to edge maps, depth maps,
and human pose skeletons. **Depth and pose both require their own
trained estimator model** to produce during data preparation, which is
real additional training work on top of everything else in this
roadmap. To keep this feature honest about the i3 + free-Kaggle-GPU
budget, **v1 scopes structural conditioning to edge maps only** (derived
deterministically via classical, non-learned edge detection —
`imageproc`'s Canny-equivalent — requiring no additional trained model to
produce training data). Depth-map and pose-skeleton conditioning are
explicitly deferred (§30, out of scope v1) rather than left as an
unstated gap.

### 12.2 Architecture — zero-convolution conditioning branch

```
Real image ──► classical Canny-equivalent edge extraction (imageproc,
               deterministic, no training required) ──► edge map
                                  │
                                  ▼
                     Small conv/patch encoder branch
                     (structural map → tokens, same shape as image tokens)
                                  │
                                  ▼
                     Zero-initialised output projection
                     ("zero convolution" — starts at exactly zero output,
                      so this branch begins training as a true no-op,
                      identical in spirit to AdaLN-Zero's zero-init gates,
                      Part 1 §8.1)
                                  │
                                  ▼
                     Added as a residual signal into the dual-stream
                     blocks' image-token path
```

The zero-initialised projection is what makes this safe to add on top of
an already-trained generation model — at the start of this phase's
training, the branch contributes nothing, so the base model's existing
generation quality cannot regress before the branch has learned anything
useful.

### 12.3 Loss function

Same rectified-flow loss (Part 1 §8.2), conditioned on `(text,
edge_map)` pairs:
```
L_structure = E[ ‖ v_θ(x_t, t, text, edge_map) − (x₁ − x₀) ‖² ]
```

### 12.4 Data

Fully self-supervised from the existing image pool — extract the edge
map from every training image, then train the model to regenerate an
image matching that edge structure (and the paired caption). No new data
collection, no external structural-conditioning dataset.

---

## 13. Sampling — ODE Solvers & Classifier-Free Guidance

### 13.1 ODE solver

- **Euler (default)** — `x_{t-Δt} = x_t − Δt · v_θ(x_t, t)`.
- **Heun's method (2nd-order, optional)** — one extra velocity evaluation
  per step, ~2× compute per step, fewer total steps for equal quality.

### 13.2 Classifier-Free Guidance (CFG)

```
v_cfg = v_uncond + guidance_scale · (v_cond − v_uncond)
```
`guidance_scale` is a `DrishtiRequest` field, typically 3.5–7.5.

### 13.3 Step counts

| Solver | Typical steps | Relative cost |
|---|---|---|
| Euler | 20–50 | 1× |
| Heun | 10–25 | ~2× per step, similar total wall-clock |
| Distilled (§14) | 1–4 | fraction of either |

---

## 14. Few-Step Distillation — Training In Detail

### 14.1 What changed here

The first draft said only "distillation" with no method. The named,
concrete recipe (mirroring the published fixes for the two well-known
failure modes of naive few-step distillation — blurry one-step output
and oversaturated few-step output) is a **three-stage pipeline**:

```
Stage A — Reflow
  Using the trained teacher (post-alignment checkpoint, §19), generate
  many (noise, image) pairs via full multi-step sampling. Retrain the
  SAME architecture from scratch on these straightened pairs. This is
  literally what "rectified" flow refers to — each round of reflow makes
  the ODE trajectories straighter, which is what allows large steps
  later. One round of reflow is the v1 target; a second round is a
  stretch goal if time/compute allow.

Stage B — Consistency distillation
  On top of the reflowed model, train a student to map any point along
  a trajectory directly to the trajectory's endpoint in one function
  evaluation, using the reflowed model as the teacher for intermediate
  targets (self-consistency training, in the style of Latent Consistency
  Models).

Stage C — Adversarial correction
  Stage A+B alone reproduces two documented failure modes: blurry
  one-step output and oversaturated few-step output. Stage C adds a
  discriminator loss on the STUDENT's few-step output specifically
  (adversarial correction on top of the consistency objective) — this is
  what fixes both failure modes in published work, rather than accepting
  them as an inherent cost of few-step sampling.
```

### 14.2 Loss function (Stage C, the novel part)

```
L_distill = L_consistency (Stage B's self-consistency loss)
          + λ_adv · L_adversarial (patch-GAN, same family as §6.3's VAE discriminator)
```

### 14.3 Measuring the tradeoff

Distillation quality is measured, not assumed: the eval harness (§22)
computes FID/CLIP-score/aesthetic-score for the distilled model at 1, 2,
and 4 steps against the full multi-step teacher, and against the named
external baselines (§22.2), so "is 4-step good enough" has a number
behind it rather than a visual guess.

---

## 15. Super-Resolution / Upscale Refinement — Training In Detail

### 15.1 Architecture

```
base image ──► frozen VAE encode ──► latent
                                        │
                                        ▼
                     Small refinement transformer (2–4 layers)
                                        │
                                        ▼
                     Predicted high-res residual latent
                                        │
                                        ▼
                     frozen VAE decode ──► upscaled image
```

### 15.2 Loss function

```
L_upscale = λ_recon · L_reconstruction (L1 + LPIPS)
          + λ_adv   · L_adversarial    (same patch-GAN family as §6.3)
```
Trained on downsampled-then-upsampled pairs derived from existing
high-resolution data — no new data collection.

### 15.3 Why a separate small model

Keeps the base model's training cost tractable on i3 + free Kaggle GPU,
same "separate, controllable stages" philosophy as `aarambh-voice-studio`.

---

## 16. Custom Kernels (`aarambh-vision-kernel`)

CPU SIMD kernels first-class, CUDA feature-gated, exercised once free
Kaggle GPU time is available — same strategy as `aarambh-ai` and
`aarambh-voice-studio`:

- **Fused patchify/unpatchify**
- **Fused 2D-RoPE application** (bucket-aware, Part 1 §8.5)
- **AdaLN-Zero modulation fusion**
- **Zero-convolution branch fusion** (§12.2's structural-conditioning
  residual add, fused to avoid kernel-launch overhead at small batch size)

---

## 17. Quantisation (`aarambh-vision-quant`)

INT8 (calibration-based) and INT4 (GGUF-style, per-block scales) for the
MMDiT and text-encoder wrapper. **VAE kept at higher precision** (F16
minimum) — reconstruction quality is disproportionately sensitive to
tokenizer precision.

---

## 18. Fine-Tuning (`aarambh-vision-finetune`)

LoRA / QLoRA / DoRA adapters over MMDiT attention + MLP weights, for
style adaptation and DreamBooth-style subject adaptation. This is the
mechanism self-learning (§20) builds on, and the mechanism GRPO/DPO
alignment (§19) runs against once a base checkpoint exists.

---

## 19. Alignment — GRPO + DPO (`aarambh-vision-align`)

| Reward source | What it scores |
|---|---|
| CLIP-score (Part 1 §9's encoder) | Text-image alignment |
| Aesthetic score (learned predictor, human-preference-labelled pairs) | Perceptual quality |
| Edit-consistency score | Non-edited-region fidelity for editing outputs |

GRPO refines general quality across many prompts; DPO applies where
paired preference data exists. Alignment runs after fine-tuning (§18) is
proven, so it doesn't need to be redone after every future adapter.

---

## 20. Self-Learning (Summary) (`aarambh-vision-selflearn`)

Full design in `SELF_LEARNING_VISION_STUDIO.md`: an associative memory
bank of subject/style embeddings, gradient orthogonalisation against
existing memory + the frozen base model, and confidence-gated commits
against the eval harness (§22) — a regressing update is rejected
automatically.

---

## 21. Safety Layer & Provenance Watermarking (`aarambh-vision-safety`)

- **Content filtering** — a classifier built on Part 1 §9's understanding
  encoder, screening prompts and outputs against a configurable policy.
- **Provenance watermarking** — every generated/edited image carries an
  invisible spread-spectrum watermark (survives moderate JPEG/WebP
  recompression) plus a `sha2` content hash recorded in generation
  metadata.
- No face-cloning-specific consent flag is needed — real-person
  likeness cloning is out of scope (§30).

---

## 22. Evaluation Harness + Named External Baselines (`aarambh-vision-eval`)

### 22.1 Metrics

| Metric | What it measures | Used by |
|---|---|---|
| FID | Distributional realism | Generation quality, release-over-release |
| CLIP-score | Text-image alignment | Generation + editing; GRPO/DPO reward |
| Aesthetic score | Learned human-preference predictor | Generation + editing; GRPO/DPO reward |
| Edit-consistency | Non-edited-region fidelity | Editing only; GRPO/DPO reward |

### 22.2 Named external baselines — **fixed in this revision**

The first draft said only "a frozen open-weight checkpoint" — no name.
v1 uses **two** named, permissively-licensed models, at two different
scales of ambition:

| Baseline | Role | License | Why this one |
|---|---|---|---|
| **Sana** (NVIDIA) | Primary, peer-scale comparison | Permissive open weights | Explicitly optimised for speed/efficiency at small model scale — the fairest comparison for a project bound to i3 + free-Kaggle-GPU training, rather than comparing against a model trained on datacenter-scale compute |
| **FLUX.2 [klein] 4B** (Black Forest Labs) | Aspirational stretch comparison | Apache 2.0 | Compact enough to run inference on a single consumer-class GPU, but a meaningfully higher quality ceiling than Sana — gives a sense of "how far from frontier-compact are we," not just "are we better than nothing" |

**Important boundary, stated explicitly:** both baselines are used
**read-only, for scoring comparison metrics only** — never as a training
initialisation, distillation teacher, or weight-merge source for your
own model. This preserves the "from scratch, no vendored checkpoints"
claim for the model itself, while still giving every release a real
external number to compare against instead of only your own prior
checkpoint.

### 22.3 Report generation

A single command runs all four metrics for the current checkpoint,
Sana, and FLUX.2 [klein] 4B, on the same fixed validation prompt set, and
produces one comparison report — this is what every release's "how good
is it" answer is measured against, not a vibe check.

---

## 23. Full Control Layer (`aarambh-vision-control`)

```rust
pub struct DrishtiRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub mode: DrishtiMode,           // Generate | Edit | Inpaint | Upscale
    pub reference_image: Option<ImageRef>,       // editing source OR reference-prompt image
    pub reference_mode: Option<ReferenceMode>,   // Edit | Prompt (§10 vs §11)
    pub reference_strength: Option<f32>,         // §11.1 — 0..1
    pub structural_map: Option<StructuralRef>,   // §12 — edge map
    pub structural_strength: Option<f32>,        // 0..1
    pub mask: Option<MaskRef>,
    pub resolution: (u32, u32),
    pub guidance_scale: f32,
    pub steps: u32,
    pub solver: SolverKind,          // Euler | Heun | Distilled
    pub seed: Option<u64>,
    pub output_format: ImageOutputFormat,  // Png | Jpeg{quality} | WebP{quality}
    pub style_adapter: Option<AdapterRef>, // LoRA/self-learned style or subject
}
```

---

## 24. Crate-by-Crate Reference — 25 Crates

| Crate | Layer | Responsibility |
|---|---|---|
| `aarambh-vision-core` | L0 | Config, request/response types, errors |
| `aarambh-vision-tokenizer` | L1 | Image VAE — encode/decode + fallback mode (Part 1 §6.5) |
| `aarambh-vision-textprep` | L1 | Prompt tokenization, normalisation |
| `aarambh-vision-data` | L1 | Dataset loaders, auto-captioning, resolution bucketing |
| `aarambh-vision-understand` | L1 | CLIP-style encoder + captioner |
| `aarambh-vision-nn` | L2 | MMDiT block, AdaLN-Zero, 2D-RoPE (bucket-aware) |
| `aarambh-vision-kernel` | L2 | CPU SIMD, CUDA prep, fused ops |
| `aarambh-vision-textencoder` | L3 | Thin loader for `aarambh-ai`'s decoder-only checkpoint |
| `aarambh-vision-model` | L3 | Full MMDiT assembly — gen + edit + structure + refprompt heads |
| `aarambh-vision-weights` | L3 | SafeTensors save/load, checkpoint conversion |
| `aarambh-vision-train` | L4 | Pretraining loops, rectified flow, distillation (§14) |
| `aarambh-vision-quant` | L4 | INT8/INT4/GGUF-style quantisation |
| `aarambh-vision-finetune` | L5 | LoRA/QLoRA/DoRA adapters |
| `aarambh-vision-align` | L5 | GRPO + DPO |
| `aarambh-vision-selflearn` | L5 | Online self-learning |
| `aarambh-vision-edit` | L6 | Reference-image conditioning (editing), masks |
| `aarambh-vision-structure` | L6 | **New** — structural/spatial conditioning (§12) |
| `aarambh-vision-refprompt` | L6 | **New** — reference-image prompting (§11) |
| `aarambh-vision-upscale` | L7 | Super-resolution refinement |
| `aarambh-vision-safety` | L9 | Content filtering, watermarking |
| `aarambh-vision-eval` | L9 | Metrics + named baseline comparison (§22) |
| `aarambh-vision-control` | L9 | `DrishtiRequest` DSL |
| `aarambh-vision-inference` | L9 | Sampler, CFG, shared runtime |
| `aarambh-vision-serve` | L10 | HTTP inference server |
| `aarambh-vision-studio` (bin) | L11 | CLI |

---

## 25. Data Flow Across the Workspace

```
raw images ──► -tokenizer (VAE, frozen after Phase 2, fallback-aware)
raw images ──► -understand (captioner) ──► auto-generated captions
(images, captions) ──► -data ──► -train (rectified flow)
(image A, image A's own caption) ──► -refprompt (self-referential training, §11.2)
real image ──► classical edge extraction ──► -structure (§12.4)
prompt-to-prompt pairs from own checkpoint ──► -edit (§10.3)
trained checkpoint ──► -weights ──► -quant / -finetune / -align / -selflearn
post-alignment checkpoint ──► -train (distillation, §14: reflow → consistency → adversarial)
final checkpoint + adapters ──► -inference ──► -upscale (optional) ──► -serve / CLI
every output ──► -safety (filter + watermark) ──► returned to caller
every checkpoint change ──► -eval (scored against Sana + FLUX.2 [klein]) before being treated as current
```

---

## 26. Memory & Compute Estimates

| Scale | Params (MMDiT + VAE + text-encoder, approx.) | FP32 training memory (single sample) | Feasible on |
|---|---|---|---|
| Tiny   | ~40M + 40M + ~30M  | ~2–3 GB  | i3 (CPU, slow) / Kaggle T4 |
| Small  | ~120M + 40M + ~60M | ~5–7 GB  | Kaggle T4/P100 |
| Medium | ~350M + 40M + ~125M| ~12–16 GB| Kaggle P100 (batch size limited) |
| Large  | ~900M + 40M + ~350M| ~28–36 GB| Kaggle GPU, gradient checkpointing required |

Treat as planning-order-of-magnitude; measure actual usage at the start
of each phase and adjust batch size accordingly.

---

## 27. Hardware Strategy

**i3 + free Kaggle GPU only**, no paid compute assumed anywhere. i3
handles Tiny-scale training, CPU-kernel work, data pipeline development,
and inference testing. Kaggle T4/P100 sessions handle Small/Medium/Large
training, VAE Stage 1, and any GAN-adversarial training (§6.3, §15.2)
impractical on CPU.

---

## 28. Image Output Formats

| Format | Use case | Encoder |
|---|---|---|
| **PNG** (default) | Lossless, generation/editing default | `image` crate |
| **JPEG** | Smaller files, lossy acceptable | `image` crate |
| **WebP** | Modern format, strong compression | `webp` crate (feature-gated, off by default) |
| **Latent export** (`.safetensors`) | Chaining tools, e.g. before an upscale pass | `safetensors` |

`ImageOutputFormat` is a `DrishtiRequest` field (§23); PNG is the default.

---

## 29. Relationship to `aarambh-ai` and `aarambh-voice-studio`

| Shared with `aarambh-ai` | Shared with `aarambh-voice-studio` | New in this project |
|---|---|---|
| Transformer block primitives | "Trained once, frozen thereafter" tokenizer discipline | MMDiT (dual/single-stream, AdaLN-Zero, 2D-RoPE) |
| Frozen CLIP-B/32, reused as bootstrap (Part 1 §9) | "Understanding before generation" ordering | Rectified flow matching + ODE sampling |
| **Decoder-only checkpoint reused directly as text encoder (Part 1 §7)** | "Separate, controllable stages" (upscale as its own pass) | Reference-image conditioning (editing, §10), prompting (§11), structural conditioning (§12) |
| NTK/YaRN RoPE extrapolation, reused for 2D buckets (Part 1 §8.5) | GRPO/DPO alignment pattern, rewards from own eval harness | Named 3-stage distillation (§14), named external baselines (§22.2) |
| BPE tokenizer (reused directly, Part 1 §7.2) | candle 0.11 major/minor pin | Provenance watermarking (image-specific) |

---

## 30. What's Explicitly Out of Scope (v1)

- **Video generation** — a natural future sibling, not attempted here.
- **3D/multi-view consistent generation** — single 2D images only.
- **Depth-map and pose-skeleton structural conditioning** — v1 scopes
  structural conditioning to edge maps only (§12.1); both would need
  their own trained estimator model, deferred past v1.
- **Real-person likeness cloning** — subject adaptation (§18/§20) is
  scoped to objects, characters, products, and styles, not real faces.
- **Training-time human-preference data collection infrastructure** —
  the aesthetic-score predictor (§19) assumes an existing labelled
  preference set; a live human-feedback pipeline is future work.
- **Multi-GPU / distributed training** — single-GPU (Kaggle free tier)
  only.
