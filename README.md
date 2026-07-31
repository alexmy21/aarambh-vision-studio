# aarambh-vision-studio

**A from-scratch image generation, editing, and understanding system, written entirely in Rust.**

Built on [`candle`](https://github.com/huggingface/candle), with no bindings to PyTorch and no vendored checkpoints. A sibling project to [`aarambh-ai`](https://github.com/AarambhDevHub/aarambh-ai) (a from-scratch LLM) and [`aarambh-voice-studio`](https://github.com/AarambhDevHub/aarambh-voice-studio) (a from-scratch audio studio) — same philosophy, applied to images.

[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Status](https://img.shields.io/badge/status-pre--implementation-yellow.svg)](./ROADMAP_VISION_STUDIO_PART1.md)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

---

## What is this?

`aarambh-vision-studio` is a **Multimodal Diffusion Transformer (MMDiT)**
trained with **Rectified Flow Matching** — the same architecture family
behind current production image models (SD3, FLUX, Qwen-Image, Sana) —
built up from raw tensors, with every tokenizer, transformer block, and
training loop implemented from scratch.

| Capability | What it does |
|---|---|
| **Text-to-Image** | Generate a new image from a text prompt |
| **Image Editing** | Give it a photo + an instruction ("make the sky orange") — it edits that specific photo |
| **Masked / Inpainting Editing** | Mark a region, change only what's inside it |
| **Reference-Image Prompting** | Use a reference photo as a style/subject prompt, zero-shot, no training needed |
| **Structural Conditioning** | Guide composition with an edge map, while the prompt controls content and style |
| **Super-Resolution Upscale** | Optional pass to sharpen final output detail |
| **Self-Learning** | Teach it a new subject/style from a few images during normal use, safely (anti-forgetting built in) |
| **Alignment** | Refined via GRPO + DPO against its own automatic quality scores |

No pretrained checkpoints are bundled with this repository — this is a
**source-only** release. Training recipes, architecture, and roadmap are
all fully documented so you can train your own.

---

## Documentation

| Document | What's in it |
|---|---|
| [`ARCHITECTURE_VISION_STUDIO_PART1.md`](./ARCHITECTURE_VISION_STUDIO_PART1.md) | Core architecture — VAE, text encoder, MMDiT backbone, understanding encoder |
| [`ARCHITECTURE_VISION_STUDIO_PART2.md`](./ARCHITECTURE_VISION_STUDIO_PART2.md) | Editing, reference-prompting, structural conditioning, sampling, distillation, alignment, safety, eval |
| [`ROADMAP_VISION_STUDIO_PART1.md`](./ROADMAP_VISION_STUDIO_PART1.md) | Step-by-step build plan, Phases 0–16 |
| [`ROADMAP_VISION_STUDIO_PART2.md`](./ROADMAP_VISION_STUDIO_PART2.md) | Step-by-step build plan, Phases 17–27 |
| [`SELF_LEARNING_VISION_STUDIO.md`](./SELF_LEARNING_VISION_STUDIO.md) | The self-learning subsystem — associative memory, anti-forgetting |
| [`VISION_STUDIO_COMPLETE_GUIDE_PART1.md`](./VISION_STUDIO_COMPLETE_GUIDE_PART1.md) / [`PART2`](./VISION_STUDIO_COMPLETE_GUIDE_PART2.md) | Every phase explained in plain language, no AI background needed |
| [`VISION_AI_DATASET_CREATION_GUIDE.md`](./VISION_AI_DATASET_CREATION_GUIDE.md) | How the training data itself gets built |
| [`VISION_STUDIO_MATH_FORMULAS_GUIDE.md`](./VISION_STUDIO_MATH_FORMULAS_GUIDE.md) | Every formula used, explained from zero, with worked examples |

New to the project? Start with the Complete Guide — it assumes no prior
AI knowledge.

---

## Architecture, at a glance

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

25 crates, one responsibility each — see `ARCHITECTURE_VISION_STUDIO_PART1.md` §4 for the full workspace layout.

---

## Project Status

**Pre-implementation.** Architecture and roadmap are complete; Phase 0
(workspace scaffold) has not yet been started. Track progress against
the 28-phase roadmap:

```
Phase 0-16  (Foundation)              → ROADMAP_VISION_STUDIO_PART1.md
Phase 17-27 (Advanced capabilities)   → ROADMAP_VISION_STUDIO_PART2.md
```

Estimated build time: ~7–10 months, part-time, on the hardware below.

---

## Hardware Philosophy

This project is deliberately built to train on **modest, free-tier
hardware only** — no paid compute assumed anywhere in the roadmap:

- A regular laptop CPU (development, Tiny-scale training, testing)
- Free-tier Kaggle GPU sessions (T4/P100) for Small/Medium/Large-scale training

If it doesn't run on that, it doesn't go in the roadmap.

---

## Building

```bash
git clone https://github.com/AarambhDevHub/aarambh-vision-studio
cd aarambh-vision-studio
cargo check --workspace
```

> Full build/training instructions land as each phase ships — see the
> Roadmap docs for the per-phase milestone commands.

---

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md)
for the workspace layout, coding conventions, and how phases are worked
through.

## Code of Conduct

This project follows the guidelines in [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).

## Security

Found a vulnerability? See [`SECURITY.md`](./SECURITY.md) for how to
report it responsibly.

---

## Part of Aarambh Dev Hub

`aarambh-vision-studio` is part of the [Aarambh Dev Hub](https://github.com/AarambhDevHub)
family of from-scratch Rust AI projects:

- [`aarambh-ai`](https://github.com/AarambhDevHub/aarambh-ai) — from-scratch LLM
- [`aarambh-voice-studio`](https://github.com/AarambhDevHub/aarambh-voice-studio) — from-scratch audio AI studio
- `aarambh-vision-studio` — from-scratch image generation, editing & understanding (this project)

If this project is useful to you, consider supporting its development:

- ☕ [Buy Me a Coffee](https://buymeacoffee.com/aarambhdevhub)
- 💖 [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- 💳 [Razorpay](https://razorpay.me/@aarambhdevhub) (UPI / cards, India)

---

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](./LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))

at your option.

Unless explicitly stated otherwise, any contribution intentionally
submitted for inclusion in this project by you shall be dual-licensed as
above, without any additional terms or conditions.
