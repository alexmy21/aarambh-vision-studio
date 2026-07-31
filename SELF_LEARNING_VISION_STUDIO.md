# SELF_LEARNING_VISION_STUDIO.md — aarambh-vision-studio

> Full design for the self-learning subsystem (`aarambh-vision-selflearn`,
> Phase 19). Read alongside `ARCHITECTURE_VISION_STUDIO_PART2.md` §17 and
> `ROADMAP_VISION_STUDIO_PART2.md` Phase 19. This mirrors Manas's
> associative-memory + anti-forgetting design and `aarambh-voice-studio`'s
> self-learning subsystem, adapted for images.

---

## Table of Contents

1. [Why Self-Learning, Not Just Fine-Tuning](#1-why-self-learning-not-just-fine-tuning)
2. [Architecture](#2-architecture)
3. [Associative Memory Bank](#3-associative-memory-bank)
4. [Gradient Orthogonalisation](#4-gradient-orthogonalisation)
5. [Confidence-Gated Commit](#5-confidence-gated-commit)
6. [Update Procedure, Step by Step](#6-update-procedure-step-by-step)
7. [Anti-Forgetting Verification](#7-anti-forgetting-verification)
8. [Relationship to Manas and `aarambh-voice-studio`](#8-relationship-to-manas-and-aarambh-voice-studio)
9. [Crate Layout (`aarambh-vision-selflearn`)](#9-crate-layout-aarambh-vision-selflearn)
10. [What This Is Not](#10-what-this-is-not)

---

## 1. Why Self-Learning, Not Just Fine-Tuning

`aarambh-vision-finetune` (Phase 17) already provides LoRA/QLoRA/DoRA
adapters for style and subject adaptation — but every adapter there is a
deliberate, offline training job: you choose images, run a training loop,
evaluate, ship. Self-learning exists for a different situation: the
system is *already running*, someone provides 3–5 new reference images of
a subject or style **during a session**, and the system should be able to
use them going forward without a human kicking off a full training run
each time — while never silently degrading anything it already knows.

That second half — "without silently degrading anything it already
knows" — is the entire reason this is a dedicated subsystem instead of
"just run a fast LoRA job in the background." A naive incremental update
overwrites shared weight directions and causes **catastrophic
forgetting**: the model gets better at the new subject and measurably
worse at subjects/styles it previously handled well. Self-learning is the
combination of three mechanisms that make incremental updates safe:
associative memory (§3), gradient orthogonalisation (§4), and
confidence-gated commits (§5).

---

## 2. Architecture

```
New reference images + subject/style name
              │
              ▼
   Lightweight adapter trained fast, few steps
   (small LoRA rank, aarambh-vision-finetune's machinery, reused not duplicated)
              │
              ▼
   Gradient orthogonalisation vs.
   - existing associative memory entries
   - frozen base model weight directions
              │
              ▼
   Candidate update
              │
              ▼
   Confidence gate: score candidate against aarambh-vision-eval
   on (a) the new subject/style, (b) a fixed regression suite covering
   every previously-learned subject/style
              │
        ┌─────┴─────┐
        ▼           ▼
     PASS         FAIL
        │           │
        ▼           ▼
   Commit to     Reject, log reason,
   associative   base model and memory
   memory bank   unchanged
```

---

## 3. Associative Memory Bank

Each learned subject or style is stored as a small, named entry:

```rust
pub struct MemoryEntry {
    pub name: String,               // e.g. "aria-mascot", "watercolor-poster-style"
    pub adapter: LoraAdapter,        // small-rank adapter weights
    pub reference_embedding: Vec<f32>, // pooled embedding from aarambh-vision-understand
    pub created_at_checkpoint: String, // which base checkpoint this was learned against
}
```

At inference time, a `DrishtiRequest`'s `style_adapter` field (§20,
ARCHITECTURE Part 2) can reference a memory entry by name, and the
adapter is applied on top of the frozen base weights for that request
only — the base model itself is never permanently modified by using a
memory entry, only by committing a *new* one through the gated process
below.

---

## 4. Gradient Orthogonalisation

When training the fast adapter for a new subject/style, its gradient
update is projected to be **orthogonal** to the gradient directions
already "claimed" by existing memory entries and by directions considered
load-bearing for the frozen base model's general capability. Conceptually,
for a candidate update direction `g_new` and a set of protected directions
`{g_1, ..., g_k}` (one per existing memory entry, plus a small set of
general-capability probe directions):

```
g_orth = g_new − Σ_i  ( (g_new · g_i) / (g_i · g_i) ) · g_i
```

This is a standard Gram-Schmidt-style projection: subtract out the
component of the new update that overlaps with each protected direction,
leaving only the part of the update that's genuinely specific to the new
subject/style. This is the same principle Manas uses for its own
anti-forgetting design, applied here to LoRA adapter gradients instead of
Manas's native weight updates.

---

## 5. Confidence-Gated Commit

Before a candidate update is written to the associative memory bank, it
is scored by `aarambh-vision-eval` (Phase 20) against two things:

1. **New-subject/style quality** — does the candidate actually produce
   recognisable, on-target output for the new subject/style (CLIP-score +
   aesthetic score against the reference images)?
2. **Regression suite** — a fixed held-out prompt set covering every
   previously-committed memory entry, scored the same way. If any
   previously-learned entry's score drops by more than an agreed
   tolerance, the candidate is rejected outright, regardless of how well
   it does on the new subject.

Only a candidate that passes **both** checks is committed. A rejected
candidate is logged with the specific regression that caused the
rejection, and the base model and memory bank are left completely
unchanged — this is what "safely" means in "learn after ship, safely"
(§2, ARCHITECTURE Part 1).

---

## 6. Update Procedure, Step by Step

```
1. Receive 3–5 reference images + a name for the new subject/style.
2. Compute reference_embedding via aarambh-vision-understand.
3. Train a small-rank LoRA adapter for a small, fixed number of steps
   (fast — this is not a full Phase 17 training run).
4. Orthogonalise the adapter's gradient updates against existing memory
   entries + protected general-capability directions (§4).
5. Score the candidate: new-subject quality + regression suite (§5).
6. If PASS: commit MemoryEntry to the associative memory bank.
   If FAIL: reject, log the reason, no changes persisted.
7. The new memory entry is immediately usable via DrishtiRequest.style_adapter
   in subsequent requests.
```

---

## 7. Anti-Forgetting Verification

Every commit is followed by an automatic re-run of the full regression
suite (not just the check that gated the commit) to confirm the persisted
state matches what was scored — this catches any discrepancy between the
gating check and the actually-committed weights before the next request
relies on them. The regression suite itself grows by one entry every time
a new subject/style is committed, so the bar for future updates gets
stricter, not looser, as the memory bank grows — mirroring exactly how
`aarambh-ai`'s forgetting diagnostics and Manas's own anti-forgetting
verification are designed to behave over time.

---

## 8. Relationship to Manas and `aarambh-voice-studio`

| Mechanism | Manas | `aarambh-voice-studio` | `aarambh-vision-studio` (this doc) |
|---|---|---|---|
| Incremental update mechanism | Native associative memory + weight update | LoRA-style adapter + gradient orthogonalisation | Same LoRA-style adapter + gradient orthogonalisation |
| Safety mechanism | Anti-forgetting verification | Confidence-gating against eval harness | Confidence-gating against `aarambh-vision-eval` (§5) |
| What's being protected | Previously-learned associative memory | Previously-learned speaker/style adaptations | Previously-learned subject/style adaptations |
| Failure mode being prevented | Catastrophic forgetting | Quality regression on prior speakers/styles | Quality regression on prior subjects/styles |

This subsystem is a direct port of the same idea across three different
modalities (native architecture, audio, image) rather than three
independently-designed mechanisms — a regression here is diagnosed the
same way a regression in Manas or `aarambh-voice-studio`'s self-learning
would be.

---

## 9. Crate Layout (`aarambh-vision-selflearn`)

```
aarambh-vision-selflearn/
├── src/
│   ├── memory.rs          — MemoryEntry, associative memory bank storage
│   ├── fast_adapter.rs    — quick LoRA training for a new subject/style
│   ├── orthogonalize.rs   — gradient projection (§4)
│   ├── gate.rs            — confidence-gated commit logic (§5)
│   ├── regression_suite.rs — grows by one fixed prompt set per commit
│   └── lib.rs
```

---

## 10. What This Is Not

- **Not** a replacement for `aarambh-vision-finetune`'s full adapter
  training pipeline — it's a fast, safety-gated path for incremental,
  session-time updates, not the primary way to train a high-quality
  style/subject adapter from a large reference set.
- **Not** a mechanism for learning real-person likenesses — subject
  adaptation here is scoped identically to Phase 17's fine-tuning scope
  (§27, ARCHITECTURE Part 2: objects, characters, products, styles, not
  real faces).
- **Not** continuous/always-on training — updates are explicit,
  triggered events (a new subject/style is provided), not a background
  process that trains on every generated image.
