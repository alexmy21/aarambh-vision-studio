# Aarambh-Vision-Studio: The Complete Beginner's Guide

### Everything we built, in plain human language

This document explains, step by step, everything inside
**aarambh-vision-studio** — the from-scratch image generation, editing,
and understanding system built in Rust using Candle. It covers the
planned **v1.0.0** roadmap, Phases 0 through 27. Think of this as a
story: each phase builds on top of the one before it, like constructing
a building floor by floor — exactly the same idea as the
`aarambh-studio-complete-guide.md` you already have, just for images instead
of text.

No prior AI knowledge assumed. Every section has:
- A **plain-English definition**
- A **beginner explanation**
- **Why we actually need it** in an image model
- A **real-world example**
- A **diagram**
- **Common questions** a beginner would ask

---

## The Big Picture First

Before diving into 28 phases, here's the one-sentence version of what an
image generation model actually is:

> An image generation model is a giant mathematical function that has
> been shown so many (image, caption) pairs that it learned the
> statistical relationship between "words describing a picture" and
> "pixels that match those words" well enough that, given ONLY new
> words, it can invent pixels that have never existed before.

Here is the full pipeline, zoomed way out, so you can see where every
phase below fits:

```
 RAW IMAGES + CAPTIONS
        │
        ▼
┌─────────────┐
│  IMAGE VAE  │  (turns pictures into compact number-grids, "latents")
└─────────────┘
        │
        ▼
┌─────────────┐
│    TEXT     │  (turns your prompt into numbers the model understands)
│  ENCODER    │
└─────────────┘
        │
        ▼
┌─────────────┐
│   MMDiT     │  (the actual "brain" — the transformer that learns
│  BACKBONE   │   to turn noise into a matching picture)
└─────────────┘
        │
        ▼
┌─────────────┐
│  TRAINING   │  (the model practices on millions of examples)
│    LOOP     │
└─────────────┘
        │
        ▼
┌─────────────┐
│  SAMPLING /  │  (the trained model draws you a new picture)
│  INFERENCE   │
└─────────────┘
```

Everything else — editing, reference-image prompting, structural
conditioning, quantization, LoRA, alignment, self-learning — are
**upgrades** bolted onto this core pipeline, exactly the same philosophy
as `aarambh-studio`. Keep this diagram in your head as we go.

---

# PART 1 — Foundation (Phases 0–13)

## Phase 0: Workspace + Core Types

**Definition:** The workspace is the empty skeleton of the whole
project — folders and empty crates (Rust's word for "sub-project"), with
no real code inside yet, just the shape everything else will fill in.

**Beginner explanation:**
Think of building a house. Before you lay a single brick, you mark out
where every room will be with string and stakes. Phase 0 is that
string-and-stakes step — 25 empty crates, each one destined to hold one
piece of the system (the tokenizer, the training loop, the server, and
so on), plus one shared file of basic types every other crate will use.

**Why we need it:**
Without agreeing on the shape up front, later phases would constantly
be moving code between files, arguing about where things belong. Phase 0
settles that once, so every later phase knows exactly which file its
new code goes into.

**Example:**
```
Before Phase 0: one big folder with no organization
After Phase 0:
  crates/aarambh-vision-tokenizer/   ← the VAE will live here (Phase 1)
  crates/aarambh-vision-nn/          ← the MMDiT will live here (Phase 5)
  crates/aarambh-vision-train/       ← the training loop lives here
  ...22 more crates, each with ONE job
```

**Diagram:**
```
   aarambh-vision-studio/
        │
        ├── aarambh-vision-core     (shared types — built NOW)
        ├── aarambh-vision-tokenizer (empty — filled in Phase 1)
        ├── aarambh-vision-nn        (empty — filled in Phase 5)
        └── ...21 more, all empty until their phase arrives
```

**Common beginner questions:**
- *Q: Why so many separate crates instead of one big file?* → Each crate
  compiles somewhat independently and has one clear job, so a bug in the
  safety filter can't accidentally break the VAE — the same reason
  `aarambh-studio` has many small crates instead of one giant one.
- *Q: What's actually "real" after Phase 0?* → Only the shared config and
  request types. Everything else is an empty shell waiting to be filled.

---

## Phase 1: Image VAE (The "Tokenizer" for Pictures)

**Definition:** The Image VAE (Variational Autoencoder) is the component
that squeezes a big picture down into a small grid of numbers (a
"latent"), and can also do the reverse — turn that small grid back into
a full picture.

**Beginner explanation:**
`aarambh-studio` has a tokenizer that turns words into numbers. The Image VAE
is the picture equivalent — instead of turning "playing chess" into
`[1045, 2075, 8899]`, it turns a 512×512 photo into a much smaller grid
of numbers that still captures everything important about the photo.
Working with that small grid instead of the full-size photo is what
makes training an image model computationally possible at all — a raw
512×512 photo has over 780,000 numbers (pixels × color channels); the
compressed latent has a tiny fraction of that.

**Why we need it:**
Without this compression step, every later phase (the MMDiT backbone,
training, editing) would have to work directly on full-resolution pixels
— far too slow and memory-hungry on the free-tier hardware this whole
project is built for. This is also, like `aarambh-studio`'s tokenizer, the
**one component trained first and then frozen forever** — every phase
after this treats it as a fixed, trustworthy tool.

**Example:**
```
Input:  a 512×512 photo of a cat (786,432 numbers: 512×512×3 colors)
                    │
                    ▼  Image VAE encoder
Output: a 64×64 grid of 16 numbers per cell (only 65,536 numbers)
                    │
                    ▼  Image VAE decoder (the reverse direction)
Output: the cat photo again, reconstructed almost perfectly
```

**Diagram:**
```
  full photo (huge)
        │
        ▼
  ┌───────────────┐
  │   ENCODER     │  "squeeze"
  └───────────────┘
        │
        ▼
  small latent grid  ← this is what every other phase actually works with
        │
        ▼
  ┌───────────────┐
  │   DECODER     │  "un-squeeze"
  └───────────────┘
        │
        ▼
  full photo again (reconstructed)
```

**Common beginner questions:**
- *Q: Does the VAE lose information when it compresses?* → A little —
  the reconstructed photo isn't pixel-perfect, but it's close enough
  that a human usually can't tell the difference. That small, careful
  loss is exactly what "auto**encoder**" training is designed to minimize.
- *Q: Why is this the riskiest phase in the whole roadmap?* → Because
  training something to compress AND reconstruct well at the same time
  is notoriously finicky — this project builds it in careful stages
  (start simple, add complexity only once each step works) specifically
  because of that risk, with a documented fallback if the trickiest
  stage doesn't cooperate.
- *Q: Why is it frozen after this phase — can't it keep improving?* →
  Every later phase's training assumes the "translation" between
  pictures and latents never changes underneath it. If the VAE kept
  changing, all that later training would have to restart.

---

## Phase 2: Text Prep + Prompt Tokenizer

**Definition:** This phase gets your typed prompt ready to be understood
by a language model — cleaning it up and converting it into numbers.

**Beginner explanation:**
This reuses `aarambh-studio`'s own tokenizer directly, rather than building
a new one — the exact same "words into numbers" translator you already
have. This phase's job is just to wire that existing translator into
this project and handle small details, like what "nothing" (an empty
prompt) should look like as numbers.

**Why we need it:**
Every prompt you type has to become numbers before any neural network
can touch it — same reason as `aarambh-studio`'s Phase 1, just reused rather
than rebuilt.

**Example:**
```
Your prompt: "a red apple on a wooden table"
                    │
                    ▼  aarambh-studio's existing tokenizer
Tokens: [512, 8831, 219, 4471, 90, 3350, 1122]
```

**Diagram:**
```
  "a red apple on a wooden table"
              │
              ▼
     (aarambh-studio's tokenizer — reused, not rebuilt)
              │
              ▼
  [512, 8831, 219, 4471, 90, 3350, 1122]
```

**Common beginner questions:**
- *Q: Why not build a brand-new tokenizer for images?* → Because the
  tokenizer's job is entirely about text, and `aarambh-studio` already solved
  that problem well — rebuilding it would be pure duplicated effort.
- *Q: What's a "null" or "empty" prompt for?* → It's used later
  (Phase 8) so the model can learn what "no instruction at all" looks
  like — that's the trick behind a feature called classifier-free
  guidance, explained in Phase 9.

---

## Phase 3: Data Pipeline + Resolution Bucketing

**Definition:** The data pipeline is the system that loads images and
their captions, resizes them consistently, and groups them into
efficient batches for training.

**Beginner explanation:**
Same factory-conveyor-belt idea as `aarambh-studio`'s data pipeline, but for
pictures instead of text. One extra wrinkle specific to images:
photos come in all sorts of shapes (a phone photo is tall, a landscape
photo is wide, a profile picture is square). "Resolution bucketing" is
the system that sorts incoming images into a handful of standard shapes
so the model can be trained efficiently on all of them without forcing
every image into an ugly, stretched square.

**Why we need it:**
Without organized batches, training either crawls or crashes — same
reason as `aarambh-studio`. The bucketing part specifically exists because
real-world images aren't all one shape, and this project wants to
generate wide photos, tall photos, and square photos equally well.

**Example:**
```
Incoming photo: 1200×800 pixels (a landscape shape)
                    │
                    ▼ compare to standard bucket shapes
Closest bucket: "landscape, 4:3" → resize/crop to 296×224
                    │
                    ▼
Goes into a batch with OTHER landscape-bucket photos only
(never mixed with square or portrait photos in the same batch)
```

**Diagram:**
```
   photo (any shape)
        │
        ▼
 ┌───────────────┐
 │  which bucket  │   square? landscape? wide? portrait? tall?
 │  is closest?   │
 └───────────────┘
        │
        ▼
  resize/crop to that bucket's exact size
        │
        ▼
  batched only with same-bucket photos
```

**Common beginner questions:**
- *Q: Why not just squish every photo into a square?* → Squishing
  distorts the picture (a round plate would become oval), and the model
  would learn those distortions as if they were normal — bucketing keeps
  photos looking correct.
- *Q: What happens to a photo too small for its bucket?* → It's dropped
  from training rather than stretched up — stretching a small photo
  bigger just teaches the model on blurry, made-up detail.

---

## Phase 4: Image Understanding (Teaching the Model to "See" Before It "Draws")

**Definition:** This phase builds a component that can look at a picture
and produce both (a) a summary "fingerprint" of what's in it, and (b) an
actual one-sentence caption describing it.

**Beginner explanation:**
Before this project teaches a model to *draw* pictures, it teaches a
model to *understand* pictures — the same "understanding before
generation" idea used in the audio sibling project. This component is
bootstrapped from a vision component `aarambh-studio` already has (rather
than starting from nothing), then taught two skills: first, matching
pictures to the right caption out of many candidates (like a matching
game); second, writing its own caption for a picture with no caption at
all.

**Why we need it:**
Two big payoffs: first, this is what lets the project auto-caption huge
piles of uncaptioned photos instead of needing every single training
image hand-labeled by a human. Second, the "fingerprint" this component
produces becomes the project's main way of scoring "does this generated
image actually match the prompt" later on (Phase 23).

**Example:**
```
Input: a photo of a golden retriever running on a beach

Matching-game skill: given the photo + 5 candidate captions,
  correctly picks "a golden retriever running on a beach" as the match

Captioning skill: given ONLY the photo, writes its own caption:
  "A dog running along the shore near the ocean"
```

**Diagram:**
```
        photo
          │
          ▼
 ┌─────────────────┐
 │  vision encoder  │  (bootstrapped from aarambh-studio's existing component)
 └────────┬─────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
"fingerprint"  caption-writer
(used for      (writes a sentence
 matching/     describing the photo)
 scoring)
```

**Common beginner questions:**
- *Q: Why teach "understanding" before "drawing"?* → Because the
  drawing phase (Phase 8) needs captions for almost every training photo,
  and hand-writing millions of captions isn't realistic — this phase's
  auto-captioning skill solves that problem first.
- *Q: Why reuse an existing component instead of starting fresh?* →
  Training a good "vision fingerprint" component from nothing is a big,
  separate undertaking; reusing one that already exists and fine-tuning
  it saves a huge amount of redundant work.

---

## Phase 5: NN Primitives — the MMDiT Building Blocks

**Definition:** This phase builds the actual Lego bricks the image-
drawing brain (Phase 8) will be assembled from — the specific kind of
transformer block used by modern image models, called MMDiT.

**Beginner explanation:**
`aarambh-studio` already has transformer building blocks (used for
understanding and generating text). This phase adapts that same idea for
pictures, with two picture-specific additions: (1) a way for the model
to know WHERE a piece of the picture sits (top-left corner? dead
center?) — since text has a natural left-to-right order but a picture's
patches are arranged in a 2D grid; and (2) a way for a single number (a
"timestep," explained in Phase 8) to gently steer every layer of the
network.

**Why we need it:**
Without a 2D sense of position, the model would have no idea that a
patch of "sky" and a patch of "ground" are supposed to be arranged
vertically rather than in some random order — position awareness is what
lets the model draw sensible, coherent pictures instead of a scrambled
patchwork.

**Example:**
```
An image cut into a 4×4 grid of patches:

  [0,0] [0,1] [0,2] [0,3]
  [1,0] [1,1] [1,2] [1,3]
  [2,0] [2,1] [2,2] [2,3]
  [3,0] [3,1] [3,2] [3,3]

Each patch gets a "2D address" ([row, column]) baked into its numbers,
so the model always knows a patch's actual position in the picture —
even if patches get reordered inside the math, the position travels with them.
```

**Diagram:**
```
  patch tokens + 2D position info
            │
            ▼
  ┌───────────────────┐
  │  MMDiT building     │
  │  block (attention +  │
  │  timestep steering)  │
  └───────────────────┘
            │
            ▼
  transformed patch tokens, ready for the next block
```

**Common beginner questions:**
- *Q: Is this a totally new invention?* → No — it directly reuses
  `aarambh-studio`'s attention-block ideas, just with the extra 2D-position
  and timestep-steering pieces added on top for pictures specifically.
- *Q: What's a "timestep" and why does it steer the network?* → Explained
  fully in Phase 8, where the training process is described — for now,
  think of it as "how noisy is this picture right now," and the network
  needs to behave differently at different noise levels.

---

## Phase 6: CPU SIMD Kernels + CUDA Prep

**Definition:** This phase writes specialized, extra-fast versions of
the most-repeated small operations from Phase 5, so training and
generating images isn't slower than it needs to be.

**Beginner explanation:**
Some operations (like the 2D-position math from Phase 5) get run an
enormous number of times per single generated image. A "kernel" here
just means "a small, hand-optimized piece of code for one specific
operation," written to take advantage of the CPU's ability to do several
similar calculations at once (SIMD = doing the same math on multiple
numbers simultaneously, instead of one at a time).

**Why we need it:**
This whole project is built to run on modest, free-tier hardware — a
regular laptop CPU and free cloud GPU sessions, not an expensive
data-center computer. Without these speed-ups, even simple experiments
would take uncomfortably long.

**Example:**
```
Naive way: calculate one patch's position-math, one at a time,
           4×4 = 16 times in a slow loop

SIMD way:  calculate 4 or 8 patches' position-math AT THE SAME TIME,
           using the CPU's built-in "do many at once" instructions
           → meaningfully faster for the exact same result
```

**Diagram:**
```
  naive:  [patch1] → [patch2] → [patch3] → [patch4]   (one at a time)

  SIMD:   [patch1, patch2, patch3, patch4]  → all together, one instruction
```

**Common beginner questions:**
- *Q: Does this change what the model learns?* → No — a fused/optimized
  kernel produces the exact same numeric answer as the slow version,
  just faster. It's a speed change, not a behavior change.
- *Q: Why prepare CUDA (GPU) code now if training happens on CPU first?*
  → So the same code is ready to use the moment a free-tier GPU session
  is available, without a separate rewrite later.

---

## Phase 7: Text Encoder Integration

**Definition:** This phase connects `aarambh-studio`'s existing language
model directly into this project, so it can read prompts and describe
what it understood in a way the MMDiT backbone (Phase 5/8) can use.

**Beginner explanation:**
Rather than building a brand-new "prompt understanding" component, this
phase loads `aarambh-studio`'s already-trained brain exactly as-is and takes
a peek at what's happening partway through it — not its very final
output, but a snapshot from partway through its thinking — because that
snapshot turns out to carry richer information about the prompt's
meaning than the very last step does.

**Why we need it:**
A model that already deeply understands language (spatial words like
"behind," counting words like "three," relationships like "on top of")
gives the image-drawing brain a real head start, instead of the image
model having to relearn basic language understanding from scratch with a
much smaller, simpler text component.

**Example:**
```
Prompt: "a red cube behind a blue sphere"
              │
              ▼  aarambh-studio's existing brain (loaded, not retrained)
        [thinking... layer 1, layer 2, ... layer 16 (snapshot taken HERE), ...final layer]
              │
              ▼
       rich per-word understanding, handed to the MMDiT (Phase 8)
```

**Diagram:**
```
  prompt
    │
    ▼
  aarambh-studio's brain, layer by layer
    │
    ├── layer 8
    ├── layer 16  ← snapshot taken here (chosen by a small experiment)
    ├── layer 24
    └── final layer
    │
    ▼
  used as "what the prompt means" for the image model
```

**Common beginner questions:**
- *Q: Why not just use the very last layer, like you might expect?* →
  Testing shows an in-between snapshot often captures meaning better for
  this specific purpose — this project actually runs a small experiment
  during this phase to pick the best layer, rather than assuming.
- *Q: Does this phase train anything new?* → Barely — it mostly wires up
  something that already exists. That's the entire point: reuse instead
  of rebuild.

---

## Phase 8: Text-to-Image Baseline — the Model Draws Its First Picture

**Definition:** This is the phase where all the previous pieces (VAE,
text understanding, MMDiT blocks) are connected together and trained for
the first time, so the model can turn a text prompt into an actual
picture.

**Beginner explanation:**
The training trick here is called "rectified flow." Imagine starting
with pure TV-static noise and a caption like "a red apple." Training
teaches the model: "if you're looking at pure static, and you want to
end up at a red apple, which direction should every pixel-value nudge?"
Do that nudge a bunch of times in a row, each time asking the same
question again, and the static gradually turns into a red apple. The
model doesn't memorize apple pictures — it learns the general skill of
"which direction to nudge, given how noisy things currently are and what
the caption says."

**Why we need it:**
This is the actual core capability of the entire project — everything
before this phase was preparation, and everything after this phase is
either scaling it up or adding new abilities on top of it.

**Example:**
```
Step 0 (pure noise):        [static, static, static...]
Step 1 (nudge a little):    [static, but VAGUELY apple-red-ish]
Step 5 (nudge more):        [a blurry red round shape appears]
Step 20 (final nudge):      [a recognizable red apple]

Each "nudge" is the model answering: "given this much noise + this
caption, which direction should the numbers move?"
```

**Diagram:**
```
   pure noise + "a red apple"
            │
            ▼  nudge (repeat many times)
       ...getting less noisy, more apple-shaped...
            │
            ▼
       finished latent  →  VAE decoder (Phase 1)  →  actual picture
```

**Common beginner questions:**
- *Q: Why "rectified" flow specifically?* → Because the training target
  (which direction to nudge) is set up to be a straight, constant line
  from noise to picture for a given example — a "straight line" is much
  easier and faster to follow than a wiggly path, which is part of why
  this approach needs fewer nudging steps than older methods.
- *Q: Why is this called the heaviest phase in the whole roadmap?* →
  Everything before it was building tools; this is the phase where those
  tools have to actually work together correctly for the first time —
  the highest-uncertainty step, same as any "first real integration"
  moment in a big project.

---

## Phase 9: Sampler + CLI

**Definition:** This phase builds the actual "nudge repeatedly" loop
described in Phase 8 into a real, usable tool, plus a command you can
type to use it.

**Beginner explanation:**
Phase 8 taught the model HOW to nudge; Phase 9 builds the tool that
actually performs those nudges in sequence, a chosen number of times, and
hands you back a finished picture. It also adds a trick called
"classifier-free guidance" — remember Phase 2's "empty prompt" idea?
This phase uses it: the model nudges once pretending it has no caption
at all, and once with your real caption, then exaggerates the
difference between the two — which makes the final picture follow your
prompt more closely.

**Why we need it:**
Without this, there'd be no way to actually ask the trained model for a
picture — Phase 8 proves the model CAN learn to draw; Phase 9 is what
lets a person actually use that ability.

**Example:**
```
$ aarambh-vision-studio generate --prompt "a red apple on a table" --steps 30
→ produces apple.png
```

**Diagram:**
```
  "a red apple" (with caption)  →  nudge direction A
  "" (no caption)                →  nudge direction B
                    │
                    ▼
  final nudge = B + (extra push) × (A − B)
                    │
                    ▼
        follows your prompt more strongly
```

**Common beginner questions:**
- *Q: Why do two nudge-calculations per step instead of one?* → That's
  the cost of classifier-free guidance — it roughly doubles the work per
  step, but meaningfully improves how closely the output matches your
  words.
- *Q: What does "steps" in the CLI command control?* → How many times
  the nudge loop repeats — more steps generally means a more refined
  picture, at the cost of more time; Phase 24 later builds a way to get
  good results in far fewer steps.

---

## Phase 10: Multi-Resolution / Aspect-Ratio Bucket Training

**Definition:** This phase teaches the model to draw well at all of
Phase 3's bucket shapes (square, landscape, wide, portrait, tall), not
just one.

**Beginner explanation:**
Phase 8 proved the model could learn to draw at all, using just one
simple square shape. This phase repeats that training across every
bucket shape, so asking for a wide landscape picture works just as well
as asking for a square one.

**Why we need it:**
Real requests aren't always square — a phone wallpaper is tall, a
banner image is wide. Training across all the bucket shapes from the
start means the model isn't secretly "bad" at any particular shape.

**Example:**
```
Same prompt, different requested shapes:
  "a mountain landscape" + square bucket    → works
  "a mountain landscape" + wide bucket      → also works, without retraining
```

**Diagram:**
```
   square bucket ─┐
   landscape ─────┤
   wide ──────────┼──► same model, trained across ALL of them
   portrait ──────┤
   tall ──────────┘
```

**Common beginner questions:**
- *Q: Does the model need a totally different brain for each shape?* →
  No — the 2D position-awareness from Phase 5 lets one model handle
  differently-shaped grids of patches without needing a separate copy of
  itself per shape.
- *Q: What if someone asks for an even bigger picture than anything
  trained on?* → A math trick (borrowed directly from a similar
  "handling longer text than trained on" trick in `aarambh-studio`) lets the
  position-awareness stretch to somewhat larger grids than it directly
  saw in training.

---

## Phase 11: Scale-Up to Small/Medium

**Definition:** This phase retrains the same architecture at bigger
sizes (more "brain cells"), producing noticeably better pictures.

**Beginner explanation:**
Phase 8's proof-of-concept used the smallest possible version of the
model (called "Tiny") to prove the idea works quickly and cheaply. This
phase repeats the same training recipe at larger sizes ("Small," then
"Medium") — same ingredients, more of them.

**Why we need it:**
A bigger model generally produces sharper, more prompt-accurate images —
the same "prove it small first, then scale up" philosophy `aarambh-studio`
uses for its own model sizes.

**Example:**
```
Tiny model:   ~40 million learned numbers  → proves the idea works
Medium model: ~350 million learned numbers → noticeably better pictures
```

**Diagram:**
```
   Tiny  ──►  Small  ──►  Medium  ──►  (Large comes much later, Phase 25)
  (prove     (bigger,     (bigger
   it works)  better)      still)
```

**Common beginner questions:**
- *Q: Why not train the biggest size immediately?* → A mistake in a
  small, fast-to-train model is cheap to catch and fix; the same mistake
  in a big, slow-to-train model wastes far more time and free-tier
  compute budget.
- *Q: Does scaling up need any new code?* → No — the exact same MMDiT
  code from Phase 5 is reused, just with bigger numbers plugged into its
  size settings.

---

## Phase 12: Reference-Image Prompting

**Definition:** This phase teaches the model to accept a picture as part
of the prompt, not just words — "make something inspired by THIS image's
style or subject," without any extra training needed per-image.

**Beginner explanation:**
Up to now, the only input was text. This phase adds a second kind of
input: an actual reference picture. The model looks at that picture's
"fingerprint" (from Phase 4) and lets it gently influence the new
picture being drawn, alongside your text prompt — like showing an artist
a photo and saying "something in this vein, but different."

**Why we need it:**
Sometimes words alone can't describe exactly the style or mood you want
— showing an example image is often much faster and more precise than
trying to describe it. And unlike self-learning (Phase 22), this works
instantly on any picture you provide, with no separate training step.

**Example:**
```
Reference image: [a photo in a soft watercolor style]
Text prompt: "a lighthouse at sunset"
                    │
                    ▼
Output: a lighthouse-at-sunset picture, drawn with a similar
        soft watercolor feel to the reference photo
```

**Diagram:**
```
  reference photo → "fingerprint" (Phase 4) ─┐
                                              ├──► blended influence
  text prompt → text understanding (Phase 7) ─┘         │
                                                          ▼
                                                  new picture drawn
```

**Common beginner questions:**
- *Q: Is this the same as editing (Phase 13)?* → No — editing changes a
  SPECIFIC picture you give it. This creates a brand-new picture that's
  merely "inspired by" the reference, with no expectation the output
  looks like the input pixel-for-pixel.
- *Q: Do I need to train anything to use a new reference image?* → No —
  that's the whole point of this phase; it works instantly, unlike the
  self-learning feature (Phase 22), which is for when you want the model
  to permanently remember a subject or style.

---

## Phase 13: Image Editing

**Definition:** This phase teaches the model to take an existing photo
plus a written instruction ("make the sky orange") and produce an edited
version of that SAME photo.

**Beginner explanation:**
Unlike Phase 12 (inspired by, but a new image), editing expects the
output to look like the input photo almost exactly, except for whatever
the instruction asked to change. The trickiest part of teaching this
skill is that there's no giant pile of "before and after" edited-photo
pairs sitting around to train on — so this phase invents its own
training examples using a clever trick: generate two pictures from two
very similar captions (like "a red car" and "a blue car"), forcing them
to share almost all the same layout, so the ONLY real difference between
the two images is the thing the captions disagreed about.

**Why we need it:**
This is one of the most commonly wanted image-AI abilities — "here's my
photo, change this one thing about it" — and it can't be built at all
without solving the "where do we get training examples" problem first.

**Example:**
```
Caption A: "a red car parked on a street"
Caption B: "a blue car parked on a street"
                    │
                    ▼  generated together, sharing everything except the
                       one differing detail
Image A (red car) + Image B (blue car), same street, same composition
                    │
                    ▼
Training example: "given Image A + instruction 'change the car color to
blue' → produce Image B"
```

**Diagram:**
```
  caption A ─┐
             ├──► generated as a matched pair, sharing composition
  caption B ─┘         │
                        ▼
             (source image, instruction, target image)
                        │
                        ▼
             a real training example, with NO human hand-editing needed
```

**Common beginner questions:**
- *Q: Does the model "know" which part of the photo to leave alone?* →
  Yes — during training, the model is only ever graded on the parts of
  the picture that were supposed to change, which is exactly what teaches
  it to leave everything else untouched.
- *Q: Why not just download an existing "photo editing" dataset from the
  internet?* → Because this project's philosophy is building things from
  scratch without depending on external datasets or checkpoints wherever
  reasonably possible — generating the training pairs from the project's
  own already-trained model keeps that promise.

---

*(Continued in Part 2 — Phase 14 onward: masked editing, structural
conditioning, upscaling, safety, quantization, fine-tuning, alignment,
self-learning, evaluation, distillation, scale-up, serving, and release.)*
