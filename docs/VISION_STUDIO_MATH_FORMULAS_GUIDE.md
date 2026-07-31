# Aarambh-Vision-Studio: The Complete Math Formula Guide

### Every formula we use, explained like you've never seen math notation before

This is the image-generation companion to `aarambh-ai-math-formulas-guide.md`.
It's for someone from a **non-math background** — every formula below is
broken down piece-by-piece before we ever touch a real number.

For every formula, you'll get:
- **What it's called** (definition)
- **How to read the symbols** (a beginner "translation" of the notation)
- **Why we use it in aarambh-vision-studio** (which phase it belongs to)
- **The formula itself**
- **2 fully solved examples**, step by step, with real numbers

---

## How to Read Any Formula (read this first!)

Here's the decoder ring, extended from the `aarambh-ai` guide with a few
symbols specific to images:

| Symbol | Say it as | Meaning |
|---|---|---|
| `Σ` (sigma) | "sum of" | Add up a bunch of things |
| `x` | "x" | Usually the input (here: a clean image latent) |
| `x₀` | "x-naught" | The clean data sample (a real image's latent) |
| `x₁` | "x-one" | The noise sample |
| `xₜ` | "x-sub-t" | The image latent at "noise level" t, somewhere between clean and pure noise |
| `t` | "t" | The timestep / noise level, a number between 0 and 1 |
| `v_θ` | "v-theta" | The model's predicted velocity (θ = "all the model's learnable numbers") |
| `μ` (mu) | "mu" | A learned mean (average) value |
| `σ` (sigma, lowercase) | "sigma" | A learned standard deviation (spread) value |
| `ε` (epsilon) | "epsilon" | A random noise sample, drawn fresh each time |
| `⊙` | "elementwise multiply" | Multiply matching positions together, WITHOUT summing (unlike a dot product) |
| `∇` (nabla) | "gradient of" | The direction that increases something the fastest — used for training updates |
| `‖x‖` | "norm of x" | Roughly: "how big is this whole vector/grid, as one single size number" |
| `‖x‖²` | "norm squared" | The norm, squared — used constantly in loss functions below |
| `[i, j]` | "row i, column j" | A position in a 2D grid — used for patch/pixel coordinates |
| `·` (dot) | "dot product" | Multiply matching pairs of numbers, then add them all up |
| `exp(x)` / `eˣ` | "e to the x" | A way of making numbers grow fast (~2.718 raised to power x) |
| `cos(θ)` | "cosine of theta" | How aligned two directions are, from -1 (opposite) to 1 (identical) |

Just refer back to this table whenever a symbol below looks unfamiliar.

---

## 1. Dot Product & Cosine Similarity

**Definition:** The dot product multiplies two equal-length lists of
numbers pair-by-pair and adds the results into one number. Cosine
similarity is the dot product, adjusted so the result always falls
between -1 and 1, regardless of how "big" the two lists are.

**How to read it:**
```
a · b = a1×b1 + a2×b2 + ... + an×bn

cosine_similarity(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

**Why we use it:** This is the single most-used operation in the entire
model — patch embeddings, attention (§9 below), and CLIP-score all boil
down to dot products. Cosine similarity specifically is what powers
**CLIP-score** (Phase 4's understanding encoder, Phase 23's evaluation
harness) — measuring how well a generated image's "fingerprint" matches
its prompt's "fingerprint," regardless of each fingerprint's raw size.

**Example 1 (plain dot product):**
```
a = [2, 3, 4]
b = [1, 0, 5]

Step 1: multiply matching pairs → 2, 0, 20
Step 2: add them up → 2 + 0 + 20 = 22

Answer: a · b = 22
```

**Example 2 (cosine similarity, like comparing an image and text
embedding):**
```
image_embedding = [3, 4]     (‖a‖ = √(3²+4²) = √25 = 5)
text_embedding   = [4, 3]     (‖b‖ = √(4²+3²) = √25 = 5)

Step 1: dot product → (3×4) + (4×3) = 12 + 12 = 24
Step 2: divide by (‖a‖ × ‖b‖) = 24 / (5 × 5) = 24 / 25 = 0.96

Answer: cosine_similarity = 0.96  (very close to 1 — strong match)
```

**Beginner question:** *Why not just use the raw dot product as
CLIP-score directly?* Because a raw dot product can be large just
because the embeddings themselves happen to be "big" numbers, not
because they actually point in a similar direction — dividing by both
norms removes that size-bias, leaving a pure "how aligned are these two
things" measurement.

---

## 2. Patch Embedding (Patchify)

**Definition:** Patchify cuts an image's latent grid into small square
patches and flattens each patch into one long list of numbers — a
"token" the MMDiT transformer can process, exactly the way a word
becomes a token in `aarambh-ai`.

**How to read it:**
```
latent grid: H × W × C   (height × width × channels)
patch size: p × p

number of patches = (H / p) × (W / p)
each patch's token length = p × p × C
```

**Why we use it:** A transformer (Phase 5's MMDiT blocks) works on
sequences of tokens, not 2D grids directly — patchify is the bridge that
turns a picture into something structurally identical to a sentence of
tokens, which is exactly why the same attention machinery `aarambh-ai`
uses for words works for image patches too.

**Example 1 (a small 4×4 latent, patch size 2×2):**
```
Latent grid is 4×4×1 (16 numbers total, 1 channel for simplicity)
Patch size: 2×2

Number of patches = (4/2) × (4/2) = 2 × 2 = 4 patches
Each patch's token length = 2×2×1 = 4 numbers

So a 4×4×1 grid becomes 4 tokens, each holding 4 numbers.
```

**Example 2 (this project's real Tiny-scale numbers, §5 ARCHITECTURE
Part 1):**
```
Latent grid (from the VAE, 64×64 image): 8×8×16   (H/8=8, W/8=8, C=16)
Patch size: 2×2

Number of patches = (8/2) × (8/2) = 4 × 4 = 16 patches
Each patch's token length = 2×2×16 = 64 numbers

So one 64×64 training image becomes a sequence of 16 tokens,
each a 64-number vector — this is what actually enters the MMDiT.
```

**Beginner question:** *Does patchify lose any information?* No — unlike
the VAE's compression (which does lose a little detail), patchify is a
pure reshape: every number from the latent grid is still present,
just rearranged into a sequence-of-tokens shape instead of a 2D grid
shape. Unpatchify (the reverse operation) recovers the exact original
grid.

---

## 3. KL Divergence (VAE Latent Regularization)

**Definition:** KL (Kullback-Leibler) Divergence measures how different
one probability distribution is from another. In this project, it
measures how different the VAE's learned latent distribution is from a
simple, well-behaved "standard" distribution (mean 0, spread 1).

**How to read it:**
```
KL = 0.5 × (μ² + σ² − log(σ²) − 1)
```
"Take the learned mean, squared, plus the learned spread, squared, minus
the log of the spread squared, minus 1, times a half."

**Why we use it:** Phase 1's Image VAE needs its latent space to be
predictable (roughly centered at 0, roughly unit spread) so that later,
when Phase 8 samples random noise to start the "nudge toward a picture"
process, that noise is compatible with what the VAE actually produces.
KL divergence is the loss term that nudges the VAE's learned latents
toward that well-behaved shape.

**Example 1 (a latent value close to the target shape):**
```
μ = 0.02, σ² = 1.02

KL = 0.5 × (0.02² + 1.02 − log(1.02) − 1)
   = 0.5 × (0.0004 + 1.02 − 0.0198 − 1)
   = 0.5 × 0.0006
   = 0.0003

Answer: KL ≈ 0.0003 (tiny — this latent is already close to ideal)
```

**Example 2 (a latent value further from the target shape):**
```
μ = 0.4, σ² = 0.81

KL = 0.5 × (0.4² + 0.81 − log(0.81) − 1)
   = 0.5 × (0.16 + 0.81 − (−0.2107) − 1)
   = 0.5 × 0.1807
   = 0.0904

Answer: KL ≈ 0.0904 (bigger — this latent is being pushed harder
                       toward the well-behaved shape)
```

**Beginner question:** *Why is the target "mean 0, spread 1" instead of
some other shape?* Because that's the same shape as the random noise
Phase 8/9 samples to start generating a new image — matching the VAE's
real latent distribution to that shape is what makes "start from random
noise" a sensible strategy at all.

---

## 4. The Reparameterization Trick

**Definition:** A specific way of randomly sampling from the VAE's
learned distribution that still allows the training process to compute
gradients (i.e., still allows learning) through that random step.

**How to read it:**
```
z = μ + σ × ε         where ε is drawn fresh from a standard N(0,1) distribution
```
"The actual latent value equals the learned mean, plus the learned
spread times a fresh random number."

**Why we use it:** Phase 1's VAE needs to randomly SAMPLE a latent
value (not just always use the average), but directly sampling
randomly would block the training process from learning — you can't
compute "how should I adjust my numbers" through a step that's pure
randomness. This trick moves the randomness into a separate variable
(`ε`) that doesn't need to be learned, so gradients can still flow
through `μ` and `σ`.

**Example 1:**
```
μ = 0.5, σ = 0.2, ε = 1.3  (a random sample, drawn fresh)

z = 0.5 + 0.2 × 1.3 = 0.5 + 0.26 = 0.76

Answer: z = 0.76
```

**Example 2 (a different random draw, same μ and σ):**
```
μ = 0.5, σ = 0.2, ε = -0.7  (a different random sample)

z = 0.5 + 0.2 × (-0.7) = 0.5 − 0.14 = 0.36

Answer: z = 0.36
```
Notice: same `μ` and `σ`, but a different random `ε` gives a different
final `z` — that's the "sampling" part; the learned `μ`/`σ` control
roughly where and how spread-out those samples land.

**Beginner question:** *Why can't the model just always use μ directly
and skip the randomness?* Using only μ would make the VAE deterministic
and its latent space poorly-behaved for later sampling — the small
built-in randomness during training is part of what makes the latent
space smooth and well-organized enough for Phase 8 to later generate
new, coherent images from fresh random noise.

---

## 5. Rectified Flow Matching Loss

**Definition:** The core training objective for the MMDiT backbone
(Phase 8) — teaches the model to predict which direction ("velocity") to
nudge a noisy image latent, to move it toward a real image.

**How to read it:**
```
xₜ = (1 − t)·x₀ + t·x₁              (the noisy latent at noise-level t)
L_flow = ‖ v_θ(xₜ, t, text) − (x₁ − x₀) ‖²
```
"The noisy latent is a straight-line blend of the clean image and pure
noise. The loss is: how far off was the model's predicted direction from
the actual (noise minus clean-image) direction, squared."

**Why we use it:** This is THE central training signal for the whole
project's drawing ability (Phase 8). It's simpler than older methods
(which predicted noise at many discrete steps) because the target
direction here is a constant straight line for a given (x₀, x₁) pair —
easier to learn, and it's why sampling (Phase 9) needs relatively few
steps.

**Example 1:**
```
x₀ = 2.0 (clean image latent value)
x₁ = -1.0 (noise value)
t = 0.3

Step 1: compute xₜ = (1-0.3)×2.0 + 0.3×(-1.0) = 1.4 − 0.3 = 1.1
Step 2: compute the target velocity = x₁ − x₀ = -1.0 − 2.0 = -3.0
Step 3: say the model predicted v_θ = -2.5
Step 4: loss = (-2.5 − (-3.0))² = (0.5)² = 0.25

Answer: L_flow = 0.25
```

**Example 2 (same pair, later point on the path, t = 0.8):**
```
x₀ = 2.0, x₁ = -1.0, t = 0.8

Step 1: xₜ = (1-0.8)×2.0 + 0.8×(-1.0) = 0.4 − 0.8 = -0.4
Step 2: target velocity = x₁ − x₀ = -3.0    (unchanged! Notice this)
Step 3: say the model predicted v_θ = -2.8
Step 4: loss = (-2.8 − (-3.0))² = (0.2)² = 0.04

Answer: L_flow = 0.04
```

**Beginner question:** *Why doesn't the target velocity change between
the two examples, even though `t` changed?* That's the entire "rectified
flow" idea — for a given pair of (clean image, noise), the straight-line
path between them has exactly one constant direction the whole way. The
model only has to learn WHICH constant direction to move in, not a
direction that changes moment-to-moment — that simplicity is what makes
this approach both easier to train and faster to sample from.

---

## 6. Classifier-Free Guidance (CFG)

**Definition:** A sampling-time trick that exaggerates the difference
between "what the model would draw with your prompt" and "what it would
draw with no prompt at all," making the final output follow your prompt
more strongly.

**How to read it:**
```
v_cfg = v_uncond + guidance_scale × (v_cond − v_uncond)
```
"The final direction equals the no-prompt direction, plus an amplified
version of how much the with-prompt direction differs from it."

**Why we use it:** Phase 9's sampler uses this at every generation step
to make the output more strongly follow your actual prompt than the raw
model would on its own — a bigger `guidance_scale` means a bigger push
toward prompt-following (at some cost to output variety).

**Example 1 (guidance_scale = 2):**
```
v_uncond = 1.0   (direction predicted with no prompt)
v_cond   = 1.6   (direction predicted WITH your prompt)
guidance_scale = 2

v_cfg = 1.0 + 2 × (1.6 − 1.0) = 1.0 + 2×0.6 = 1.0 + 1.2 = 2.2

Answer: v_cfg = 2.2  (further from v_uncond than v_cond alone was)
```

**Example 2 (guidance_scale = 5, a much stronger push):**
```
v_uncond = 1.0, v_cond = 1.6, guidance_scale = 5

v_cfg = 1.0 + 5 × (1.6 − 1.0) = 1.0 + 5×0.6 = 1.0 + 3.0 = 4.0

Answer: v_cfg = 4.0  (a much bigger amplification of the prompt's effect)
```

**Beginner question:** *Does this mean the model runs twice per
sampling step?* Yes — once with your real prompt, once with an empty
prompt (the "null conditioning" from Phase 2) — that's the direct cost
of using CFG, roughly doubling the compute per step in exchange for
noticeably better prompt-following.

---

## 7. 2D Rotary Position Embedding (RoPE)

**Definition:** A technique for encoding a patch's position (row and
column) into its numbers by literally rotating pairs of numbers by an
angle tied to that position — extended, for this project, to work with
2D row/column coordinates instead of a single 1D sequence position.

**How to read it (simplified, one pair of numbers):**
```
For a position p and a pair of numbers (x1, x2):

x1' = x1×cos(p×θ) − x2×sin(p×θ)
x2' = x1×sin(p×θ) + x2×cos(p×θ)
```
"Rotate this pair of numbers by an angle that depends on the position
`p` and a fixed frequency `θ`."

**Why we use it:** Phase 5's MMDiT needs to know WHERE each patch sits
in the 2D image grid (not just that it's "patch #7 in a list"). Rotating
by a position-dependent angle is a clever trick that lets attention
(§9 below) naturally pick up on relative position — "this patch is 2
positions to the right of that one" — without needing a separate
position number bolted on.

**Example 1 (rotating the pair (1, 0) by 90°, i.e. θ×p = 90°, so
cos=0, sin=1):**
```
x1 = 1, x2 = 0
cos(90°) = 0, sin(90°) = 1

x1' = 1×0 − 0×1 = 0
x2' = 1×1 + 0×0 = 1

Answer: (1, 0) rotates to (0, 1) — a quarter-turn, as expected
```

**Example 2 (rotating the pair (2, 1) by 30°, cos(30°)≈0.866,
sin(30°)=0.5):**
```
x1 = 2, x2 = 1
cos(30°) ≈ 0.866, sin(30°) = 0.5

x1' = 2×0.866 − 1×0.5 = 1.732 − 0.5 = 1.232
x2' = 2×0.5 + 1×0.866 = 1.0 + 0.866 = 1.866

Answer: (2, 1) rotates to approximately (1.232, 1.866)
```

**Beginner question:** *Why "2D" specifically for this project?* Text is
naturally a 1D sequence (word 1, word 2, word 3...), but an image patch
lives on a 2D grid (row AND column) — 2D RoPE applies this same rotation
idea independently along both the row and column directions, so the
model has a genuine sense of 2D layout, not just "patch number N in a
flattened list."

---

## 8. AdaLN-Zero Conditioning Modulation

**Definition:** A way for a single "condition" (the timestep, and the
overall prompt meaning) to gently steer every layer of the MMDiT, by
scaling, shifting, and gating that layer's output.

**How to read it:**
```
scale, shift, gate = small_MLP(condition)

output = input + gate × ( layer( normalize(input) × (1 + scale) + shift ) )
```
"A small helper network looks at the condition and produces three
numbers: a scale, a shift, and a gate. The layer's normal output gets
scaled and shifted by those numbers, then multiplied by the gate before
being added back."

**Why we use it:** Phase 5's MMDiT blocks need to behave differently
depending on both the noise level (`t`) and what the prompt means — this
is the mechanism that injects both of those into every single block.
The "Zero" part matters specifically: the gate starts at exactly 0, so
at the very beginning of training, every block does nothing extra (acts
like a plain pass-through) — a stability trick that makes early training
much less likely to blow up.

**Example 1 (gate = 0, at the very start of training):**
```
input = 5.0, layer_output = 3.0 (whatever the inner layer computed)
gate = 0   (this is the "Zero" in AdaLN-Zero — true at initialization)

output = 5.0 + 0 × 3.0 = 5.0 + 0 = 5.0

Answer: output = 5.0 — identical to the input; the block is a no-op
```

**Example 2 (gate = 0.4, after some training has happened):**
```
input = 5.0, layer_output = 3.0
gate = 0.4   (the network has now learned this block should matter some)

output = 5.0 + 0.4 × 3.0 = 5.0 + 1.2 = 6.2

Answer: output = 6.2 — the block now genuinely changes the input
```

**Beginner question:** *Why is starting at "does nothing" a good idea?*
A freshly-initialized deep network with many blocks that ALL do
something significant right away tends to produce chaotic, unstable
early training. Starting every block as a no-op means training begins
from a simple, stable point and gradually "turns on" each block's
influence only as it learns something useful to contribute.

---

## 9. Cross-Attention (Scaled Dot-Product, Text ↔ Image)

**Definition:** The mechanism that lets image patch tokens "look at" text
tokens (and vice versa) and decide which ones are most relevant right
now — the same core idea as `aarambh-ai`'s self-attention, applied here
between two DIFFERENT sequences (image patches and text tokens) instead
of one sequence looking at itself.

**How to read it:**
```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d ) · V
```
"Take the query (from image patches), compare it against every key
(from text tokens) with a dot product, scale it down, turn those scores
into probabilities (softmax), then use those probabilities to blend the
values (also from text tokens) together."

**Why we use it:** This is literally how a prompt actually influences
what gets drawn — every image patch token "asks" (query) which text
tokens (keys) are most relevant to it, and pulls in a weighted blend of
their meaning (values). Phase 8's whole "draw the picture the prompt
describes" ability runs through this mechanism.

**Example 1 (one image-patch query against 2 text-token keys, single
number simplified from real multi-dimension vectors):**
```
Q (image patch) = 2
K (text token "red") = 3, K (text token "table") = 1
scale = √d = √4 = 2 (for a 4-dimensional real vector, simplified here)

raw scores: (2×3)/2 = 3    and    (2×1)/2 = 1
softmax([3, 1]) ≈ [0.88, 0.12]   (turns raw scores into probabilities)

If V("red")=[1,0,0] and V("table")=[0,0,1]:
output = 0.88×[1,0,0] + 0.12×[0,0,1] = [0.88, 0, 0.12]

Answer: the patch mostly "attends to" the word "red" (0.88 weight)
```

**Example 2 (a query more aligned with "table"):**
```
Q (image patch) = 1
K ("red") = 1, K ("table") = 4
scale = 2

raw scores: (1×1)/2 = 0.5    and    (1×4)/2 = 2
softmax([0.5, 2]) ≈ [0.18, 0.82]

output = 0.18×[1,0,0] + 0.82×[0,0,1] = [0.18, 0, 0.82]

Answer: this patch mostly attends to "table" instead (0.82 weight)
```

**Beginner question:** *Why divide by √d before the softmax?* Without
scaling, dot products between long vectors can get very large, which
makes softmax produce near all-or-nothing weightings (almost 1.0 on one
token, almost 0 on everything else) — dividing by √d keeps the scores in
a range where softmax produces a more useful, smoother blend.

---

## 10. GAN Hinge Loss (VAE & Upscale Discriminators)

**Definition:** The loss function used by the "discriminator" —
a component trained to tell apart real photos from the VAE's (or
upscaler's) reconstructed ones, whose feedback pushes the reconstructor
toward sharper, more realistic output.

**How to read it:**
```
Discriminator loss:    L_D = max(0, 1 − D(real)) + max(0, 1 + D(fake))
Generator (VAE) loss:  L_G = −D(fake)
```
"The discriminator wants D(real) ≥ 1 and D(fake) ≤ -1. The generator
(the VAE/upscaler) wants D(fake) to be as large/positive as possible —
i.e., wants to fool the discriminator into scoring fakes highly."

**Why we use it:** Phase 1's VAE (§6.3, ARCHITECTURE Part 1) and Phase
16's upscaler both use this adversarial pressure to sharpen output
detail beyond what a plain reconstruction loss alone achieves — a
discriminator specifically penalizes the kind of subtle blurriness a
pixel-by-pixel loss tends to tolerate.

**Example 1 (discriminator loss, given its own scores):**
```
D(real) = 0.8, D(fake) = -0.3

L_D = max(0, 1 − 0.8) + max(0, 1 + (−0.3))
    = max(0, 0.2) + max(0, 0.7)
    = 0.2 + 0.7
    = 0.9

Answer: L_D = 0.9 (the discriminator still has room to improve on both)
```

**Example 2 (a well-trained discriminator, correctly confident):**
```
D(real) = 1.5, D(fake) = -1.2

L_D = max(0, 1 − 1.5) + max(0, 1 + (−1.2))
    = max(0, −0.5) + max(0, −0.2)
    = 0 + 0
    = 0

Answer: L_D = 0 (the discriminator is already comfortably past the
                 hinge thresholds on both sides — no more loss to pay)
```

**Beginner question:** *Why is it called "hinge" loss?* Because of the
`max(0, ...)` — once a score passes a certain threshold (the "hinge
point"), the loss for that example flattens out at exactly 0 and stops
providing any further push. It only actively penalizes scores that
haven't yet crossed the threshold.

---

## 11. Reconstruction Loss (L1 + Perceptual)

**Definition:** The straightforward part of the VAE's and upscaler's
training — directly comparing a reconstructed image to the real one,
pixel by pixel (L1) and via a learned notion of "does this look similar
to a human" (perceptual/LPIPS-style).

**How to read it:**
```
L1 = (1/n) × Σ |real_pixel − reconstructed_pixel|

L_recon = λ_L1 × L1 + λ_perceptual × L_perceptual
```
"L1 is the average absolute pixel-value difference. The full
reconstruction loss combines that with a perceptual term, each weighted
by its own importance factor."

**Why we use it:** This is the most direct, easy-to-understand part of
Phase 1/Phase 16's training — "does the output actually resemble the
input." It's what Stage 0a of the VAE's staged training (§6.5,
ARCHITECTURE Part 1) relies on entirely, before any adversarial term is
added.

**Example 1 (a tiny 4-pixel image, L1 only):**
```
real          = [100, 150, 200, 50]
reconstructed = [110, 140, 190, 60]

|100-110| + |150-140| + |200-190| + |50-60| = 10+10+10+10 = 40
L1 = 40 / 4 = 10

Answer: L1 = 10 (average pixel-value error of 10, out of a 0-255 range)
```

**Example 2 (a closer reconstruction, same real image):**
```
real          = [100, 150, 200, 50]
reconstructed = [102, 148, 198, 52]

|100-102|+|150-148|+|200-198|+|50-52| = 2+2+2+2 = 8
L1 = 8 / 4 = 2

Answer: L1 = 2 (a much smaller average error — a better reconstruction)
```

**Beginner question:** *Why do we also need a "perceptual" term, if L1
already measures error directly?* Because L1 treats every pixel equally
and can be fooled — a slightly blurry image can have a low L1 error
while still looking noticeably worse to a human than a sharper image
with a similar L1 score. A perceptual term (comparing learned features,
not raw pixels) captures "does this look right to a human" more
faithfully than raw pixel differences alone.

---

## 12. Consistency Distillation Loss

**Definition:** The training objective for Phase 24's fast, few-step
"student" model — teaches it to jump from any point on a noise-to-image
path directly to that path's endpoint in a single leap.

**How to read it:**
```
L_consistency = ‖ student(xₜ, t) − teacher_endpoint(x_{t'}, t') ‖²
```
"Compare the student's one-leap prediction against where the (slower,
multi-step) teacher model eventually ends up, starting from a nearby
point on the same path."

**Why we use it:** Waiting for 20-50 small nudging steps (Phase 9) is
slow; a model trained with this loss learns to predict the FINAL picture
in one or a few leaps instead, dramatically speeding up everyday use —
covered fully in Phase 24 and §14, ARCHITECTURE Part 2.

**Example 1:**
```
student's one-leap prediction = 4.2
teacher's actual multi-step endpoint = 4.5

L_consistency = (4.2 − 4.5)² = (−0.3)² = 0.09

Answer: L_consistency = 0.09
```

**Example 2 (a better-trained student, closer prediction):**
```
student's one-leap prediction = 4.45
teacher's actual multi-step endpoint = 4.5

L_consistency = (4.45 − 4.5)² = (−0.05)² = 0.0025

Answer: L_consistency = 0.0025 (much smaller error — the student has
                                 learned to leap accurately)
```

**Beginner question:** *Why not just train the fast model from scratch
directly on real images, skipping the "teacher" entirely?* Because the
"correct" one-leap jump from a noisy point all the way to a finished
image is a MUCH harder thing to learn directly than the small, careful
nudges the original (teacher) model was trained on — using the teacher's
own already-learned multi-step answer as the target gives the student a
much more learnable signal than raw images alone would.

---

## 13. Gradient Orthogonalization (Self-Learning)

**Definition:** A technique for adjusting a proposed weight update so it
doesn't overlap with (and therefore doesn't interfere with) directions
already "claimed" by things the model previously learned.

**How to read it:**
```
g_orth = g_new − Σᵢ ( (g_new · gᵢ) / (gᵢ · gᵢ) ) × gᵢ
```
"Take the new proposed update. For each previously-protected direction,
subtract out however much the new update overlaps with it. What's left
is the part of the update that's genuinely new."

**Why we use it:** `SELF_LEARNING_VISION_STUDIO.md`'s whole
anti-forgetting design (Phase 22) depends on this — when the system
learns a new subject or style on the fly, this formula is what keeps
that new learning from quietly damaging a previously-learned subject or
style that happens to share some of the same underlying weight
directions.

**Example 1 (a new update that's mostly independent of a protected
direction):**
```
g_new = [3, 1]
g_protected = [0, 1]

Step 1: dot product (g_new · g_protected) = 3×0 + 1×1 = 1
Step 2: (g_protected · g_protected) = 0×0 + 1×1 = 1
Step 3: projection = (1/1) × [0, 1] = [0, 1]
Step 4: g_orth = [3, 1] − [0, 1] = [3, 0]

Answer: g_orth = [3, 0] — the overlapping part (the "1" in the second
                  slot) was removed; only the independent part remains
```

**Example 2 (a new update that heavily overlaps a protected direction):**
```
g_new = [1, 4]
g_protected = [0, 1]

Step 1: (g_new · g_protected) = 1×0 + 4×1 = 4
Step 2: (g_protected · g_protected) = 1
Step 3: projection = (4/1) × [0, 1] = [0, 4]
Step 4: g_orth = [1, 4] − [0, 4] = [1, 0]

Answer: g_orth = [1, 0] — almost the entire "4" was overlapping the
                  protected direction and got removed
```

**Beginner question:** *What happens to the part that gets subtracted
out?* It's simply dropped from this particular update — the model
doesn't move in that already-claimed direction at all for this new
learning event, which is precisely what protects whatever was
previously learned along that direction.

---

## 14. Reward-Weighted Alignment (GRPO / DPO Signal)

**Definition:** The core idea behind Phase 21's alignment step — nudging
the model toward whichever of several sampled outputs scored higher on
automatic quality metrics, and away from whichever scored lower.

**How to read it (a simplified group-relative version):**
```
advantage_i = reward_i − average(all rewards in the group)
```
"How much better (or worse) did this particular sample do, compared to
the average of everything sampled alongside it?"

**Why we use it:** Phase 21's GRPO step uses exactly this kind of
relative comparison — sample several outputs for the same prompt, score
each with Phase 23's metrics (CLIP-score, aesthetic score,
edit-consistency), then nudge the model toward the above-average ones
and away from the below-average ones.

**Example 1 (a group of 4 sampled outputs for one prompt):**
```
rewards = [7.5, 6.0, 8.5, 6.0]
average = (7.5+6.0+8.5+6.0) / 4 = 28 / 4 = 7.0

advantage for the 8.5-scoring sample = 8.5 − 7.0 = 1.5   (nudge toward this)
advantage for a 6.0-scoring sample   = 6.0 − 7.0 = −1.0  (nudge away from this)
```

**Example 2 (a group where everything scored similarly — small
advantages):**
```
rewards = [7.1, 6.9, 7.0, 7.0]
average = (7.1+6.9+7.0+7.0) / 4 = 28.0 / 4 = 7.0

advantage for the 7.1-scoring sample = 7.1 − 7.0 = 0.1  (a very gentle nudge)
advantage for the 6.9-scoring sample = 6.9 − 7.0 = −0.1 (a very gentle nudge)
```

**Beginner question:** *Why compare against the GROUP's average instead
of some fixed target score?* Because a fixed target would need to be
guessed correctly in advance for every possible prompt; comparing
against other samples of the SAME prompt automatically adapts — an
"easy" prompt where everything scores well and a "hard" prompt where
everything scores lower both still produce a meaningful signal about
which specific samples did relatively better or worse.

---

# Quick Reference: Every Formula in One Table

| # | Formula | Used in | What it does |
|---|---|---|---|
| 1 | Dot Product & Cosine Similarity | CLIP-score, everywhere | Measures alignment between two vectors |
| 2 | Patch Embedding (Patchify) | Phase 5, all image processing | Turns an image latent into a token sequence |
| 3 | KL Divergence | Phase 1 (VAE) | Keeps the latent space well-behaved |
| 4 | Reparameterization Trick | Phase 1 (VAE) | Samples randomly while staying trainable |
| 5 | Rectified Flow Matching Loss | Phase 8 (core training) | Teaches the model which direction to nudge noise toward an image |
| 6 | Classifier-Free Guidance | Phase 9 (sampling) | Makes output follow the prompt more strongly |
| 7 | 2D Rotary Position Embedding | Phase 5 | Encodes each patch's row/column position |
| 8 | AdaLN-Zero Modulation | Phase 5 | Lets timestep/prompt steer every MMDiT block, safely |
| 9 | Cross-Attention | Phase 5/8 | Lets image patches "read" the text prompt |
| 10 | GAN Hinge Loss | Phase 1, Phase 16 | Sharpens VAE/upscaler output via adversarial pressure |
| 11 | Reconstruction Loss (L1 + Perceptual) | Phase 1, Phase 16 | Direct "does it look like the input" measurement |
| 12 | Consistency Distillation Loss | Phase 24 | Trains a fast, few-step version of the model |
| 13 | Gradient Orthogonalization | Phase 22 (self-learning) | Prevents new learning from damaging old learning |
| 14 | Reward-Weighted Alignment | Phase 21 | Nudges the model toward higher-scoring outputs |

---

# Frequently Asked "Big Picture" Questions

**Q: Do I need to understand every formula here before working on the
project?**
No — same answer as the `aarambh-ai` guide. Understanding #1 (dot
product), #5 (rectified flow), and #9 (cross-attention) covers the
conceptual core of how the model actually draws a picture from a prompt;
the rest are specific tools used by specific phases, best learned when
you actually reach that phase.

**Q: Which formula is the single most important one to really
understand?**
#5, Rectified Flow Matching Loss — it's the actual training signal
behind the project's core drawing ability (Phase 8). Nearly everything
else either prepares data for it (patchify, RoPE, cross-attention) or
builds on top of a model already trained with it (distillation,
alignment, self-learning).

**Q: Why do so many of these formulas involve squaring a difference
(like `(a − b)²`)?**
Squaring does two useful things at once: it makes every error positive
(so a too-high guess and a too-low guess both count as "wrong" the same
way), and it penalizes bigger errors disproportionately more than small
ones — a difference of 2 becomes 4, but a difference of 10 becomes 100,
which pushes training to prioritize fixing the biggest mistakes first.

**Q: How does this guide connect to
`VISION_AI_DATASET_CREATION_GUIDE.md` and the roadmap/architecture
docs?**
The dataset guide explains how the raw (image, caption) pairs these
formulas operate on get built in the first place; this guide explains
the actual math each phase runs on that data; the ARCHITECTURE and
ROADMAP docs explain how all of it fits together into 28 concrete
build phases.

---

*This guide covers every formula used across aarambh-vision-studio's
phases — the mathematical foundation underneath
`ARCHITECTURE_VISION_STUDIO_PART1/2.md`, `ROADMAP_VISION_STUDIO_
PART1/2.md`, and `SELF_LEARNING_VISION_STUDIO.md`.*
