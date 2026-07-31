# Computer Vision, Generative Image AI & Dataset Creation: The Complete Beginner's Guide

### The image-side companion to `ai-ml-dl-dataset-creation-guide.md`

This guide explains, from zero, everything about how **aarambh-vision-studio**'s
training data actually gets built — starting from the core AI/ML/DL
vocabulary (adapted for pictures instead of words), then walking through
the full practical pipeline of collecting, cleaning, and preparing an
image-text dataset for training a from-scratch image generation model.

No prior AI knowledge assumed.

---

## The Big Picture First

> A generative image model is trained on **pairs**: a picture, and a
> sentence describing that picture. Show it millions of these pairs, and
> it learns the statistical relationship between "words" and "pixels"
> well enough to invent new pixels for words it has never seen paired
> together before.

Everything in this guide is about how those (picture, caption) pairs —
and a few related kinds of training examples this project also needs —
actually get built, cleaned, and made trustworthy before any training
happens.

```
   Raw Photos From The Internet / Self-Generated
                    │
                    ▼
          ┌─────────────────┐
          │   Collection     │  (scraping, public datasets, self-generation)
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Cleaning       │  (corrupt files, bad captions, wrong color mode)
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Deduplication   │  (remove repeated/near-identical photos)
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Filtering      │  (blurry, NSFW, watermarked, low-quality)
          └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Formatting     │  (structure into shards, train/val/test)
          └─────────────────┘
                    │
                    ▼
   Ready for the Data Pipeline (Phase 3) → Training (Phase 8 onward)
```

---

# PART 1 — Understanding the Terms

## 1. Artificial Intelligence (AI)

**Definition:** AI is the broad field of building machines that can
perform tasks which normally require human intelligence — recognizing a
face, understanding a sentence, or drawing a picture from a description.

**Beginner explanation:**
"AI" is the biggest circle in a set of nested ideas. Everything else in
this guide — machine learning, deep learning, computer vision, and image
generation — is a smaller circle living inside this big one.

**Diagram:**
```
┌─────────────────────────────────────────────┐
│  Artificial Intelligence (AI)                  │
│  ┌───────────────────────────────────────┐    │
│  │  Machine Learning (ML)                   │    │
│  │  ┌─────────────────────────────────┐    │    │
│  │  │  Deep Learning (DL)                │    │    │
│  │  │  ┌───────────────────────────┐    │    │    │
│  │  │  │  Computer Vision (CV)        │    │    │    │
│  │  │  │  ┌─────────────────────┐    │    │    │    │
│  │  │  │  │ Generative Image AI   │    │    │    │    │
│  │  │  │  │  (this project)       │    │    │    │    │
│  │  │  │  └─────────────────────┘    │    │    │    │
│  │  │  └───────────────────────────┘    │    │    │
│  │  └─────────────────────────────────┘    │    │
│  └───────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Common beginner questions:**
- *Q: Is every image-editing filter "AI"?* → No — a filter that always
  applies the exact same fixed math (like a simple sharpen or blur) isn't
  AI in this sense; AI specifically means the behavior was *learned* from
  examples, not hand-programmed.

---

## 2. Machine Learning (ML)

**Definition:** ML is a way of building AI where, instead of a person
writing exact rules, a program learns patterns automatically by looking
at many examples.

**Beginner explanation:**
Instead of a programmer writing "if the pixels form a round red shape,
call it an apple," an ML system is shown thousands of photos labeled
"apple" and "not apple" and works out the pattern itself.

**Real-world example:**
```
Traditional programming: pixels → hand-written rules → "apple" or "not apple"
Machine learning:        pixels + many labeled examples → learned pattern
                           → new, never-seen photo → "apple" or "not apple"
```

**Common beginner questions:**
- *Q: Does ML need labeled examples?* → Often, but not always — some ML
  (like Phase 4's contrastive matching-game training) learns from
  (image, caption) pairs that already exist naturally, without a human
  manually labeling "apple/not apple" one by one.

---

## 3. Deep Learning (DL)

**Definition:** DL is a specific style of ML that uses many-layered
artificial neural networks, loosely inspired by how neurons connect in a
brain.

**Beginner explanation:**
"Deep" refers to having many stacked layers, each one learning a
slightly more abstract pattern than the layer before — early layers in
an image model might learn to notice edges and colors, middle layers
learn shapes and textures, and later layers learn whole objects and
scenes.

**Diagram:**
```
  raw pixels
       │
       ▼  layer 1: notices edges/colors
       ▼  layer 2: notices simple shapes
       ▼  layer 3: notices textures
       ▼  layer 4: notices whole objects
       ▼  ...many more layers...
  rich understanding of the picture
```

**Common beginner questions:**
- *Q: Why does "deep" (many layers) matter for images specifically?* →
  A picture's meaning is genuinely layered — pixels build into edges,
  edges build into shapes, shapes build into objects — a deep network
  mirrors that natural hierarchy.

---

## 4. Neural Network

**Definition:** A neural network is the actual mathematical structure
DL is built from — layers of simple mathematical units ("neurons")
connected together, each one doing a small calculation and passing its
result to the next layer.

**Beginner explanation:**
Each "neuron" is really just a small formula: take some numbers in,
multiply them by learned weights, add them up, and pass the result
onward (often through a small non-linear tweak). No single neuron is
smart — the *pattern* of millions of these small calculations working
together is where the "intelligence" actually comes from.

**Common beginner questions:**
- *Q: Is a neural network anything like a real human brain?* → Only
  loosely inspired by it — the actual math is quite different from real
  biological neurons; "neural network" is more a helpful metaphor than a
  literal brain simulation.

---

## 5. Computer Vision (CV)

**Definition:** Computer Vision is the field of AI focused specifically
on getting computers to understand and interpret pictures and video.

**Beginner explanation:**
CV is to pictures what NLP (Natural Language Processing) is to text —
the umbrella field covering everything from "is there a cat in this
photo" to "describe this photo in a sentence" to "draw me a new photo of
a cat."

**Real-world example:**
```
Classic CV task:      "is there a stop sign in this photo?" (yes/no)
Understanding CV task: "write a caption describing this photo" (Phase 4)
Generative CV task:    "draw me a photo matching this caption" (Phase 8)
```

**Common beginner questions:**
- *Q: Is this project doing all of CV, or just one part?* → Just the
  generative + understanding slice — recognizing objects in security
  footage or reading license plates, for example, are CV tasks this
  project doesn't attempt.

---

## 6. Generative Vision Model (Diffusion & Flow Models)

**Definition:** A generative vision model is a neural network trained to
produce entirely new images, rather than just analyzing existing ones.

**Beginner explanation:**
This project's specific technique — rectified flow matching, explained
fully in Phase 8 and `VISION_STUDIO_MATH_FORMULAS_GUIDE.md` — belongs to
a family of methods that start from random noise and gradually "nudge"
it, step by step, into a picture that matches a text description. This
is the current, state-of-the-art approach used by the leading production
image models as of this project's design.

**Diagram:**
```
   random noise → nudge → nudge → nudge → ... → finished picture
                    (guided by your text prompt the whole way)
```

**Common beginner questions:**
- *Q: Is this the same technique as older "AI art" tools from a few
  years ago?* → Related, but meaningfully improved — the underlying
  "start from noise, nudge toward a picture" idea is similar, but the
  specific math (rectified flow) and the transformer-based architecture
  (MMDiT) are newer and more efficient than the original wave of tools.

---

## 7. Generative AI (For Images)

**Definition:** Generative AI is the broader category (spanning text,
images, audio, and more) of AI that creates new content rather than
just classifying or analyzing existing content.

**Beginner explanation:**
`aarambh-studio` is generative AI for text; `aarambh-voice-studio` is
generative AI for audio; `aarambh-vision-studio` is generative AI for
images. All three share a common underlying idea — learn the statistical
patterns in a huge pile of examples, then sample new examples from that
learned pattern — applied to a different kind of content each time.

**Common beginner questions:**
- *Q: Why do text, audio, and image generation share so much
  underlying philosophy?* → Because at a deep level they're all the same
  question — "given a huge pile of real examples, learn to produce
  plausible new ones" — the content type changes, but the fundamental
  approach doesn't.

---

## 8. Types of Machine Learning (As Used in This Project)

**Definition:** Different ML techniques used across the project's
phases, each suited to a different kind of learning problem.

**Beginner explanation, with where each shows up:**
```
Self-supervised learning:
  The "labels" come from the data itself, not a human.
  Used by: the Image VAE (Phase 1) — learns to reconstruct a photo
  it was just shown, no external labels needed.

Contrastive learning:
  Learn by comparing — pull matching pairs together, push
  non-matching pairs apart.
  Used by: Image Understanding (Phase 4) — matching photos to captions.

Supervised learning (caption generation):
  Learn to produce a specific correct output (a caption) for a
  given input (a photo).
  Used by: the captioning half of Phase 4.

Generative modeling (rectified flow matching):
  Learn to transform noise into realistic samples matching a
  condition (a text prompt).
  Used by: the MMDiT backbone (Phase 8) — the project's core capability.

Reinforcement-style alignment (GRPO/DPO):
  Learn from automatic quality SCORES on your own outputs, nudging
  toward higher-scoring behavior.
  Used by: Alignment (Phase 21).
```

**Common beginner questions:**
- *Q: Does the project use just one of these, or all of them?* → All of
  them, at different phases — no single technique does everything; each
  phase uses whichever is the right tool for that specific problem.

---

# PART 2 — Collecting Data & Building an Image-Text Dataset

Now the practical side: how do you actually get real images with
matching captions, and turn them into something this project's training
phases can use?

## 9. What Is an "Image-Text Dataset" (In This Context)?

**Definition:** For this project, a dataset is a large, organized
collection of (image, caption) pairs — plus, for specific phases, a few
related kinds of examples (edit pairs, edge maps) — structured
consistently so the data pipeline (Phase 3) can reliably read it.

**Beginner explanation:**
A folder of random downloaded photos isn't a dataset yet. A dataset
means: every photo has been checked, cleaned, matched with a caption (or
routed through Phase 4's auto-captioner), deduplicated, filtered for
quality and safety, and organized into a consistent file structure with
train/validation/test splits.

**Why it matters:**
Exactly as true for images as for text: the quality of this dataset
matters as much as, or more than, almost any other single factor in how
good the final model turns out. A smaller, clean, well-filtered image
set consistently beats a huge, messy, duplicate-riddled one.

**Diagram (this project's specific dataset lifecycle):**
```
  Raw Photos (scraped / public / self-generated)
          │
          ▼
  ┌─────────────┐
  │  Collection  │
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Cleaning    │  (corrupt files, bad color mode, mismatched captions)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │Deduplication │  (perceptual hashing — catches near-identical photos)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Filtering   │  (blur, resolution, NSFW, watermarks, aesthetics)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Formatting  │  (bucketed shards, train/val/test)
  └─────────────┘
          │
          ▼
  Ready for the Data Pipeline (Phase 3) → Training (Phase 8 onward)
```

---

## 10. Where to Collect Data From (Sources)

### 10a. Web Scraping (Images + Alt-Text)

**Definition:** Automated downloading of images from web pages, along
with any nearby text (alt-text, captions, surrounding article text) that
can serve as a training caption.

**Beginner explanation:**
A scraper visits pages, downloads the actual image files, and also grabs
the HTML `alt` attribute (the text a screen-reader would read aloud for
that image) or nearby descriptive text — this is one of the most common
ways large image-caption datasets have historically been built.

**Example (conceptual scraping flow):**
```python
import requests
from bs4 import BeautifulSoup

response = requests.get("https://example-blog.com/article-1")
soup = BeautifulSoup(response.text, "html.parser")

for img_tag in soup.find_all("img"):
    image_url = img_tag.get("src")
    caption_text = img_tag.get("alt", "")   # the image's alt-text, if present
    # download image_url, pair it with caption_text
```

**Diagram:**
```
  Web Page (HTML)
        │
        ▼
  ┌───────────────┐
  │   Scraper      │  finds <img> tags + their alt-text
  └───────────────┘
        │
        ▼
  (downloaded image file, alt-text caption) pair
```

**Common beginner questions:**
- *Q: Is alt-text always a good caption?* → Not always — some alt-text
  is missing, generic ("image1.jpg"), or unhelpful; Phase 4's
  auto-captioner exists specifically to fill in captions for images that
  arrive without a usable one.
- *Q: Same legality caveat as text scraping?* → Yes, and more so — image
  copyright is generally treated even more strictly than text copyright
  in many jurisdictions; §17 below covers this in depth.

### 10b. Public Image Datasets

**Definition:** Pre-collected, often pre-cleaned, publicly released
collections of images (sometimes with captions already attached) you can
download directly.

**Examples of the kind of public sources that exist:**
```
- Public-domain photo archives (e.g. government/museum-released collections)
- Openly-licensed stock photo collections (specifically the ones marked
  CC0 or equivalent — not all "free" stock sites are actually license-free)
- Openly-licensed image-caption research datasets, where the license
  explicitly permits training use
```

**Diagram:**
```
   Organization does the collection/curation work once
                    │
                    ▼
        Publicly released image dataset
                    │
                    ▼
        You download it directly (faster and safer, license-wise,
        than scraping millions of images yourself)
```

**Common beginner questions:**
- *Q: Are all "public" image datasets safe to train on?* → No — check
  the specific license of each one individually; some research datasets
  are released for research-only use, not for training models that will
  be shipped or used commercially.

### 10c. APIs

**Definition:** Structured, official access to a service's image content
(e.g. a stock-photo provider's API), in a clean, ready-to-use format.

**Example:**
```python
import requests

response = requests.get(
    "https://api.example-stock-photos.com/v1/search",
    params={"query": "mountain landscape", "license": "cc0", "limit": 100}
)
data = response.json()

for photo in data["results"]:
    image_url = photo["download_url"]
    caption = photo.get("description", "")
```

**Common beginner questions:**
- *Q: Why prefer an API over scraping when both are available?* → An
  API usually makes licensing terms explicit and machine-readable (like
  a `license` field in the response), which makes it much easier to
  filter for only permissively-licensed images automatically.

### 10d. Self-Generated / Synthetic Data — **specific to this project**

**Definition:** Training examples the project generates itself, using
its own already-trained model, rather than collecting them from an
external source at all.

**Beginner explanation:**
This is how Phase 13 (Image Editing) solves a problem that has no good
external-data answer: there's no giant pile of "before and after" edited
photo pairs sitting around online. Instead, once the base text-to-image
model (Phase 11) exists, it's used to generate two closely-related
images from two similar captions sharing the same layout — the
difference between them becomes a real, usable "before/after" edit
training example, with zero external data collection required.

**Example:**
```
Caption A: "a red car parked on a street"
Caption B: "a blue car parked on a street"
                    │
                    ▼  generated together, sharing composition
(Image A, "change the car color to blue", Image B)
                    │
                    ▼
A genuine editing-training example, self-generated, no download needed
```

**Diagram:**
```
   your own trained model
            │
            ▼  generate matched pairs from related captions
   synthetic training examples
   (used for editing, Phase 13; not needed for the base
    text-to-image training, which relies on §10a-10c instead)
```

**Common beginner questions:**
- *Q: Doesn't this create a "the student teaching itself" problem?* →
  It's specifically used only where it's structurally sound — the model
  isn't asked to invent brand-new visual knowledge, only to generate two
  variations of what it already knows how to draw, which is a much
  safer use of self-generated data than trying to bootstrap the entire
  dataset this way.
- *Q: Could this also work for structural conditioning (Phase 15)?* →
  That phase actually uses an even simpler, fully non-learned method —
  extracting outlines mathematically from real photos — covered in
  §10a-10b's collected images directly, no model-generation needed.

---

## 11. Cleaning the Collected Data

**Definition:** Removing broken files and fixing inconsistent formats
across the collected image set before it's usable.

**Beginner explanation:**
Raw collected images have real-world problems: some files are corrupted
and won't open, some are in the wrong color mode (grayscale when the
pipeline expects full color, or have a transparency channel the model
doesn't need), some captions are empty or nonsensical, and some images
are absurdly small or absurdly large.

**Example (conceptual cleaning checks):**
```python
from PIL import Image

def is_clean(path):
    try:
        img = Image.open(path)
        img.verify()  # catches corrupted/truncated files
    except Exception:
        return False
    if img.mode not in ("RGB", "RGBA"):
        return False   # skip grayscale/CMYK/paletted images
    width, height = img.size
    if width < 64 or height < 64:
        return False   # too small to be useful
    return True
```

**Diagram:**
```
   raw collected image + caption
              │
              ▼
   ┌─────────────────────┐
   │  file opens correctly?│
   │  right color mode?     │
   │  reasonable size?       │
   │  caption non-empty?      │
   └─────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
      keep        discard
```

**Common beginner questions:**
- *Q: Why does color mode matter so much?* → Every downstream phase
  (especially the VAE, Phase 1) expects a consistent number of color
  channels; mixing grayscale and full-color images without converting
  them first would break shape assumptions throughout the whole pipeline.

---

## 12. Deduplication

**Definition:** Removing exact and near-identical repeated images from
the collected set.

**Beginner explanation:**
Text deduplication (covered in `ai-ml-dl-dataset-creation-guide.md`)
typically compares exact or near-exact text strings. Images need a
different tool: **perceptual hashing** — a technique that produces a
short fingerprint of an image's overall visual appearance, such that two
images that look almost the same (even if resized, slightly recompressed,
or lightly cropped) get very similar fingerprints, while genuinely
different images get very different fingerprints.

**Example (conceptual perceptual hashing):**
```python
import imagehash
from PIL import Image

hash_a = imagehash.phash(Image.open("photo_a.jpg"))
hash_b = imagehash.phash(Image.open("photo_b.jpg"))

difference = hash_a - hash_b   # small number = visually very similar
if difference < 5:
    print("likely a near-duplicate, consider dropping one")
```

**Diagram:**
```
   photo A ──► perceptual hash ──► fingerprint A
   photo B ──► perceptual hash ──► fingerprint B
                                          │
                                          ▼
                          compare fingerprints
                                          │
                            ┌─────────────┴────────────┐
                            ▼                            ▼
                    very similar                    very different
                    (near-duplicate,                (genuinely
                     drop one)                        different photos)
```

**Common beginner questions:**
- *Q: Why does near-duplication matter for images specifically?* → Stock
  photo sites and social media often host many slightly-different crops
  or recompressions of the exact same underlying photo — without
  perceptual-hash deduplication, the model would see the "same" image
  many times, wasting training capacity and skewing what it thinks is
  common.
- *Q: Isn't exact-file deduplication (matching identical files) enough?*
  → No — a photo saved as both a JPEG and a PNG, or resized slightly, is
  a different file byte-for-byte but visually near-identical; only
  perceptual hashing catches that.

---

## 13. Filtering (Quality & Safety)

**Definition:** Removing images that are technically low-quality, or
that violate content-safety policy, before they reach training.

**Beginner explanation, by filter type:**
```
Blur/quality filtering:
  Measure sharpness (e.g. via Laplacian variance — a genuinely useful
  quick math trick: sharp images have high-variance edges, blurry
  images have low-variance edges). Drop images below a threshold.

Resolution filtering:
  Drop images too small to be useful at the target training resolution
  (echoes Phase 3's bucket-dropping rule for undersized images).

NSFW/content-safety filtering:
  A classifier screens out sexual, violent, or otherwise policy-
  violating imagery before it ever reaches the training set — this is
  a dataset-level filter, separate from (and in addition to) the
  runtime safety filter built in Phase 18.

Watermark detection:
  Many stock/scraped photos carry visible watermarks; a detector flags
  these so they can be excluded — training on heavily-watermarked
  images teaches the model to reproduce watermark-like artifacts.

Aesthetic filtering (optional, later in the project):
  Once Phase 21's learned aesthetic-quality predictor exists, it can
  also be used to deprioritize (not necessarily fully exclude) lower-
  scoring images in the training mix.
```

**Diagram:**
```
   cleaned, deduplicated image
              │
              ▼
   ┌─────────────────────┐
   │  sharp enough?         │
   │  big enough?            │
   │  policy-safe?            │
   │  no visible watermark?    │
   └─────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
      keep        discard
```

**Common beginner questions:**
- *Q: Is dataset-level NSFW filtering enough on its own?* → No — it's
  the first of two layers; Phase 18's runtime safety filter is the
  second, independent layer that also screens prompts and outputs at
  generation time. Neither layer alone is treated as sufficient.
- *Q: Why exclude watermarked images instead of just cropping the
  watermark out?* → Cropping only removes a visible corner mark;
  training on the rest of a heavily-watermarked photo can still teach
  subtle artifacts, so full exclusion is the safer default.

---

## 14. Handling Personal Information — Faces & Privacy

**Definition:** The image-specific equivalent of text's PII (Personally
Identifiable Information) scrubbing — deciding how to handle real,
identifiable human faces appearing in collected photos.

**Beginner explanation:**
Text datasets scrub names, emails, and phone numbers. Image datasets
have a parallel — but visual — privacy concern: photos of real,
identifiable people. This project's scope (§30, ARCHITECTURE Part 2)
explicitly excludes real-person likeness cloning as a capability, and
that boundary starts here, at the dataset level, not just at the
model-capability level.

**Example (conceptual face-handling policy):**
```
Photos containing recognizable faces of real people:
  Option A: exclude entirely from training
  Option B: include only for general "a person exists in this scene"
            learning, with face regions blurred/obscured so the model
            never learns to reproduce any specific real person's likeness
```

**Diagram:**
```
   collected photo
          │
          ▼
   ┌─────────────────┐
   │ face detection    │
   └─────────────────┘
          │
     ┌────┴────┐
     ▼         ▼
  no face    face detected
     │            │
     ▼            ▼
   keep      exclude, or blur the face
             region before including
```

**Common beginner questions:**
- *Q: Doesn't the model need SOME photos of people to draw people at
  all?* → Yes — the policy isn't "exclude all humans," it's "don't let
  the model learn any SPECIFIC real person's identifiable likeness,"
  which is why face-blurring (rather than blanket exclusion of every
  photo containing a person) is often the more practical middle ground.
- *Q: Does this connect to the project's watermarking feature (Phase
  18)?* → Different purpose — watermarking marks AI-generated OUTPUT as
  AI-generated; this section is about protecting real people's privacy
  in the training INPUT. Both exist, for different reasons.

---

## 15. Formatting the Final Dataset

**Definition:** Structuring the cleaned, filtered, deduplicated images
and their captions into a consistent file layout the data pipeline
(Phase 3) can load efficiently.

**Beginner explanation:**
Loading millions of tiny individual image files directly off disk is
slow. Most large-scale image training instead groups many (image,
caption) pairs into bundled "shard" files, so the training pipeline can
read a big sequential chunk at once instead of opening millions of tiny
files individually.

**Example (a shard-friendly layout, and the alternative simple layout):**
```
Shard-based layout (preferred at scale):
  shard_00001.tar  → contains photo1.jpg + photo1.json (caption + metadata)
                              photo2.jpg + photo2.json
                              ... (thousands of pairs per shard)
  shard_00002.tar  → next batch of pairs
  ...

Simple layout (fine for smaller Tiny/Small-scale experiments):
  images/photo1.jpg
  images/photo2.jpg
  captions.jsonl → {"file": "photo1.jpg", "caption": "a red apple on a table"}
                    {"file": "photo2.jpg", "caption": "a mountain at sunset"}
```

**What gets stored, per project-specific need:**
```
Base generation training (Phase 8):     (image, caption) pairs
Editing training (Phase 13):            (source image, instruction, target image)
Masked editing (Phase 14):              above + a mask image
Structural conditioning (Phase 15):     (image, caption, extracted edge map)
Reference-prompting (Phase 12):         reuses the base (image, caption) pairs directly
```

**Diagram:**
```
   cleaned, filtered pairs
            │
            ▼
   ┌─────────────────────┐
   │  bundle into shards    │  (or simple folder+jsonl for smaller experiments)
   └─────────────────────┘
            │
            ▼
   Ready for aarambh-vision-data's loader (Phase 3)
```

**Common beginner questions:**
- *Q: Does every phase need its own separate dataset format?* → No —
  most phases reuse the same base (image, caption) shards; only editing
  and structural conditioning need extra fields (an instruction/target,
  or an edge map), layered on top of the same underlying pairs.
- *Q: When should I use the simple folder layout instead of shards?* →
  For Tiny/Small-scale experiments (i3, small curated sets) the simple
  layout is easier to inspect and debug; shard-based bundling matters
  more once the dataset is large enough that file-count itself becomes
  a loading bottleneck.

---

## 16. Splitting the Dataset (Train / Validation / Test)

**Definition:** Dividing the final cleaned dataset into separate,
non-overlapping portions — one for training, one for checking progress
during training, and one held back for final, honest evaluation.

**Beginner explanation:**
Same core idea as the text version of this guide, with one image-
specific wrinkle: splitting should respect Phase 3's resolution buckets,
so the validation and test sets aren't accidentally skewed toward only
one aspect ratio.

**Example (a typical split):**
```
Training set:    ~90% of the data → what the model actually learns from
Validation set:  ~5%  of the data → checked periodically during training
                                     to catch problems early
Test set:        ~5%  of the data → touched ONLY at the very end, for a
                                     final, honest quality check
```

**Diagram:**
```
   full cleaned dataset (all 5 buckets represented)
                │
                ▼
   ┌──────────┬───────────┬──────────┐
   │  Train    │ Validation │   Test    │
   │  (~90%)   │   (~5%)    │  (~5%)    │
   └──────────┴───────────┴──────────┘
     each split keeps the same bucket proportions as the full set
```

**Common beginner questions:**
- *Q: Why does bucket-balance matter for the splits specifically?* → If,
  by chance, the test set ended up almost entirely square-bucket images,
  a real weakness in the model's wide or tall-bucket generation could go
  completely unnoticed until real-world use.
- *Q: Can the test set be reused across every phase?* → The same held-
  out test images are reused wherever it makes sense (e.g. for both base
  generation and later alignment evaluation), but phase-specific test
  sets (like a held-out editing-pairs set for Phase 13) are kept
  separate, since they're testing a different capability.

---

## 17. Data Licensing & Ethics (Image-Specific)

**Definition:** Understanding the legal rights and ethical considerations
that apply to collected images specifically — a domain where the
questions are, in real and current ways, more actively contested than
for text data.

**Beginner explanation:**
Everything the text-side guide says about licenses (public domain, CC
licenses, all-rights-reserved, site Terms of Service) applies equally to
images — but images carry two additional live considerations worth
naming directly:

```
Artist consent & style:
  There's genuine, ongoing public debate about whether training on an
  artist's publicly-posted work — even when technically legal under
  a site's Terms of Service — is ethically fair to that artist,
  especially regarding learning to imitate a specific living artist's
  recognizable style. This project's fine-tuning/self-learning features
  (Phases 20, 22) are scoped to styles and subjects the person using
  the system has genuine rights to (their own art, their own
  characters/products), not to reproducing a specific living artist's
  style from scraped examples without consent.

Opt-out registries & takedown mechanisms:
  Some image platforms and individual artists now provide explicit
  "do not use my content for AI training" opt-out signals (e.g. a
  robots.txt-style flag, or a dedicated opt-out registry). Respecting
  these signals during collection (§10a-10c) is treated as a hard
  requirement, not an optional nicety.
```

**Example:**
```
License/consent types you'll encounter for images specifically:
  - Public Domain: no restrictions (e.g. very old artwork, some
    government/institutional photo archives)
  - CC0 / CC-BY: open licenses, often requiring attribution
  - Explicitly AI-training-permitted stock libraries: increasingly common,
    explicitly labeled
  - All Rights Reserved: standard copyright, no reuse without permission
  - Opt-out flagged: technically may be otherwise accessible, but the
    creator has explicitly asked for AI-training exclusion — respected
    regardless of the site's general Terms of Service
```

**Diagram:**
```
   Found an image online
             │
             ▼
   ┌──────────────────────────┐
   │ Check license/ToS AND       │
   │ check for an opt-out flag     │
   └──────────────────────────┘
             │
        ┌────┴─────┐
        ▼           ▼
   Permitted,    Not permitted, OR
   no opt-out    opt-out flagged, OR
   flag              unclear
        │             │
        ▼             ▼
   Use it,      Don't use it
   with proper
   attribution
   if required
```

**Common beginner questions:**
- *Q: Is this stricter than the text-data guide's approach?* → In
  practice, yes — image licensing/consent is a more actively contested
  space right now, so this project treats opt-out signals as binding
  even where a site's general Terms of Service might technically permit
  scraping.
- *Q: Does this affect the self-generated data described in §10d?* →
  No — data the project generates from its own already-trained model
  (editing pairs, for instance) doesn't carry third-party licensing
  questions the way collected external images do; the licensing concern
  in this section is specifically about content collected from outside
  sources.

---

# Quick Reference: Full Image Dataset-Building Pipeline in One Table

| Step | What happens | Why it matters |
|---|---|---|
| 1. Collection | Scrape image+alt-text, use APIs, download public datasets, self-generate (editing pairs) | Get the raw material to work with |
| 2. Cleaning | Fix corrupt files, wrong color mode, mismatched captions | Remove junk before it teaches bad habits |
| 3. Deduplication | Perceptual hashing to catch near-identical photos | Avoid wasted training time & skewed frequency |
| 4. Filtering | Blur, resolution, NSFW, watermark, aesthetic checks | Improve overall data quality & safety |
| 5. Privacy handling | Face detection, exclude or blur real identifiable people | Protect privacy, respect the project's likeness-cloning boundary |
| 6. Formatting | Bundle into shards (or simple folder+jsonl), bucket-tagged | Make data readable and efficient for Phase 3's pipeline |
| 7. Splitting | Train / validation / test, bucket-balanced | Enable honest measurement of model quality |
| 8. Licensing & consent check | Confirm rights AND check opt-out flags | Avoid legal risk, respect artists' and creators' choices |

---

# Frequently Asked "Big Picture" Questions

**Q: Which matters more for a good image model — more photos, or
cleaner photos?**
Same answer as the text-side guide: a smaller, well-cleaned, well-
filtered set consistently outperforms a much larger but messy,
duplicate-riddled, low-quality one.

**Q: How many images does this project actually need?**
It scales with the model size being trained (Phase 5's ModelConfig) —
Tiny-scale proof-of-concept training needs far less than Large-scale
(Phase 25) training. There's no single right number; ARCHITECTURE Part 1
§5's model-scale table and each phase's own "Data Setup" step are the
concrete guide for how much is needed at each stage.

**Q: Can I use images generated by OTHER AI image models as training
data for this project?**
Similar to the text-side guide's answer on this: it's a genuinely
contested area, both legally and in terms of what it does to the
model's own learned style (training heavily on another model's outputs
can cause a model to inherit that other model's specific quirks and
artifacts). This project's own self-generated data (§10d) is scoped
narrowly and deliberately (editing pairs, generated from its OWN
already-trained model) specifically to avoid this concern — that's
different from broadly training on a different AI system's outputs.

**Q: Is dataset-building a one-time step for this project?**
No — iterative, same as the text-side answer. An initial dataset gets
the base model (Phase 8) working; the evaluation harness (Phase 23) then
reveals specific weaknesses (a bucket shape that looks worse, an editing
instruction type that fails more often), which sends you back to collect
or filter more targeted data for that specific gap.

**Q: How does all of this connect back to aarambh-vision-studio's
phases?**
Everything in Part 2 above happens *before* Phase 3 (Data Pipeline) even
finishes being built, and continues to feed it throughout the project —
Phase 3 is the "raw material intake" system this guide's output flows
into, which then feeds Phase 8's training loop, gets scored by Phase
23's evaluation harness, and any gaps found there send you back to
targeted re-collection.

---

*This guide covers the foundational AI/ML/DL/CV terminology and the full
practical process of collecting, cleaning, and preparing an image-text
dataset — the essential groundwork that every phase in
`ROADMAP_VISION_STUDIO_PART1/2.md` and every formula in
`VISION_STUDIO_MATH_FORMULAS_GUIDE.md` is built on top of.*
