# Contributing to aarambh-vision-studio

Thanks for your interest in contributing! This project is built entirely
from scratch in Rust, phase by phase, following
[`ROADMAP_VISION_STUDIO_PART1.md`](./ROADMAP_VISION_STUDIO_PART1.md) /
[`PART2.md`](./ROADMAP_VISION_STUDIO_PART2.md). Reading the roadmap and
[`ARCHITECTURE_VISION_STUDIO_PART1.md`](./ARCHITECTURE_VISION_STUDIO_PART1.md) /
[`PART2.md`](./ARCHITECTURE_VISION_STUDIO_PART2.md) before opening a PR
will save you time — most design decisions are already made and
documented there, with the reasoning behind them.

---

## Ground rules

- **No vendored checkpoints, no bindings to PyTorch or other Python
  inference frameworks.** Everything is implemented in Rust on `candle`.
  This is the whole point of the project — PRs that route around it
  (e.g. shelling out to a Python script) won't be merged.
- **Runs on modest hardware.** Every phase is designed to work on a
  regular laptop CPU plus free-tier Kaggle GPU sessions (T4/P100). If
  your change assumes paid or multi-GPU compute, flag that explicitly in
  the PR — it likely needs a design discussion first, not just a code
  review.
- **One crate, one responsibility.** See `ARCHITECTURE_VISION_STUDIO_PART1.md`
  §4 for the full 25-crate layout. New functionality goes in the crate
  whose job it matches — if nothing fits, that's worth raising as an
  issue before writing code, since it may mean a new crate is needed.
- **Phases are built in order.** The roadmap's phase ordering isn't
  arbitrary — later phases depend on earlier ones being frozen/stable
  (see each phase's "Why This Order" reasoning in the roadmap). PRs that
  jump ahead of the current phase are welcome as drafts/discussion, but
  won't be merged until their dependencies land.

---

## Getting set up

```bash
git clone https://github.com/AarambhDevHub/aarambh-vision-studio
cd aarambh-vision-studio
cargo check --workspace
cargo test --workspace
```

You'll need a recent stable Rust toolchain (`rustup update stable`).

For anything beyond Tiny-scale training or CPU-only work, you'll want a
free [Kaggle](https://www.kaggle.com) account for T4/P100 GPU sessions —
see each phase's "Hardware" line in the roadmap for what's actually
needed at that phase.

---

## Making a change

1. **Check which phase your change belongs to.** Look at the Phase Map
   in `ROADMAP_VISION_STUDIO_PART1/2.md`. If it's not obviously covered
   by an existing phase, open an issue first to discuss where it fits.
2. **Read that phase's Goal, Tasks, and Tests** in the roadmap before
   writing code — the task list is deliberately specific about which
   crate and which file each piece of work belongs in.
3. **Write tests as you go**, matching the style already in the roadmap
   (`#[test] fn descriptive_name_of_what_is_being_checked()`), not just
   at the end.
4. **Run the full check before opening a PR:**
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
5. **Open the PR** using the provided template — it asks which phase
   the change belongs to and which of that phase's tasks it completes.

---

## Commit messages

Keep them descriptive and scoped to one crate/phase where possible:

```
aarambh-vision-tokenizer: implement Stage 0a reconstruction-only VAE loop

Phase 1, Stage 0a per ARCHITECTURE_VISION_STUDIO_PART1.md §6.5.
No adversarial/KL loss yet — isolates encoder/decoder correctness
from GAN-stability risk.
```

---

## Documentation changes

If a code change means an architecture or roadmap doc is now
inaccurate, update the doc in the **same PR** — the docs are meant to
track the real, shipped state of the project, not a historical plan.
This includes updating a file's own `description`/summary text if the
change makes it misleading.

---

## Reporting bugs / requesting features

Use the issue templates — they ask for the minimum context needed to
act on a report (which crate, which phase, repro steps for bugs;
motivation and where it'd fit in the roadmap for feature requests).

For security vulnerabilities, **do not open a public issue** — see
[`SECURITY.md`](./SECURITY.md) instead.

---

## Code of Conduct

This project follows [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).
Participation in this project means agreeing to abide by it.

---

## License

By contributing, you agree your contributions are licensed under the
same dual MIT OR Apache-2.0 terms as the rest of the project (see
[`README.md`](./README.md#license)).
