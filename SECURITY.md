# Security Policy

## Supported Versions

This project is currently in **pre-implementation** (see `README.md`
for status) — no tagged releases exist yet. Once `v1.0.0` ships, this
table will track which versions receive security fixes:

| Version | Supported |
|---|---|
| `main` (pre-release) | ✅ during active development |
| < 1.0.0 | N/A — not yet released |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security
vulnerabilities.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](../../security) of this repository
2. Click **"Report a vulnerability"**
3. Fill in as much detail as you can — affected crate(s), a
   reproduction case if possible, and the potential impact

This keeps the report private between you and the maintainers until a
fix is ready, and GitHub will walk you through the disclosure process
from there.

## What Counts as a Security Issue Here

Given this project's shape — a source-only release with no bundled
model checkpoints — the categories most relevant are:

- **Memory safety issues** in any `unsafe` code (particularly in
  `aarambh-vision-kernel`'s SIMD/CUDA-prep paths)
- **Dependency vulnerabilities** surfaced via `cargo audit` in any
  workspace crate
- **Malicious input handling** — e.g. a crafted image, prompt, or mask
  input causing a panic, crash, or unexpected resource exhaustion in
  `aarambh-vision-serve`'s HTTP server or the CLI
- **Provenance watermark or safety-filter bypass** — a way to strip
  `aarambh-vision-safety`'s watermarking, or to reliably evade its
  content filtering, that undermines what those components are meant to
  guarantee
- **Supply-chain concerns** — e.g. a workspace dependency being
  compromised or typosquatted

Bugs that only affect output *quality* (a blurry image, a bad edit, a
model that doesn't train well) are **not** security issues — please file
those as regular issues instead.

## Response Expectations

This is a from-scratch, community-driven project without a dedicated
security team, so response times will vary — but every report submitted
through GitHub's private vulnerability reporting will be acknowledged
and looked at. Please be patient, and feel free to follow up on the
report thread if you haven't heard back after a reasonable time.

## Disclosure

Once a fix is ready, we'll coordinate disclosure timing with the
reporter through the same private report thread, then publish a GitHub
Security Advisory crediting the reporter (unless they prefer to remain
anonymous).
