# Changelog

All notable changes to `aarambh-vision-studio` will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once tagged releases begin.

## [Unreleased]

### Added

Phase 0 complete — workspace + core types ([`ROADMAP_VISION_STUDIO_PART1.md`](./ROADMAP_VISION_STUDIO_PART1.md)):

- Cargo workspace: 24 library crates + 1 binary (`aarambh-vision-studio`), all 25 crates present
- `aarambh-vision-core` implemented:
  - `ModelConfig` with `tiny`/`small`/`medium`/`large` scales
  - `DrishtiRequest` + supporting types (stubbed, fully wired by Phase 17)
  - `AarambhVisionError` via `thiserror`
- `.gitignore` (renamed from `gitignore`)
- CI now runs `cargo check`/`test` against the real workspace

### Planned

The remaining 27 phases (Phase 1 → Phase 27) follow the plan in
[`ROADMAP_VISION_STUDIO_PART1.md`](./ROADMAP_VISION_STUDIO_PART1.md)
/ [`PART2.md`](./ROADMAP_VISION_STUDIO_PART2.md), each milestone tagged
`v0.1.0-phaseN` per the roadmap's tagging convention, culminating in
`v1.0.0`.

<!--
Entry template for when phases start shipping:

## [v0.1.0-phase0] - YYYY-MM-DD
### Added
- Workspace scaffold: 25 crates, `aarambh-vision-core` fully implemented
-->
