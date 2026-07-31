//! `src/request.rs` — the full control-surface types for aarambh-vision-studio.
//!
//! Every parameter the system accepts is an explicit typed field on
//! [`DrishtiRequest`] — no hidden presets, no magic strings. The per-engine
//! substructs (`ImageRef`, `StructuralRef`, `MaskRef`, `AdapterRef`) are
//! stubs here and will be fleshed out with full fields in Phase 17
//! (ARCHITECTURE_VISION_STUDIO_PART2.md §23).
//!
//! # Request flow
//!
//! ```text
//! DrishtiRequest
//!   ├── prompt           (the text to generate / edit toward)
//!   ├── mode             (Generate / Edit / Inpaint / Upscale)
//!   ├── reference_image  (editing source OR reference-prompt image)
//!   ├── structural_map   (edge / depth / pose conditioning)
//!   ├── mask             (inpainting region)
//!   ├── resolution       (output size)
//!   ├── solver           (Euler / Heun / Distilled)
//!   ├── output_format    (PNG / JPEG / WebP)
//!   └── style_adapter    (LoRA or self-learned style / subject)
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DrishtiMode
// ---------------------------------------------------------------------------

/// The four top-level capabilities the system can execute.
///
/// Selected via [`DrishtiRequest::mode`]. Each mode routes to a different
/// engine combination; [`ReferenceMode`] further disambiguates how
/// [`DrishtiRequest::reference_image`] is used when `Edit` is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrishtiMode {
    /// Text-to-image: generate a new image from [`DrishtiRequest::prompt`].
    Generate,
    /// Instruction-based editing of [`DrishtiRequest::reference_image`].
    Edit,
    /// Masked / inpainting editing inside [`DrishtiRequest::mask`].
    Inpaint,
    /// Super-resolution refinement of [`DrishtiRequest::reference_image`].
    Upscale,
}

// ---------------------------------------------------------------------------
// ReferenceMode
// ---------------------------------------------------------------------------

/// How [`DrishtiRequest::reference_image`] is interpreted.
///
/// Disambiguates Phase 12's reference-image prompting (a reference photo
/// acts as a style/subject prompt) from Phase 13's editing (a reference
/// photo is the image being edited).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceMode {
    /// The reference image is the editing source (instruction editing).
    Edit,
    /// The reference image is a style/subject prompt (zero-shot, no training).
    Prompt,
}

// ---------------------------------------------------------------------------
// ImageRef & MaskRef
// ---------------------------------------------------------------------------

/// A reference to an on-disk image used as input conditioning.
///
/// Used either as the editing source / upscale input, or as the
/// reference-prompt image, depending on [`DrishtiRequest::reference_mode`].
///
/// # Future
///
/// Phase 17 will extend this to also accept in-memory image data so the
/// HTTP server (`aarambh-vision-serve`) can pass decoded pixels directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRef {
    /// Path to the image file on disk.
    pub path: PathBuf,
}

/// A reference to an on-disk binary mask for inpainting.
///
/// A white pixel marks the region to regenerate; a black pixel marks the
/// region to keep. Consumed by the masked-editing engine in Phase 14.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskRef {
    /// Path to the mask image file on disk.
    pub path: PathBuf,
}

// ---------------------------------------------------------------------------
// StructuralKind & StructuralRef
// ---------------------------------------------------------------------------

/// The type of structural map used for spatial conditioning.
///
/// All three map types share the same conditioning branch (Phase 15); the
/// kind only changes the preprocessing applied before the branch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralKind {
    /// A binary edge map extracted from the source image (Canny-style).
    Edge,
    /// A grayscale depth map (darker = closer to camera).
    Depth,
    /// A pose skeleton map (joints and limbs).
    Pose,
}

/// A reference to an on-disk structural map plus its [`StructuralKind`].
///
/// When `structural_map` is `Some`, the prompt controls content and style
/// while the map guides composition (§12, ARCHITECTURE_VISION_STUDIO_PART2.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralRef {
    /// Path to the map image file on disk.
    pub map_path: PathBuf,
    /// Which preprocessing / conditioning branch the map feeds into.
    pub kind: StructuralKind,
}

// ---------------------------------------------------------------------------
// AdapterRef
// ---------------------------------------------------------------------------

/// A reference to a learned style or subject adapter.
///
/// Adapters are small-rank LoRA-style parameter deltas produced by
/// `aarambh-vision-finetune` or by the self-learning subsystem, stored as
/// named entries and referenced by [`id`](Self::id). The base model's
/// weights are never modified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRef {
    /// The adapter's identifier, as stored by the self-learning memory
    /// or the fine-tuning pipeline.
    pub id: String,
}

// ---------------------------------------------------------------------------
// SolverKind
// ---------------------------------------------------------------------------

/// The ODE solver used by the sampling engine.
///
/// Rectified-flow models can be sampled with different solver orders and
/// step counts; `Distilled` selects the few-step distillation path built
/// in Phase 24.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverKind {
    /// First-order Euler steps (fast, the default).
    Euler,
    /// Second-order Heun steps (more accurate per step).
    Heun,
    /// The distilled few-step sampler (1 / 2 / 4 steps, Phase 24).
    Distilled,
}

// ---------------------------------------------------------------------------
// ImageOutputFormat
// ---------------------------------------------------------------------------

/// The container / codec format for generated image output.
///
/// The exact encoder implementation lives in `aarambh-vision-kernel` /
/// `aarambh-vision-serve` and uses the image-format libraries specified
/// in ARCHITECTURE_VISION_STUDIO_PART1.md §3.
///
/// # Feature gates
///
/// | Variant | Library | Default? | Notes |
/// |---------|---------|----------|-------|
/// | `Png`  | `image` | yes | Lossless, always available |
/// | `Jpeg` | `image` | yes | Lossy, quality 0–100 |
/// | `WebP` | `webp` | behind `webp` cargo feature | Modern lossy format, feature-gated |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ImageOutputFormat {
    /// Lossless PNG, the default output format.
    #[default]
    Png,
    /// Lossy JPEG with the given quality (0–100).
    Jpeg { quality: u8 },
    /// Lossy WebP with the given quality (0–100), behind the `webp` feature.
    WebP { quality: u8 },
}

// ---------------------------------------------------------------------------
// DrishtiRequest
// ---------------------------------------------------------------------------

/// The single top-level request type for every engine in the system.
///
/// Every generation begins with a [`DrishtiRequest`] — there is no second
/// entry path. The request specifies exactly what to generate and how to
/// format the result. Fields that are `None` are simply not activated.
///
/// # Examples
///
/// **Plain text-to-image (defaults):**
/// ```rust,ignore
/// DrishtiRequest {
///     prompt: "a red sunset over the ocean".into(),
///     ..DrishtiRequest::default()
/// }
/// ```
///
/// **Editing a photo with an instruction:**
/// ```rust,ignore
/// DrishtiRequest {
///     prompt: "make the sky orange".into(),
///     mode: DrishtiMode::Edit,
///     reference_image: Some(ImageRef { path: "photo.png".into() }),
///     reference_mode: Some(ReferenceMode::Edit),
///     ..DrishtiRequest::default()
/// }
/// ```
///
/// **Inpainting a masked region:**
/// ```rust,ignore
/// DrishtiRequest {
///     prompt: "a sailing boat".into(),
///     mode: DrishtiMode::Inpaint,
///     reference_image: Some(ImageRef { path: "photo.png".into() }),
///     mask: Some(MaskRef { path: "mask.png".into() }),
///     ..DrishtiRequest::default()
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrishtiRequest {
    /// The text prompt describing the desired output.
    ///
    /// For `Generate` this is the scene to create; for `Edit` / `Inpaint`
    /// it is the instruction that modifies the reference image.
    pub prompt: String,

    /// A prompt describing what to avoid.
    ///
    /// Used for CFG guidance against a fixed null sequence
    /// (ARCHITECTURE_VISION_STUDIO_PART1.md §8.3). `None` disables the
    /// negative direction.
    pub negative_prompt: Option<String>,

    /// Which engine combination to route to.
    ///
    /// See [`DrishtiMode`] for the four modes.
    pub mode: DrishtiMode,

    /// The input image: editing source, upscale input, or reference prompt.
    ///
    /// Required for `Edit`, `Inpaint`, and `Upscale` modes; optional for
    /// `Generate` unless used as a reference-prompt image.
    pub reference_image: Option<ImageRef>,

    /// Whether [`reference_image`](Self::reference_image) is an editing
    /// source or a reference prompt.
    ///
    /// `Some` only for `Edit` mode; disambiguates Phase 12 vs Phase 13.
    pub reference_mode: Option<ReferenceMode>,

    /// Strength of the reference image's influence, from 0.0 to 1.0.
    ///
    /// Higher values keep the output closer to the reference (style or
    /// content); lower values give the prompt more freedom.
    pub reference_strength: Option<f32>,

    /// A structural conditioning map (edge / depth / pose).
    ///
    /// When set, the prompt controls content and style while the map
    /// guides composition (ARCHITECTURE_VISION_STUDIO_PART2.md §12).
    pub structural_map: Option<StructuralRef>,

    /// Strength of the structural map's influence, from 0.0 to 1.0.
    pub structural_strength: Option<f32>,

    /// The inpainting mask, required for `Inpaint` mode.
    ///
    /// White pixels mark the region to regenerate (see [`MaskRef`]).
    pub mask: Option<MaskRef>,

    /// Output resolution as `(width, height)`.
    ///
    /// Bounded to the trained bucket table at inference time
    /// (ARCHITECTURE_VISION_STUDIO_PART1.md §8.5).
    pub resolution: (u32, u32),

    /// CFG guidance scale, typically 3.5–7.5.
    ///
    /// 1.0 means no guidance — the model follows the prompt naturally.
    pub guidance_scale: f32,

    /// Number of ODE solver steps, typically 20–40.
    pub steps: u32,

    /// Which ODE solver to use for sampling (see [`SolverKind`]).
    pub solver: SolverKind,

    /// Random seed for reproducible sampling.
    ///
    /// `None` draws a fresh seed per request.
    pub seed: Option<u64>,

    /// The output image format (see [`ImageOutputFormat`]).
    pub output_format: ImageOutputFormat,

    /// A learned style or subject adapter to apply (see [`AdapterRef`]).
    pub style_adapter: Option<AdapterRef>,
}

impl Default for DrishtiRequest {
    /// The default request: empty `Generate` prompt, 512 × 512, PNG output,
    /// Euler solver, 28 steps, guidance scale 4.0 — every field optional
    /// or engine-tuned left `None`.
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            mode: DrishtiMode::Generate,
            reference_image: None,
            reference_mode: None,
            reference_strength: None,
            structural_map: None,
            structural_strength: None,
            mask: None,
            resolution: (512, 512),
            guidance_scale: 4.0,
            steps: 28,
            solver: SolverKind::Euler,
            seed: None,
            output_format: ImageOutputFormat::default(),
            style_adapter: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the default request is a plain 512 × 512 Generate with
    /// PNG output and no conditioning active.
    #[test]
    fn test_default_request() {
        let req = DrishtiRequest::default();
        assert_eq!(req.mode, DrishtiMode::Generate);
        assert_eq!(req.output_format, ImageOutputFormat::Png);
        assert_eq!(req.resolution, (512, 512));
        assert_eq!(req.solver, SolverKind::Euler);
        assert!(req.reference_image.is_none());
        assert!(req.structural_map.is_none());
        assert!(req.mask.is_none());
        assert!(req.style_adapter.is_none());
    }

    /// Round-trip a fully-populated request (every optional field set)
    /// through `serde_json` and verify it reproduces exactly.
    #[test]
    fn test_request_round_trip() {
        let req = DrishtiRequest {
            prompt: "a red sunset over the ocean".to_string(),
            negative_prompt: Some("blurry, low quality".to_string()),
            mode: DrishtiMode::Edit,
            reference_image: Some(ImageRef {
                path: PathBuf::from("photo.png"),
            }),
            reference_mode: Some(ReferenceMode::Edit),
            reference_strength: Some(0.8),
            structural_map: Some(StructuralRef {
                map_path: PathBuf::from("edges.png"),
                kind: StructuralKind::Edge,
            }),
            structural_strength: Some(0.5),
            mask: Some(MaskRef {
                path: PathBuf::from("mask.png"),
            }),
            resolution: (1024, 1024),
            guidance_scale: 6.5,
            steps: 40,
            solver: SolverKind::Heun,
            seed: Some(42),
            output_format: ImageOutputFormat::Jpeg { quality: 90 },
            style_adapter: Some(AdapterRef {
                id: "painterly".to_string(),
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: DrishtiRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    /// Verify that output formats with data round-trip through `serde_json`
    /// without losing their quality payloads.
    #[test]
    fn test_output_format_round_trip() {
        for fmt in [
            ImageOutputFormat::Png,
            ImageOutputFormat::Jpeg { quality: 90 },
            ImageOutputFormat::WebP { quality: 75 },
        ] {
            let json = serde_json::to_string(&fmt).unwrap();
            let back: ImageOutputFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(fmt, back);
        }
    }
}
