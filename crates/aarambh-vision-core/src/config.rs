//! `src/config.rs` — shared configuration types for aarambh-vision-studio.
//!
//! Provides the one core type used by every crate in the workspace:
//!
//! - [`ModelConfig`] — Multimodal Diffusion Transformer (MMDiT)
//!   hyper-parameters with four preset scales.
//!
//! # Model scales (ARCHITECTURE_VISION_STUDIO_PART1.md §5)
//!
//! | Scale | `d_model` | `n_dual_layers` | `n_single_layers` | `n_heads` | Params (MMDiT) | Native resolution |
//! |-------|-----------|-----------------|--------------------|-----------|----------------|--------------------|
//! | Tiny   | 384  | 4  | 8  | 6  | ≈ 40 M  | 64 × 64   |
//! | Small  | 512  | 6  | 12 | 8  | ≈ 120 M | 128 × 128 |
//! | Medium | 768  | 8  | 16 | 12 | ≈ 350 M | 256 × 256 |
//! | Large  | 1024 | 12 | 24 | 16 | ≈ 900 M | 512 × 512 |
//!
//! All scales share `latent_channels = 16` and `patch_size = 2`, which are
//! fixed by the scale-independent Image VAE (§6.1), not by the MMDiT scale.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters that fully describe an MMDiT model for vision-studio.
///
/// Every engine (text-to-image, editing, reference prompting, structural
/// conditioning, upscaling) shares this same struct, differing only in the
/// conditioning branches attached to the core backbone. The struct is
/// serialisable so checkpoints carry their own config.
///
/// # Constraint
///
/// `d_model` must be divisible by `n_heads`. The training pipeline validates
/// this before constructing any tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Embedding dimension (width of every hidden state and attention output).
    ///
    /// Must be divisible by [`n_heads`](Self::n_heads).
    /// Tiny = 384, Small = 512, Medium = 768, Large = 1024.
    pub d_model: usize,

    /// Number of dual-stream transformer blocks in the MMDiT backbone.
    ///
    /// Each dual block processes text and image token streams with joint
    /// attention before merging. Tiny = 4, Small = 6, Medium = 8, Large = 12.
    pub n_dual_layers: usize,

    /// Number of single-stream transformer blocks after the dual block.
    ///
    /// Each single block operates on the merged token stream.
    /// Tiny = 8, Small = 12, Medium = 16, Large = 24.
    pub n_single_layers: usize,

    /// Number of attention heads per block.
    ///
    /// Head dimension = `d_model / n_heads`.
    /// Tiny = 6 (head_dim 64), Small = 8 (head_dim 64),
    /// Medium = 12 (head_dim 64), Large = 16 (head_dim 64).
    pub n_heads: usize,

    /// Number of channels in the VAE's continuous latent grid.
    ///
    /// Fixed at 16 by the Image VAE (§6.1) and scale-independent — the same
    /// VAE is trained once and frozen for every MMDiT scale.
    pub latent_channels: usize,

    /// Side length of the square latent patch that becomes one MMDiT token.
    ///
    /// Patchify cuts the `(H/8) × (W/8)` latent grid into `2 × 2` patches,
    /// so a 64 × 64 image yields a 4 × 4 token grid. Fixed at 2 (§6.1).
    pub patch_size: usize,
}

impl ModelConfig {
    // -----------------------------------------------------------------------
    // Presets
    // -----------------------------------------------------------------------

    /// Returns the **Tiny** preset: 384-dim, 4 dual + 8 single layers, ≈ 40 M params.
    ///
    /// Designed for CPU-fast iteration on an i3 with 8 GB RAM at 64 × 64
    /// native resolution. Use this scale for all initial experiments and
    /// for the Phase 8 text-to-image baseline.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 384 |
    /// | `n_dual_layers` | 4 |
    /// | `n_single_layers` | 8 |
    /// | `n_heads` | 6 |
    /// | `latent_channels` | 16 |
    /// | `patch_size` | 2 |
    pub fn tiny() -> Self {
        Self {
            d_model: 384,
            n_dual_layers: 4,
            n_single_layers: 8,
            n_heads: 6,
            latent_channels: 16,
            patch_size: 2,
        }
    }

    /// Returns the **Small** preset: 512-dim, 6 dual + 12 single layers, ≈ 120 M params.
    ///
    /// The default scale for end-to-end engine training when a GPU (Kaggle
    /// T4 or similar) is available, at 128 × 128 native resolution.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 512 |
    /// | `n_dual_layers` | 6 |
    /// | `n_single_layers` | 12 |
    /// | `n_heads` | 8 |
    /// | `latent_channels` | 16 |
    /// | `patch_size` | 2 |
    pub fn small() -> Self {
        Self {
            d_model: 512,
            n_dual_layers: 6,
            n_single_layers: 12,
            n_heads: 8,
            latent_channels: 16,
            patch_size: 2,
        }
    }

    /// Returns the **Medium** preset: 768-dim, 8 dual + 16 single layers, ≈ 350 M params.
    ///
    /// Requires a GPU with ≥ 24 GB VRAM for training at 256 × 256 native
    /// resolution. The largest scale that remains practical for a solo
    /// developer on a single GPU.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 768 |
    /// | `n_dual_layers` | 8 |
    /// | `n_single_layers` | 16 |
    /// | `n_heads` | 12 |
    /// | `latent_channels` | 16 |
    /// | `patch_size` | 2 |
    pub fn medium() -> Self {
        Self {
            d_model: 768,
            n_dual_layers: 8,
            n_single_layers: 16,
            n_heads: 12,
            latent_channels: 16,
            patch_size: 2,
        }
    }

    /// Returns the **Large** preset: 1024-dim, 12 dual + 24 single layers, ≈ 900 M params.
    ///
    /// The v1 maximum scale, trained at 512 × 512 native resolution.
    /// Requires multi-GPU training or a high-VRAM cloud instance (A100 80 GB).
    /// Use for the final production checkpoint after all capabilities are
    /// validated at Small or Medium.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 1024 |
    /// | `n_dual_layers` | 12 |
    /// | `n_single_layers` | 24 |
    /// | `n_heads` | 16 |
    /// | `latent_channels` | 16 |
    /// | `patch_size` | 2 |
    pub fn large() -> Self {
        Self {
            d_model: 1024,
            n_dual_layers: 12,
            n_single_layers: 24,
            n_heads: 16,
            latent_channels: 16,
            patch_size: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `ModelConfig::tiny()` produces the exact field values
    /// from ARCHITECTURE_VISION_STUDIO_PART1.md §5.
    #[test]
    fn test_tiny() {
        let cfg = ModelConfig::tiny();
        assert_eq!(cfg.d_model, 384);
        assert_eq!(cfg.n_dual_layers, 4);
        assert_eq!(cfg.n_single_layers, 8);
        assert_eq!(cfg.n_heads, 6);
        assert_eq!(cfg.latent_channels, 16);
        assert_eq!(cfg.patch_size, 2);
    }

    /// Verify that `ModelConfig::small()` produces the expected field values.
    #[test]
    fn test_small() {
        let cfg = ModelConfig::small();
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.n_dual_layers, 6);
        assert_eq!(cfg.n_single_layers, 12);
        assert_eq!(cfg.n_heads, 8);
    }

    /// Verify that `ModelConfig::medium()` produces the expected field values.
    #[test]
    fn test_medium() {
        let cfg = ModelConfig::medium();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.n_dual_layers, 8);
        assert_eq!(cfg.n_single_layers, 16);
        assert_eq!(cfg.n_heads, 12);
    }

    /// Verify that `ModelConfig::large()` produces the expected field values.
    #[test]
    fn test_large() {
        let cfg = ModelConfig::large();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.n_dual_layers, 12);
        assert_eq!(cfg.n_single_layers, 24);
        assert_eq!(cfg.n_heads, 16);
    }

    /// Verify that every scale keeps the VAE-fixed fields at 16 / 2.
    #[test]
    fn test_vae_fields_are_scale_independent() {
        for cfg in [
            ModelConfig::tiny(),
            ModelConfig::small(),
            ModelConfig::medium(),
            ModelConfig::large(),
        ] {
            assert_eq!(cfg.latent_channels, 16);
            assert_eq!(cfg.patch_size, 2);
        }
    }

    /// Round-trip through `serde_json`: serialise then deserialise must
    /// produce an identical [`ModelConfig`].
    #[test]
    fn test_config_round_trip() {
        let cfg = ModelConfig::medium();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
    }

    /// Same round-trip test for the Tiny / Large presets to catch
    /// per-preset serialisation issues.
    #[test]
    fn test_config_round_trip_all_scales() {
        for cfg in [
            ModelConfig::tiny(),
            ModelConfig::small(),
            ModelConfig::medium(),
            ModelConfig::large(),
        ] {
            let json = serde_json::to_string(&cfg).unwrap();
            let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(cfg, deserialized);
        }
    }
}
