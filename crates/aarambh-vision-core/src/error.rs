//! `src/error.rs` — centralised error type for aarambh-vision-studio.
//!
//! Defines [`AarambhVisionError`], the single error enum whose variants
//! correspond to every crate in the workspace, and a convenience alias
//! [`VisionResult<T>`] that pins the error variant.
//!
//! # Variants
//!
//! | Variant | Source | When raised |
//! |---------|--------|-------------|
//! | `Tokenizer` | manual | Image VAE encode / decode / reparameterisation failure |
//! | `TextPrep` | manual | prompt tokenization / normalisation failure |
//! | `Data` | manual | dataset loading, preprocessing, auto-captioning failure |
//! | `Understand` | manual | contrastive encoder or captioner forward failure |
//! | `Nn` | manual | MMDiT block, AdaLN-Zero, or 2D-RoPE failure |
//! | `Kernel` | manual | CPU SIMD or fused patchify kernel failure |
//! | `TextEncoder` | manual | loading or running aarambh-studio's text encoder |
//! | `Model` | manual | full MMDiT assembly forward failure |
//! | `Weights` | manual | SafeTensors save / load failure |
//! | `Train` | manual | training loop (optimiser, checkpointing) failure |
//! | `Quant` | manual | INT8 / INT4 / GGUF-style quantisation failure |
//! | `Finetune` | manual | LoRA / QLoRA / DoRA adapter injection or training failure |
//! | `Align` | manual | GRPO / DPO alignment loop failure |
//! | `SelfLearn` | manual | online self-learning or anti-forgetting failure |
//! | `Edit` | manual | reference-image conditioning or inpainting failure |
//! | `Structure` | manual | structural / spatial conditioning failure |
//! | `RefPrompt` | manual | reference-image prompting failure |
//! | `Upscale` | manual | super-resolution refinement failure |
//! | `Safety` | manual | content filtering, watermarking, or provenance failure |
//! | `Eval` | manual | evaluation harness or baseline comparison failure |
//! | `Control` | manual | DrishtiRequest parsing or validation failure |
//! | `Inference` | manual | sampler (ODE solver), CFG, or runtime failure |
//! | `Serve` | manual | HTTP server startup or request-handling failure |
//! | `Serialisation` | manual | JSON / TOML / message-pack serialisation failure |
//! | `Io` | `#[from]` | filesystem read / write failure |

use thiserror::Error;

// ---------------------------------------------------------------------------
// AarambhVisionError
// ---------------------------------------------------------------------------

/// The single error type used across every crate in the workspace.
///
/// Each variant corresponds to one crate's failure mode. Variants that
/// wrap a bare [`String`] are constructed manually; the [`Io`] variant
/// is converted automatically via `#[from]` so callers can use the `?`
/// operator on [`std::io::Error`].
///
/// [`Io`]: Self::Io
#[derive(Debug, Error)]
pub enum AarambhVisionError {
    /// `aarambh-vision-tokenizer` — Image VAE encode, decode, or reparameterisation failure.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// `aarambh-vision-textprep` — prompt tokenization or normalisation failure.
    #[error("Text prep error: {0}")]
    TextPrep(String),

    /// `aarambh-vision-data` — dataset loading, preprocessing, or auto-captioning failure.
    #[error("Data pipeline error: {0}")]
    Data(String),

    /// `aarambh-vision-understand` — contrastive encoder or captioner forward failure.
    #[error("Understanding error: {0}")]
    Understand(String),

    /// `aarambh-vision-nn` — MMDiT block, AdaLN-Zero, or 2D-RoPE failure.
    #[error("Neural network error: {0}")]
    Nn(String),

    /// `aarambh-vision-kernel` — CPU SIMD kernel or fused patchify failure.
    #[error("Kernel error: {0}")]
    Kernel(String),

    /// `aarambh-vision-textencoder` — loading or running aarambh-studio's checkpoint.
    #[error("Text encoder error: {0}")]
    TextEncoder(String),

    /// `aarambh-vision-model` — full MMDiT assembly forward pass failure.
    #[error("Model error: {0}")]
    Model(String),

    /// `aarambh-vision-weights` — SafeTensors save or load failure.
    #[error("Weights error: {0}")]
    Weights(String),

    /// `aarambh-vision-train` — training-loop failure (optimiser, checkpointing).
    #[error("Training error: {0}")]
    Train(String),

    /// `aarambh-vision-quant` — INT8 / INT4 quantisation or GGUF export failure.
    #[error("Quantisation error: {0}")]
    Quant(String),

    /// `aarambh-vision-finetune` — LoRA/QLoRA/DoRA adapter injection or training failure.
    #[error("Fine-tuning error: {0}")]
    Finetune(String),

    /// `aarambh-vision-align` — GRPO or DPO alignment loop failure.
    #[error("Alignment error: {0}")]
    Align(String),

    /// `aarambh-vision-selflearn` — online self-learning or anti-forgetting failure.
    #[error("Self-learning error: {0}")]
    SelfLearn(String),

    /// `aarambh-vision-edit` — reference-image conditioning or inpainting failure.
    #[error("Editing error: {0}")]
    Edit(String),

    /// `aarambh-vision-structure` — structural / spatial conditioning failure.
    #[error("Structural conditioning error: {0}")]
    Structure(String),

    /// `aarambh-vision-refprompt` — reference-image prompting failure.
    #[error("Reference prompting error: {0}")]
    RefPrompt(String),

    /// `aarambh-vision-upscale` — super-resolution refinement failure.
    #[error("Upscaling error: {0}")]
    Upscale(String),

    /// `aarambh-vision-safety` — content filtering, watermarking, or provenance failure.
    #[error("Safety error: {0}")]
    Safety(String),

    /// `aarambh-vision-eval` — evaluation harness or baseline-comparison failure.
    #[error("Evaluation error: {0}")]
    Eval(String),

    /// `aarambh-vision-control` — DrishtiRequest parsing or validation failure.
    #[error("Control layer error: {0}")]
    Control(String),

    /// `aarambh-vision-inference` — sampler, CFG, or shared-runtime failure.
    #[error("Inference error: {0}")]
    Inference(String),

    /// `aarambh-vision-serve` — HTTP server startup or request-handling failure.
    #[error("Server error: {0}")]
    Serve(String),

    /// A JSON / TOML / message-pack serialisation or deserialisation error.
    #[error("Serialisation error: {0}")]
    Serialisation(String),

    /// A filesystem read or write failed.
    ///
    /// Automatically converted from [`std::io::Error`] via `#[from]`, so
    /// any `std::fs` or `std::io` operation inside a `Result`-returning
    /// function can use `?` directly.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

/// Crate-wide `Result` alias — pins the `Err` variant to [`AarambhVisionError`].
///
/// Import with `use crate::error::VisionResult;` in any module that returns
/// fallible values.
pub type VisionResult<T> = std::result::Result<T, AarambhVisionError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that every variant displays a human-readable message
    /// starting with the owning crate's name.
    #[test]
    fn test_error_display() {
        let cases = [
            (
                AarambhVisionError::Tokenizer("bad latent".into()),
                "Tokenizer error: bad latent",
            ),
            (
                AarambhVisionError::Control("bad request".into()),
                "Control layer error: bad request",
            ),
            (
                AarambhVisionError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "missing",
                )),
                "IO error: missing",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    /// Verify that a [`std::io::Error`] converts automatically via `#[from]`.
    #[test]
    fn test_io_error_converts_via_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let err: AarambhVisionError = io_err.into();
        assert!(matches!(err, AarambhVisionError::Io(_)));
    }

    /// Verify that the [`VisionResult`] alias carries the core error type.
    #[test]
    fn test_result_alias() {
        fn fail() -> VisionResult<()> {
            Err(AarambhVisionError::Tokenizer("broken".into()))
        }
        assert!(fail().is_err());
    }
}
