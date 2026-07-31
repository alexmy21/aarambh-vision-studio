//! `src/lib.rs` — crate root for `aarambh-vision-core`.
//!
//! Re-exports every public type so callers can write
//! `use aarambh_vision_core::ModelConfig` instead of
//! `use aarambh_vision_core::config::ModelConfig`.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`config`] | [`ModelConfig`] with scale presets |
//! | [`error`]  | [`AarambhVisionError`], one variant per crate |
//! | [`request`]| [`DrishtiRequest`] and per-engine control types |

pub mod config;
pub mod error;
pub mod request;

pub use config::ModelConfig;
pub use error::{AarambhVisionError, VisionResult};
pub use request::{
    AdapterRef, DrishtiMode, DrishtiRequest, ImageOutputFormat, ImageRef, MaskRef, ReferenceMode,
    SolverKind, StructuralKind, StructuralRef,
};
