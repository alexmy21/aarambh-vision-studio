//! `aarambh-vision-sidecar` — the EWM state-machine side-car for the
//! aarambh-vision sampler loop.
//!
//! This crate is the **adapter** between the host (a Candle rectified-flow
//! MMDiT sampler) and the host-agnostic EWM side-car
//! (`ewm-flux-host`). It is deliberately thin: the only host-specific work
//! is reading a `candle_core::Tensor` at each denoising step; everything
//! downstream (S(t), H(t-1), D/R/N, Boolean-ring window, warnings, the
//! two-score evaluation, basis frames / time travel) lives in the EWM
//! crates and is vocabulary- and host-agnostic
//! (docs/BASIS_FRAMES.md, docs/notes/flux-notes.md).
//!
//! # Integration contract
//!
//! The sampler loop of `aarambh-vision-inference` (ARCHITECTURE_PART2 §13)
//! calls one method per denoising step:
//!
//! ```rust,ignore
//! // x_t: [seq_len, dim] f32 latent at denoising step t (batch = 1)
//! let mut sidecar = FluxSidecar::new(dim, codebook_size, seed);
//! for t in (0..steps).rev() {
//!     let x_t = model.step(&x_t_plus_dt, t);        // the host's own code
//!     sidecar.observe(steps - t - 1, &x_t)?;        // the side-car hook
//! }
//! let report = sidecar.finish();                    // JSON-ready FluxReport
//! ```
//!
//! `FluxSidecar` also implements [`SamplerHook`], the trait the inference
//! crate can take as a callback without depending on EWM types directly.

use candle_core::{DType, Result, Tensor};

use ewm_flux_host::{run_states, CodebookEncoder, CodebookProbe, FluxReport, LatentState};

/// The hook contract for the inference runtime's sampler loop.
///
/// The inference crate depends only on this trait (and `candle_core`), not
/// on the EWM crates; the side-car is injected by the caller. This keeps the
/// host-specific surface to exactly one method.
pub trait SamplerHook {
    /// Observe the latent state at one denoising step.
    fn on_step(&mut self, step: usize, latent: &Tensor) -> Result<()>;
}

/// The Flux side-car: collects per-step latent states and produces the full
/// EWM report at the end of the sampler loop.
pub struct FluxSidecar {
    probe: CodebookProbe,
    states: Vec<LatentState>,
}

impl FluxSidecar {
    /// A side-car with a deterministic codebook (`dim`-anchors, `count`
    /// anchors). The real host should share the exact codebook with the
    /// encoder side — use `with_probe` with a `CodebookEncoder::from_anchors`
    /// for bit-identical quantization.
    pub fn new(dim: usize, count: usize, seed: u64) -> Self {
        Self::with_probe(CodebookProbe::new(CodebookEncoder::new(dim, count, seed)))
    }

    /// A side-car with a caller-supplied codebook probe (shared codebook).
    pub fn with_probe(probe: CodebookProbe) -> Self {
        Self {
            probe,
            states: Vec::new(),
        }
    }

    /// Record the latent state of one denoising step.
    ///
    /// `latent` must be a 2-D `[seq_len, dim]` f32 tensor (batch = 1, the
    /// MMDiT token stream). Steps must arrive in denoising order.
    pub fn observe(&mut self, _step: usize, latent: &Tensor) -> Result<()> {
        let state = latent_to_state(latent)?;
        self.states.push(state);
        Ok(())
    }

    /// Finish the run: run the full side-car loop over the collected
    /// trajectory and return the report (per-step S(t), D/R/N, ring,
    /// warnings, two-score eval, basis frames).
    pub fn finish(mut self) -> FluxReport {
        run_states(&self.states, &mut self.probe)
    }
}

impl SamplerHook for FluxSidecar {
    fn on_step(&mut self, step: usize, latent: &Tensor) -> Result<()> {
        self.observe(step, latent)
    }
}

/// `[seq_len, dim] f32` Candle tensor → per-token latent vectors.
fn latent_to_state(latent: &Tensor) -> Result<LatentState> {
    let latent = latent.to_dtype(DType::F32)?;
    let dims = latent.dims();
    if dims.len() != 2 {
        candle_core::bail!(
            "expected a 2-D [seq_len, dim] latent tensor, got shape {dims:?}"
        );
    }
    Ok(latent.to_vec2::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observes_a_candle_trajectory_and_finishes() {
        let mut sidecar = FluxSidecar::new(8, 64, 3);
        for t in 0..5 {
            let x = Tensor::randn(0f32, 1f32, (16, 8), &candle_core::Device::Cpu).unwrap();
            sidecar.observe(t, &x).unwrap();
        }
        let report = sidecar.finish();
        assert_eq!(report.frames.len(), 5);
        assert_eq!(report.config.dim, 8);
        assert_eq!(report.config.seq_len, 16);
        assert!(report.scores.loop_order_exact);
    }

    #[test]
    fn rejects_non_2d_latents() {
        let mut sidecar = FluxSidecar::new(8, 64, 3);
        let x = Tensor::randn(0f32, 1f32, (2, 16, 8), &candle_core::Device::Cpu).unwrap();
        assert!(sidecar.observe(0, &x).is_err());
    }
}
