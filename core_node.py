"""DiversityBoost node — HF attenuation + DCT composition push."""

import time

from comfy_api.latest import io

from .core import build_diversity_fn


class DiversityBoostCoreV3(io.ComfyNode):
    """Restore composition diversity for distilled diffusion models.

    Single post-cfg hook at step 0: first attenuates HF amplitude
    (polynomial frequency modulation), then applies a random low-frequency 
    DCT spatial field to the blurred result. Push runs AFTER cleanup so its 
    signal cannot be erased by downstream processing.
    
    V3 improvements over legacy:
    - Token-grid normalization (resolution-independent)
    - Near-DC frequency protection (prevents brightness shift)
    - Configurable DCT basis size (2x2 to 8x8)
    - Smooth saturation curves (tanh, softplus) to avoid artifacts
    - Seed control for reproducibility
    - Asymmetric clamping for fine control
    - Mid-frequency preservation to retain lost details
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DiversityBoostCoreV3",
            display_name="Diversity Boost (V3)",
            category="sampling",
            description="Restore composition diversity for distilled models. "
                        "Polynomial HF modulation + DCT composition push at step 0. "
                        "New: mid-freq preservation, smooth saturation, seed control.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=2.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Composition push amplitude. "
                                       "0 = cleanup only. 1.0 = moderate. 2.0 = strong."),
                io.Float.Input("clamp", default=0.5, min=0.1, max=3.0, step=0.1,
                               tooltip="Safety clamp for DCT field values (upper bound). "
                                       "Scale is clamped to [min_scale, 1+clamp]."),
                io.Combo.Input("noise_type",
                               options=["pink", "white", "blue"],
                               default="pink",
                               tooltip="Frequency spectrum of random DCT coefficients. "
                                       "pink = stronger composition push (recommended)."),
                io.Float.Input("dc_preserve", default=0.0, min=0.0, max=1.0, step=0.1,
                               tooltip="DC amplitude preservation (1.0 = keep, 0.0 = zero). "
                                       "Only affects step 0; step 1+ always preserves full DC."),
                io.Boolean.Input("energy_compensate", default=False,
                                 tooltip="Rescale output energy to match original."),
                io.Float.Input("hf_factor", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="High-frequency attenuation [0, 1]. "
                                       "1.0 = full attenuation. Only used in polynomial mode."),
                io.Float.Input("lf_factor", default=0.3, min=0.0, max=1.0, step=0.05,
                               tooltip="Low-frequency amplification [0, 1]. "
                                       "1.0 = +50% boost. Only used in polynomial mode."),
                io.Float.Input("transition", default=2.0, min=0.5, max=4.0, step=0.1,
                               tooltip="Polynomial transition shape. "
                                       "0.5 = steep, 1.0 = linear, 2.0 = smooth, 4.0 = very smooth."),
                io.Combo.Input("schedule",
                               options=["flat", "linear", "cosine"],
                               default="linear",
                               tooltip="Timestep schedule. "
                                       "flat = step 0 only. linear/cosine = progressive decay."),
                # New advanced options
                io.Int.Input("dct_basis_size", default=4, min=2, max=8, step=1,
                             tooltip="DCT basis dimension (2-8). Larger = finer composition control. "
                                     "4x4 = default balance. 8x8 = detailed local variations."),
                io.Int.Input("seed", default=0, min=0, max=2**31-1, step=1,
                             tooltip="Random seed for reproducibility. 0 = random per generation."),
                io.Combo.Input("saturation_mode",
                               options=["hard", "tanh", "softplus"],
                               default="hard",
                               tooltip="Saturation curve for DCT field. "
                                       "hard = clamp (default). tanh/softplus = smooth, artifact-free."),
                io.Float.Input("max_amp", default=2.0, min=0.1, max=5.0, step=0.1,
                               tooltip="Maximum amplitude for tanh/softplus saturation. "
                                       "Controls strength of smooth saturation."),
                io.Float.Input("asymmetric_clamp_min", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Minimum scale bound (asymmetric clamping). "
                                       "0 = use default (0.10). Use to prevent darkening."),
                io.Float.Input("asymmetric_clamp_max", default=0.0, min=0.0, max=5.0, step=0.1,
                               tooltip="Maximum scale bound override. "
                                       "0 = use default (1+clamp). Use to limit brightening."),
                io.Float.Input("midfreq_preserve", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Mid-frequency detail preservation [0, 1]. "
                                       "Recovers structural details lost in aggressive HF attenuation."),
                io.Float.Input("midfreq_start", default=0.3, min=0.1, max=0.9, step=0.05,
                               tooltip="Normalized start of mid-frequency band [0, 1]. "
                                       "Lower = preserve more frequencies."),
                io.Float.Input("midfreq_end", default=0.7, min=0.2, max=1.0, step=0.05,
                               tooltip="Normalized end of mid-frequency band [0, 1]. "
                                       "Higher = preserve more frequencies."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time()

    @classmethod
    def execute(cls, model, strength, clamp, noise_type,
                dc_preserve, energy_compensate,
                hf_factor, lf_factor, transition, schedule,
                dct_basis_size, seed, saturation_mode, max_amp,
                asymmetric_clamp_min, asymmetric_clamp_max,
                midfreq_preserve, midfreq_start, midfreq_end) -> io.NodeOutput:
        m = model.clone()

        # Build asymmetric clamp tuple (convert 0 to None for defaults)
        asym_min = asymmetric_clamp_min if asymmetric_clamp_min > 0 else None
        asym_max = asymmetric_clamp_max if asymmetric_clamp_max > 0 else None
        asymmetric_clamp = (asym_min, asym_max)

        # Use seed=0 as "no fixed seed" (random each time)
        fixed_seed = seed if seed > 0 else None

        m.set_model_sampler_post_cfg_function(
            build_diversity_fn(
                strength=strength,
                clamp_val=clamp,
                noise_type=noise_type,
                dc_preserve=dc_preserve,
                energy_compensate=energy_compensate,
                mode="polynomial",
                hf_factor=hf_factor,
                lf_factor=lf_factor,
                transition=transition,
                schedule=schedule,
                dct_basis_size=dct_basis_size,
                seed=fixed_seed,
                saturation_mode=saturation_mode,
                max_amp=max_amp,
                asymmetric_clamp=asymmetric_clamp,
                midfreq_preserve=midfreq_preserve,
                midfreq_start=midfreq_start,
                midfreq_end=midfreq_end,
            ),
        )

        return io.NodeOutput(m)
