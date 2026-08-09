"""DiversityBoost core — HF attenuation + DCT composition push.

Restores composition diversity lost during distillation via two mechanisms
applied in a single post-cfg hook at step 0:
1. HF attenuation (polynomial frequency modulation) — blurry "composition sketch"
2. DCT composition push — multiplicative low-freq spatial field

V3 improvements over legacy:
- Token-grid normalization (resolution-independent via DiT patch_size)
- Near-DC frequency protection (prevents brightness/color shift)
- Polynomial transition (smooth vs steep Butterworth cliff)
- Timestep scheduling (flat/linear/cosine decay)
- Multiple DCT basis sizes (2x2 to 8x8)
- Smooth saturation curves (tanh, softplus) to avoid hard-clamp artifacts
- Seed control for reproducibility
- Asymmetric clamping for independent positive/negative bounds
- Mid-frequency preservation to retain detail lost in aggressive HF attenuation
"""

import logging
import math
from functools import lru_cache
from typing import Optional

import torch

from .sampling import (
    denoised_to_raw,
    raw_to_denoised,
    find_step_index,
    unpack_video_if_needed,
    repack_video_if_needed,
    apply_saturation,
)

log = logging.getLogger("ComfyUI-DiversityBoost")


# ---------------------------------------------------------------------------
# 2D DCT basis (orthonormal, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _build_dct_basis_2d(H, W, n_modes_h=4, n_modes_w=4):
    """Build orthonormal 2D DCT-II basis matrix [H*W, n_modes_h*n_modes_w].

    Same matrix for analysis (projection) and synthesis (reconstruction):
      synthesis: field = basis @ coeffs
    
    Supports configurable basis sizes from 2x2 to 8x8 for fine-grained control
    over composition diversity granularity.
    """
    def _dct1d(N, n_modes):
        n = torch.arange(N, dtype=torch.float64)
        k = torch.arange(n_modes, dtype=torch.float64)
        phi = torch.cos(math.pi * k[None, :] * (n[:, None] + 0.5) / N)
        norm = torch.full((n_modes,), math.sqrt(2.0 / N), dtype=torch.float64)
        norm[0] = 1.0 / math.sqrt(N)
        return phi * norm[None, :]

    phi_h = _dct1d(H, n_modes_h)
    phi_w = _dct1d(W, n_modes_w)
    basis_2d = torch.einsum('hu,wv->hwuv', phi_h, phi_w)
    basis_2d = basis_2d.reshape(H * W, n_modes_h * n_modes_w)
    return basis_2d.float()


@lru_cache(maxsize=8)
def _build_midfreq_mask(H, W, patch_size, preserve_start=0.3, preserve_end=0.7):
    """Build mid-frequency preservation mask for rfft2 output.
    
    This mask preserves a band of mid-frequencies that contain important
    structural details often lost in aggressive HF attenuation.
    
    Args:
        H, W: spatial dimensions
        patch_size: DiT patch size for token-grid normalization
        preserve_start: normalized frequency where preservation starts (0-1)
        preserve_end: normalized frequency where preservation ends (0-1)
    
    Returns:
        Mask tensor [1, 1, H, W//2+1] with values in [0, 1]
    """
    token_H = H // patch_size
    token_W = W // patch_size
    
    freq_y = torch.fft.fftfreq(H, device='cpu').unsqueeze(1).float()
    freq_x = torch.fft.rfftfreq(W, device='cpu').unsqueeze(0).float()
    r_norm = torch.sqrt((freq_y * token_H) ** 2 + (freq_x * token_W) ** 2)
    r_tilde = (r_norm / (1.0 + r_norm)).clamp(0, 1)
    
    # Smooth band-pass using sigmoid transitions
    transition_width = 0.1
    low_transition = 0.5 * (1.0 + torch.tanh((r_tilde - preserve_start) / transition_width))
    high_transition = 0.5 * (1.0 + torch.tanh((preserve_end - r_tilde) / transition_width))
    mask = low_transition * high_transition
    
    # Ensure DC is not affected by mid-frequency preservation
    # (DC handling is controlled separately by dc_preserve parameter)
    mask[0, 0] = 0.0
    
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Noise frequency weights
# ---------------------------------------------------------------------------

def _build_pink_weights(n_h, n_w):
    """1/f amplitude weights: lower frequencies dominate."""
    weights = []
    for u in range(n_h):
        for v in range(n_w):
            freq_sq = u * u + v * v
            weights.append(0.0 if freq_sq == 0 else 1.0 / (freq_sq ** 0.25))
    return torch.tensor(weights, dtype=torch.float32)


def _build_blue_weights(n_h, n_w):
    """f-proportional weights: higher frequencies dominate."""
    weights = []
    for u in range(n_h):
        for v in range(n_w):
            freq_sq = u * u + v * v
            weights.append(0.0 if freq_sq == 0 else (freq_sq ** 0.25))
    return torch.tensor(weights, dtype=torch.float32)


def _build_noise_weights(noise_type, n_h, n_w):
    """Build frequency weights for given noise type, or None for white."""
    if noise_type == "pink":
        return _build_pink_weights(n_h, n_w)
    elif noise_type == "blue":
        return _build_blue_weights(n_h, n_w)
    return None


# ---------------------------------------------------------------------------
# Butterworth LPF
# ---------------------------------------------------------------------------

def _get_patch_size(model):
    """Read DiT patch_size from model object. Fallback to 2."""
    try:
        inner = getattr(model, 'model', None)
        if inner is not None:
            dm = getattr(inner, 'diffusion_model', None)
            if dm is not None:
                ps = getattr(dm, 'patch_size', None)
                if ps is not None:
                    return int(ps)
    except Exception:
        pass

    try:
        ps = getattr(model, 'patch_size', None)
        if ps is not None:
            return int(ps)
    except Exception:
        pass

    return 2


def _build_freq_modulation(H, W, patch_size, mode, hf_factor, lf_factor,
                            transition, dc_preserve, device,
                            midfreq_preserve=0.0, midfreq_start=0.3,
                            midfreq_end=0.7):
    """Build frequency modulation scale for rfft2 output [1, 1, H, W//2+1].

    Token-grid normalization: uses DiT patch_size to make frequency
    modulation independent of latent resolution.

    mode="butterworth": legacy steep order-12 cliff.
    mode="polynomial":   continuous smooth modulation.
      s_hf = 1 - hf_factor   (1.0=no atten, 0.0=full atten)
      s_lf = 1 + lf_factor*0.5  (1.0=no boost, 1.5=max boost)
      scale = s_hf + (s_lf - s_hf) * (1 - r_tilde)^transition
    
    Args:
        midfreq_preserve: strength of mid-frequency preservation [0, 1]
        midfreq_start: normalized start of mid-frequency band
        midfreq_end: normalized end of mid-frequency band
    """
    token_H = H // patch_size
    token_W = W // patch_size

    freq_y = torch.fft.fftfreq(H, device=device).unsqueeze(1)
    freq_x = torch.fft.rfftfreq(W, device=device).unsqueeze(0)
    r_norm = torch.sqrt((freq_y * token_H) ** 2 +
                        (freq_x * token_W) ** 2)

    if mode == "butterworth":
        # Legacy: steep order-12 cliff
        scale = 1.0 / torch.sqrt(1.0 + r_norm.pow(12))
        scale[0, 0] = 0.0
    else:
        # Polynomial: continuous smooth modulation
        # r_tilde in [0, 1), soft-saturated at high frequencies
        r_tilde = (r_norm / (1.0 + r_norm)).clamp(0, 1)
        s_hf = 1.0 - hf_factor
        s_lf = 1.0 + lf_factor * 0.5
        scale = s_hf + (s_lf - s_hf) * (1.0 - r_tilde).pow(transition)
        # Protect near-DC frequencies (0 < r_norm < 1.0) at full amplitude.
        # Hard cutoff preserves brightness; smooth blend (dc_preserve * (1-blend)
        # + poly * blend) attenuates near-DC and causes brightness shift.
        near_dc = (r_norm > 0) & (r_norm < 1.0)
        scale[near_dc] = 1.0
        # Exact DC bin is controlled by user setting
        scale[0, 0] = dc_preserve
    
    # Mid-frequency preservation: blend back details lost in attenuation
    if midfreq_preserve > 0.0:
        mid_mask = _build_midfreq_mask(H, W, patch_size, midfreq_start, midfreq_end).to(device=device)
        # Blend: final = scale * (1 - midfreq_preserve) + mid_mask * midfreq_preserve
        # But we want to preserve mids, so we add back what was lost
        scale = scale + mid_mask * midfreq_preserve * (1.0 - scale)
        # Clamp to reasonable range
        scale = scale.clamp(0.0, 1.5)

    return scale.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Combined hook: HF attenuation → DCT composition push
# ---------------------------------------------------------------------------

def build_diversity_fn(strength=2.0, clamp_val=0.5, noise_type="pink",
                       dc_preserve=0.0, energy_compensate=False,
                       mode="polynomial", hf_factor=1.0, lf_factor=0.3,
                       transition=2.0, schedule="linear",
                       dct_basis_size=4, seed=None,
                       saturation_mode="hard", max_amp=2.0,
                       asymmetric_clamp=(None, None),
                       midfreq_preserve=0.0, midfreq_start=0.3,
                       midfreq_end=0.7):
    """Build a post_cfg_function that attenuates HF then applies DCT push.

    Execution order:
      1. Convert to raw latent space
      2. Frequency modulation (Butterworth or polynomial) with optional mid-freq preservation
      3. DCT composition push (configurable basis size, multiplicative)
      4. Smooth or hard saturation clamping
      5. Optional energy compensation
      6. Convert back

    Parameters:
        strength:          push amplitude (0-2). 0 = cleanup only.
        clamp_val:         safety clamp for field values (upper bound).
        noise_type:        "pink", "white", or "blue" frequency weighting.
        dc_preserve:       DC amplitude preservation [0, 1].
        energy_compensate: rescale output RMS to match original.
        mode:              "butterworth" (legacy cliff) or "polynomial" (smooth).
        hf_factor:         high-frequency attenuation strength [0, 1].
        lf_factor:         low-frequency amplification strength [0, 1].
        transition:        polynomial transition shape (0.5=steep, 4.0=smooth).
        schedule:          "flat" (step-0-only), "linear", or "cosine" decay.
        dct_basis_size:    DCT basis dimension (2-8). Larger = finer composition control.
        seed:              Random seed for reproducibility. None = random per call.
        saturation_mode:   "hard" (clamp), "tanh" (smooth), or "softplus".
        max_amp:           Maximum amplitude for tanh/softplus saturation.
        asymmetric_clamp:  (min_scale, max_scale) for asymmetric bounds. Use None for defaults.
        midfreq_preserve:  Strength of mid-frequency detail preservation [0, 1].
        midfreq_start:     Normalized start of mid-frequency band [0, 1].
        midfreq_end:       Normalized end of mid-frequency band [0, 1].
    """
    # Validate mid-frequency band parameters
    if not (0 <= midfreq_start < midfreq_end <= 1):
        raise ValueError(
            f"Invalid mid-frequency band: require 0 <= midfreq_start < midfreq_end <= 1, "
            f"got midfreq_start={midfreq_start}, midfreq_end={midfreq_end}"
        )

    n_modes_h = n_modes_w = dct_basis_size
    n_modes = n_modes_h * n_modes_w
    freq_weights = _build_noise_weights(noise_type, n_modes_h, n_modes_w)

    state = {
        "cached_key": None,
        "amp_scale": None,
        "basis_2d": None,
        "freq_weights": None,
    }

    def diversity_hook(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        model = args["model"]
        model_options = args["model_options"]

        sample_sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        if sample_sigmas is None:
            return denoised
        step_index = find_step_index(sigma, sample_sigmas)
        total_steps = len(sample_sigmas) - 1
        step_progress = step_index / max(total_steps, 1)

        # --- Timestep scheduling ---
        if schedule == "flat":
            step_weight = 1.0 if step_index == 0 else 0.0
        elif schedule == "linear":
            step_weight = max(0.0, 1.0 - step_progress * 3.0)
        elif schedule == "cosine":
            step_weight = 0.5 * (1.0 + math.cos(math.pi * step_progress))
        else:
            step_weight = 1.0 if step_index == 0 else 0.0

        if step_weight <= 0.0:
            return denoised

        # Frequency modulation decays with schedule
        effective_hf = hf_factor * step_weight
        effective_lf = lf_factor * step_weight
        # DCT push only at step 0 to avoid multi-step accumulation
        effective_strength = strength if step_index == 0 else 0.0
        # DC is only modified at step 0; step 1+ always preserves full DC.
        # This matches flat schedule behavior and avoids repeated DC
        # attenuation causing color/brightness shift in linear/cosine modes.
        effective_dc = dc_preserve if step_index == 0 else 1.0

        # --- Unpack video if needed ---
        working, pack_info = unpack_video_if_needed(denoised, args)

        # --- Convert to raw space ---
        raw_pred = denoised_to_raw(working, model)
        is_video = raw_pred.ndim == 5
        if is_video:
            B, C, T, H, W = raw_pred.shape
            raw_pred = raw_pred.view(B * T, C, H, W)
        else:
            B, C, H, W = raw_pred.shape
        device = raw_pred.device
        orig_dtype = raw_pred.dtype

        # --- Build or reuse cached tensors ---
        patch_size = _get_patch_size(model)
        cache_key = (H, W, patch_size, mode, effective_hf, effective_lf, transition, 
                     effective_dc, midfreq_preserve, midfreq_start, midfreq_end)
        if state["cached_key"] != cache_key:
            state["amp_scale"] = _build_freq_modulation(
                H, W, patch_size, mode, effective_hf, effective_lf,
                transition, effective_dc, device,
                midfreq_preserve, midfreq_start, midfreq_end,
            )
            if effective_strength > 1e-6:
                basis_cpu = _build_dct_basis_2d(H, W, n_modes_h, n_modes_w)
                state["basis_2d"] = basis_cpu.to(device=device)
                if freq_weights is not None:
                    state["freq_weights"] = freq_weights.to(device=device)
            state["cached_key"] = cache_key
        amp_scale = state["amp_scale"].to(device=device)

        # --- Step 1: Frequency modulation ---
        F_pred = torch.fft.rfft2(raw_pred.float())
        F_modulated = F_pred * amp_scale
        raw_modulated = torch.fft.irfft2(F_modulated, s=(H, W))

        # --- Step 2: DCT composition push on modulated result ---
        if effective_strength > 1e-6:
            # Set seed for reproducibility if specified
            # Note: DCT push only occurs at step 0, so step_index is always 0 here
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
                coeffs = torch.randn(B, n_modes, device=device, dtype=torch.float32, generator=generator)
            else:
                coeffs = torch.randn(B, n_modes, device=device, dtype=torch.float32)
            coeffs[:, 0] = 0.0  # Zero DC component

            if state["freq_weights"] is not None:
                coeffs = coeffs * state["freq_weights"]

            field = torch.einsum('nk,bk->bn', state["basis_2d"], coeffs)
            field = field.reshape(B, H, W)

            # Normalize field to unit std
            field_std = field.reshape(B, -1).std(dim=1).clamp(min=1e-8)
            field = field / field_std[:, None, None]
            field = field * effective_strength

            # Apply saturation/clamping
            if saturation_mode == "hard":
                min_scale = asymmetric_clamp[0] if asymmetric_clamp[0] is not None else 0.10
                max_scale = asymmetric_clamp[1] if asymmetric_clamp[1] is not None else (1.0 + clamp_val)
                scale = (1.0 + field).clamp(min=min_scale, max=max_scale).unsqueeze(1)
            else:
                scale = apply_saturation(field, saturation_mode, max_amp, clamp_val).unsqueeze(1)
            
            raw_new = raw_modulated * scale
        else:
            raw_new = raw_modulated
            scale = None

        # --- Energy compensation ---
        if energy_compensate:
            pred_rms = raw_pred.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            new_rms = raw_new.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            raw_new = raw_new * (pred_rms / new_rms)

        # --- Restore video shape if needed ---
        if is_video:
            raw_new = raw_new.view(B, C, T, H, W)
            raw_pred = raw_pred.view(B, C, T, H, W)

        # --- Logging ---
        if log.isEnabledFor(logging.INFO):
            with torch.no_grad():
                delta = (raw_new - raw_pred.float())
                delta_rms = delta.pow(2).mean().sqrt().item()
                pred_rms_val = raw_pred.float().pow(2).mean().sqrt().item()
                push_info = ""
                if scale is not None:
                    s_flat = scale.squeeze(1)
                    push_info = (
                        f"  push=[{s_flat.min().item():.4f}, {s_flat.max().item():.4f}]"
                        f"  strength={effective_strength:.3f}  clamp={clamp_val:.2f}  noise={noise_type}"
                    )
                mode_info = ""
                if mode != "butterworth":
                    mode_info = (
                        f"  mode={mode}  hf={effective_hf:.2f}  lf={effective_lf:.2f}"
                        f"  trans={transition:.1f}  sched={schedule}"
                    )
                extra_info = ""
                if midfreq_preserve > 0:
                    extra_info = f"  midfreq={midfreq_preserve:.2f}"
                if dct_basis_size != 4:
                    extra_info += f"  dct={dct_basis_size}x{dct_basis_size}"
                log.info(
                    "[DiversityBoost] step=%d  dc=%.2f"
                    "%s%s%s  shape=%s  delta_rms=%.4f  pred_rms=%.4f  ratio=%.4f",
                    step_index, effective_dc,
                    push_info, mode_info, extra_info, list(raw_pred.shape),
                    delta_rms, pred_rms_val,
                    delta_rms / max(pred_rms_val, 1e-8),
                )

        # --- Convert back ---
        modified = raw_to_denoised(raw_new, model).to(dtype=orig_dtype)
        return repack_video_if_needed(modified, pack_info)

    return diversity_hook
