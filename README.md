# ComfyUI-DiversityBoost

**Restore composition diversity for distilled diffusion models. Training-free, single-step, zero model modification.**

> [中文版 README](README_zh.md)

## 📖 Table of Contents

- [The Problem](#the-problem)
- [The Solution (V3)](#the-solution-v3)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Complete Parameter Guide](#complete-parameter-guide)
- [Practical Usage: Creating Great Images](#practical-usage-creating-great-images)
  - [Basic Setup](#basic-setup)
  - [Recommended Settings by Use Case](#recommended-settings-by-use-case)
  - [Step-by-Step Workflow Examples](#step-by-step-workflow-examples)
  - [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Advanced Techniques](#advanced-techniques)
- [Node Reference](#node-reference)
- [Tested Models](#tested-models)
- [Tips & Best Practices](#tips--best-practices)
- [License](#license)

---

## The Problem

Step-distilled models like **FLUX2.[Klein] 9B**, **z-image-turbo**, and similar fast-generation models can produce high-quality images in just a few sampling steps. However, they suffer from a critical limitation called **composition collapse**:

### What is Composition Collapse?

When you generate images with different random seeds, you expect varied compositions:
- Different subject positions (left, right, center)
- Varied camera angles and perspectives  
- Different horizon lines in landscapes
- Diverse spatial arrangements of elements

**But distilled models don't do this.** Instead:
- A portrait prompt *always* puts the subject dead-center
- A landscape prompt *always* uses the same horizon line
- Different seeds produce nearly identical layouts with only minor texture variations

### Why Does This Happen?

During distillation (the process that makes models faster), the **spatial distribution of token norms gets frozen across seeds**. This locks the model into a single "average" composition regardless of the initial noise pattern. The model essentially learns one "safe" composition and repeats it.

---

## The Solution (V3)

**DiversityBoost V3** restores natural composition variation through two complementary mechanisms applied in a single post-CFG hook:

### 1. Polynomial Frequency Modulation
Applies smooth, continuous attenuation of high-frequency amplitude in the frequency domain. Key features:
- **Token-grid normalized**: Works consistently across different resolutions via DiT patch_size
- **Near-DC protected**: Prevents unwanted brightness or color shifts
- **Smooth transition**: No harsh artifacts from abrupt frequency cutoffs

### 2. DCT Composition Push
Applies a random low-frequency spatial field that redistributes energy across the latent space, nudging the model toward different compositions for each seed.

### The Result

The model freely reconstructs coherent details at subsequent steps, with per-seed noise driving different reconstruction paths. You get:
- ✅ **Varied compositions** across different seeds
- ✅ **Natural subject positioning** (not always centered)
- ✅ **Diverse spatial arrangements**
- ✅ **Zero model modification**
- ✅ **Zero training required**
- ✅ **One simple node**

---

## Quick Start

**Want to try it immediately?** Here's the fastest setup:

1. Install the custom node (see [Installation](#installation))
2. In ComfyUI, add the **Diversity Boost (V3)** node between your MODEL and KSampler
3. Use these default settings:
   - `strength`: **2.0**
   - `schedule`: **flat** (for euler sampler) or **linear** (for res_2m/heunpp2)
   - Leave all other parameters at defaults
4. Generate images with different seeds — compositions will now vary!

```
[Your Model] → [Diversity Boost (V3)] → [KSampler] → [VAE Decode] → [Save Image]
```

That's it! You now have composition diversity.

---

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-DiversityBoost.git
```

Restart ComfyUI.

### Method 2: Manual Download

1. Download the repository as a ZIP file
2. Extract to `ComfyUI/custom_nodes/ComfyUI-DiversityBoost`
3. Restart ComfyUI

### Requirements

**No extra dependencies!** Only requires PyTorch, which ships with ComfyUI.

### Verification

After installation and restart:
1. Open ComfyUI
2. Right-click in the canvas → Add Node → sampling
3. You should see **Diversity Boost (V3)** in the list

---

## How It Works

### Technical Overview

DiversityBoost intercepts the denoising process at step 0 and applies these transformations:

```
1. Convert model prediction → raw latent space
2. Apply polynomial frequency modulation (HF attenuation)
   ├─ Token-grid normalized via DiT patch_size
   ├─ Near-DC frequencies protected
   └─ Smooth polynomial transition (no harsh cutoffs)
3. Generate random DCT spatial field
   ├─ 4x4 low-frequency basis (configurable 2x2 to 8x8)
   ├─ Zero DC component (prevents brightness shift)
   └─ Pink/white/blue noise weighting
4. Normalize field to unit standard deviation
5. Apply multiplicative push: modulated × (1 + field)
6. Clamp to prevent dead zones: [0.1, 1+clamp]
7. Optional energy compensation
8. Convert back to model space
```

### Execution Timing

- **Primary effect**: Step 0 (initial denoising step)
- **Optional decay**: Progressive reduction over first ~3 steps with `linear` or `cosine` schedule
- **Model recovery**: Remaining steps reconstruct coherent details from the perturbed latent

### Why This Works

By operating on the **raw latent prediction** rather than the final output, DiversityBoost gives the model freedom to reconstruct details coherently while being guided toward different compositional choices. The per-seed noise drives different reconstruction paths, breaking the composition collapse.

---

## Complete Parameter Guide

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **strength** | 2.0 | 0.0 – 2.0 | Composition push amplitude. Controls how strongly the DCT field perturbs the latent. |
| **clamp** | 0.5 | 0.1 – 3.0 | Upper bound for multiplicative scale factor. Scale clamped to [min_scale, 1+clamp]. |
| **noise_type** | pink | pink / white / blue | Frequency spectrum of random DCT coefficients. |
| **dc_preserve** | 0.0 | 0.0 – 1.0 | DC amplitude preservation at step 0. Controls brightness/color retention. |
| **energy_compensate** | False | Boolean | Rescale output RMS to match original energy. |

### Frequency Modulation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **hf_factor** | 1.0 | 0.0 – 1.0 | High-frequency attenuation strength. 1.0 = full HF zeroing. |
| **lf_factor** | 0.3 | 0.0 – 1.0 | Low-frequency amplification. 1.0 = +50% boost. |
| **transition** | 2.0 | 0.5 – 4.0 | Polynomial transition shape. Controls smoothness of frequency roll-off. |
| **midfreq_preserve** | 0.0 | 0.0 – 1.0 | Mid-frequency detail preservation. Recovers structural details. |
| **midfreq_start** | 0.3 | 0.1 – 0.9 | Normalized start of mid-frequency band. |
| **midfreq_end** | 0.7 | 0.2 – 1.0 | Normalized end of mid-frequency band. |

### Scheduling Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| **schedule** | linear | flat / linear / cosine | Timestep schedule for applying the effect. |

### Advanced Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **dct_basis_size** | 4 | 2 – 8 | DCT basis dimension. Larger = finer local variations. |
| **seed** | 0 | 0 – 2³¹-1 | Random seed for reproducibility. 0 = random each generation. |
| **saturation_mode** | hard | hard / tanh / softplus | Saturation curve for DCT field values. |
| **max_amp** | 2.0 | 0.1 – 5.0 | Maximum amplitude for tanh/softplus saturation. |
| **asymmetric_clamp_min** | 0.0 | 0.0 – 1.0 | Minimum scale bound override. 0 = use default (0.10). |
| **asymmetric_clamp_max** | 0.0 | 0.0 – 5.0 | Maximum scale bound override. 0 = use default (1+clamp). |

---

### Detailed Parameter Explanations

#### strength (Composition Push Amplitude)

Controls how aggressively the DCT field redistributes spatial energy.

| Value | Effect | When to Use |
|-------|--------|-------------|
| **0.0** | HF cleanup only, no composition push | When you only want to reduce artifacts |
| **0.5** | Subtle composition variation | Portraits where you want minimal change |
| **1.0** | Moderate composition changes | General purpose, balanced diversity |
| **1.5** | Strong composition changes | Landscapes, architectural shots |
| **2.0** | Maximum composition changes (default) | When you want maximum diversity |

**Pro Tip**: Start at 2.0 and reduce if you see artifacts. Most scenes benefit from the default.

#### clamp (Safety Clamp)

Prevents extreme values in the multiplicative scale factor that could cause dead zones or blowouts.

- **Scale range**: `[max(0.1, asymmetric_clamp_min), 1 + max(clamp, asymmetric_clamp_max)]`
- **Lower values** (0.1–0.3): Conservative, prevents brightening
- **Default** (0.5): Balanced safety margin
- **Higher values** (1.0–3.0): Allows stronger contrast variations

**When to adjust**: Increase if you want more dramatic lighting variations; decrease if you see dark patches.

#### noise_type (Frequency Spectrum)

Determines the frequency characteristics of the random DCT coefficients.

| Type | Characteristics | Best For |
|------|----------------|----------|
| **pink** (1/f) | Lower frequencies dominate | **Recommended for most cases**. Produces smooth, natural composition shifts. Strong low-frequency composition modes. |
| **white** (flat) | All frequencies equal | Neutral baseline. Less predictable results. |
| **blue** (f) | Higher frequencies dominate | Fine-grained local variations. Can introduce texture-level changes. |

**Recommendation**: Stick with **pink** unless you have a specific reason to experiment.

#### dc_preserve (DC Amplitude Preservation)

Controls how much of the original DC (average brightness/color) component is preserved at step 0.

| Value | Effect |
|-------|--------|
| **0.0** (default) | Zero DC in the modulation. Maximum diversity. Brightness may shift slightly. |
| **0.5** | Partial DC preservation. Moderate brightness retention. |
| **1.0** | Full DC preservation. Original brightness maintained. |

**Important**: This only affects step 0. Steps 1+ always preserve full DC to prevent cumulative brightness shifts.

**When to use**: Set to 0.0 for maximum diversity. Increase to 0.5–1.0 if you notice unwanted brightness changes.

#### hf_factor (High-Frequency Attenuation)

Controls how aggressively high frequencies are attenuated during the frequency modulation step.

| Value | Effect |
|-------|--------|
| **0.0** | No HF attenuation. Only DCT push applies. |
| **0.5** | Moderate attenuation. HF reduced to ~50%. |
| **0.7** | Strong attenuation. HF reduced to ~30%. |
| **1.0** (default) | Full attenuation. HF effectively zeroed. |

**How it works**: Higher values create a blurrier "composition sketch" that the model then reconstructs, allowing more freedom in the reconstruction.

**Trade-off**: Higher `hf_factor` increases diversity but may reduce fine detail sharpness in the first step. The model recovers details in subsequent steps.

#### lf_factor (Low-Frequency Amplification)

Amplifies low-frequency components during modulation.

| Value | Effect |
|-------|--------|
| **0.0** | No LF amplification |
| **0.3** (default) | Moderate boost (~15%) |
| **0.5** | Stronger boost (~25%) |
| **1.0** | Maximum boost (+50%) |

**Purpose**: Enhances large-scale compositional elements (subject placement, horizon position) while HF attenuation handles fine details.

**Recommendation**: Default 0.3 works well. Increase for more dramatic compositional shifts.

#### transition (Polynomial Transition Shape)

Controls the smoothness of the frequency roll-off curve.

| Value | Shape | Effect |
|-------|-------|--------|
| **0.5** | Steep | Sharp transition, closer to Butterworth filter |
| **1.0** | Linear | Straight-line transition |
| **2.0** (default) | Smooth | Gentle, natural roll-off |
| **4.0** | Very smooth | Extremely gradual transition |

**Visual analogy**: Think of it as the "curve" adjustment for frequency response. Lower values = harsh cutoff; higher values = smooth blend.

**Recommendation**: Keep at 2.0 for most cases. Reduce to 1.0 if you want stronger HF attenuation without increasing `hf_factor`.

#### schedule (Timestep Schedule)

**CRITICAL PARAMETER** — determines when the effect is applied during sampling.

| Mode | Behavior | Sampler Compatibility |
|------|----------|----------------------|
| **flat** | Effect applied at step 0 only. Model has all remaining steps to recover. | ✅ **All samplers** (1st and 2nd order) |
| **linear** | HF attenuation decays linearly over first ~3 steps. DCT push still at step 0 only. | ⚠️ **2nd-order samplers only** (res_2m, heunpp2) |
| **cosine** | HF attenuation decays with cosine curve. Smoother than linear. | ⚠️ **2nd-order samplers only** (res_2m, heunpp2) |

**⚠️ IMPORTANT**: First-order samplers (euler, euler_ancestral) are sensitive to denoised modifications at step 1+. Using `linear` or `cosine` with these samplers can cause incomplete denoising or artifacts.

**Quick Decision**:
- Using **euler**? → Set `schedule = flat`
- Using **res_2m** or **heunpp2**? → Set `schedule = linear` or `cosine`

#### dct_basis_size (DCT Basis Dimension)

Controls the granularity of the composition push.

| Size | Effect | Best For |
|------|--------|----------|
| **2×2** (4 modes) | Very coarse, global shifts | Simple compositions, single subjects |
| **3×3** (9 modes) | Coarse | Basic scene layouts |
| **4×4** (16 modes, default) | Balanced | **Recommended for most cases** |
| **5×5** (25 modes) | Fine | Complex scenes with multiple elements |
| **6×6** to **8×8** | Very fine, local variations | Architectural details, intricate compositions |

**Trade-off**: Larger basis sizes provide more detailed control but may introduce localized artifacts. Start with 4×4 and increase only if needed.

#### seed (Random Seed Control)

Provides reproducibility for the DCT field generation.

| Value | Behavior |
|-------|----------|
| **0** (default) | Random seed each generation. Maximum unpredictability. |
| **>0** | Fixed seed. Same DCT field for相同的 input conditions. |

**Use cases for fixed seed**:
- Reproducing a specific composition you liked
- A/B testing other parameters while holding composition constant
- Batch generation with controlled variation

**Note**: This seed is separate from your KSampler seed. Changing the KSampler seed will still produce different images even with a fixed DiversityBoost seed.

#### saturation_mode (Saturation Curve)

Controls how the DCT field values are clamped/saturated.

| Mode | Behavior | Artifacts |
|------|----------|-----------|
| **hard** (default) | Hard clamping at bounds | Possible hard edges at extreme values |
| **tanh** | Smooth hyperbolic tangent curve | Minimal artifacts, natural roll-off |
| **softplus** | Smooth logarithmic curve | Very smooth, may reduce effect strength |

**Recommendation**: Use **hard** for maximum effect. Switch to **tanh** if you notice harsh transitions or banding artifacts.

#### midfreq_preserve (Mid-Frequency Detail Preservation)

Recovers structural details that might be lost during aggressive HF attenuation.

| Value | Effect |
|-------|--------|
| **0.0** (default) | No mid-frequency preservation |
| **0.3** | Moderate detail recovery |
| **0.5** | Strong detail recovery |
| **0.7+** | Aggressive preservation (may reduce diversity) |

**When to use**: Increase if you notice loss of important structural details (facial features, text, fine patterns) after applying strong HF attenuation.

#### midfreq_start / midfreq_end (Mid-Frequency Band)

Define the normalized frequency range for mid-frequency preservation.

- **midfreq_start**: Where preservation begins (0.1–0.9, default 0.3)
- **midfreq_end**: Where preservation ends (0.2–1.0, default 0.7)

**Example**: With `start=0.3` and `end=0.7`, frequencies between 30% and 70% of the maximum are preserved.

**Adjustment guide**:
- Lower `start` → Preserve more frequencies (less aggressive)
- Higher `end` → Preserve more frequencies (less aggressive)
- Narrow band (e.g., 0.4–0.6) → Targeted preservation

---

## Practical Usage: Creating Great Images

This section provides real-world workflows and recommended settings for common scenarios.

### Basic Setup

#### Minimal Workflow

```
[Load Checkpoint] 
       ↓
[Diversity Boost (V3)] ← Set strength=2.0, schedule=flat
       ↓
[KSampler] ← Set your preferred sampler (euler, res_2m, etc.)
       ↓
[VAE Decode]
       ↓
[Save Image]
```

**Connection Notes**:
- Connect your model output to the **model** input of Diversity Boost
- Connect Diversity Boost **model** output to KSampler **model** input
- No other connections needed — it's a model wrapper

#### Positive Prompt Example

```
A young woman standing in a sunlit forest, dappled light filtering through leaves, 
natural pose, candid photography, shallow depth of field, 85mm lens, f/1.8
```

#### Negative Prompt (if using)

```
blurry, low quality, distorted, oversaturated
```

**Note**: DiversityBoost does not require negative prompts, but they work normally if your workflow uses them.

---

### Recommended Settings by Use Case

#### 1. Portrait Photography

**Goal**: Natural subject positioning (not always centered), varied poses, diverse backgrounds.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 1.5 – 2.0 | Strong push for varied subject placement |
| schedule | flat (euler) or linear (res_2m) | Match to your sampler |
| noise_type | pink | Smooth, natural composition shifts |
| dct_basis_size | 4 | Good balance for human subjects |
| dc_preserve | 0.0 – 0.3 | Allow some brightness variation for mood |
| hf_factor | 0.8 – 1.0 | Strong HF attenuation for reconstruction freedom |
| midfreq_preserve | 0.2 – 0.4 | Preserve facial features and clothing details |

**Example Settings**:
```
strength: 1.8
schedule: flat
noise_type: pink
dct_basis_size: 4
dc_preserve: 0.2
hf_factor: 0.9
lf_factor: 0.3
transition: 2.0
midfreq_preserve: 0.3
```

**Expected Results**: Subjects appear at various positions (left third, right third, center), with different body orientations and background compositions.

---

#### 2. Landscape Photography

**Goal**: Varied horizon positions, diverse foreground/background relationships, different perspectives.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 2.0 | Maximum diversity for sweeping vistas |
| schedule | linear (with res_2m) | Progressive HF release for smooth gradients |
| noise_type | pink | Emphasizes large-scale composition |
| dct_basis_size | 3 – 4 | Coarser control for broad strokes |
| dc_preserve | 0.0 | Maximum diversity, sky brightness can vary |
| hf_factor | 1.0 | Full HF attenuation for cloud/sky reconstruction |
| lf_factor | 0.5 | Enhanced low-frequency for horizon placement |

**Example Settings**:
```
strength: 2.0
schedule: linear
noise_type: pink
dct_basis_size: 3
dc_preserve: 0.0
hf_factor: 1.0
lf_factor: 0.5
transition: 2.0
```

**Expected Results**: Horizons at different heights (rule of thirds, centered, low), varied cloud formations, diverse foreground element placement.

---

#### 3. Architectural Visualization

**Goal**: Different camera angles, building positions, perspective variations.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 1.5 – 2.0 | Strong geometric variation |
| schedule | flat | Safe for all samplers |
| noise_type | pink | Natural perspective shifts |
| dct_basis_size | 5 – 6 | Finer control for structural details |
| dc_preserve | 0.3 – 0.5 | Preserve lighting consistency |
| midfreq_preserve | 0.4 – 0.6 | Maintain architectural details |
| midfreq_start | 0.25 | Preserve more structural frequencies |

**Example Settings**:
```
strength: 1.8
schedule: flat
noise_type: pink
dct_basis_size: 5
dc_preserve: 0.4
hf_factor: 0.9
lf_factor: 0.4
transition: 2.0
midfreq_preserve: 0.5
midfreq_start: 0.25
midfreq_end: 0.75
```

**Expected Results**: Buildings positioned at different frame locations, varied camera angles (eye-level, low-angle, aerial), diverse perspective distortions.

---

#### 4. Product Photography

**Goal**: Varied product positioning, different angles, diverse background arrangements.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 1.0 – 1.5 | Moderate push to maintain product clarity |
| schedule | flat | Consistent results |
| noise_type | pink | Natural variations |
| dct_basis_size | 4 | Balanced control |
| dc_preserve | 0.5 – 0.7 | Maintain consistent product lighting |
| midfreq_preserve | 0.3 – 0.5 | Preserve product details and textures |

**Example Settings**:
```
strength: 1.2
schedule: flat
noise_type: pink
dct_basis_size: 4
dc_preserve: 0.6
hf_factor: 0.8
lf_factor: 0.3
transition: 2.0
midfreq_preserve: 0.4
```

**Expected Results**: Products shown from different angles (front, three-quarter, top-down), varied positions within frame, diverse prop arrangements.

---

#### 5. Fantasy / Concept Art

**Goal**: Dramatic compositional variety, unexpected perspectives, creative arrangements.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 2.0 | Maximum creative freedom |
| schedule | cosine (with res_2m) | Smoothest HF release for painterly results |
| noise_type | pink or blue | Pink for natural, blue for textured |
| dct_basis_size | 4 – 5 | Flexible for complex scenes |
| dc_preserve | 0.0 | Embrace dramatic lighting shifts |
| hf_factor | 1.0 | Full reconstruction freedom |
| lf_factor | 0.6 | Strong low-frequency for dramatic compositions |

**Example Settings**:
```
strength: 2.0
schedule: cosine
noise_type: pink
dct_basis_size: 5
dc_preserve: 0.0
hf_factor: 1.0
lf_factor: 0.6
transition: 2.5
```

**Expected Results**: Dramatic camera angles, varied element placements, unexpected perspectives, creative framing.

---

#### 6. Cleanup Only (No Composition Push)

**Goal**: Reduce artifacts from distilled models without changing composition.

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| strength | 0.0 | Disables DCT push |
| hf_factor | 0.5 – 0.7 | Moderate HF attenuation |
| schedule | flat | Safe for all samplers |

**Example Settings**:
```
strength: 0.0
hf_factor: 0.6
lf_factor: 0.0
schedule: flat
```

**Expected Results**: Slightly smoother images with reduced high-frequency artifacts, but compositions remain similar to base model.

---

### Step-by-Step Workflow Examples

#### Example 1: Portrait Series with Varied Compositions

**Scenario**: You're creating a series of portrait images and want each seed to produce a distinctly different composition.

**Step 1: Base Setup**
```
[FLUX2.Klein Checkpoint] → [Diversity Boost (V3)] → [KSampler] → [VAE Decode] → [Save Image]
```

**Step 2: Configure Diversity Boost**
```
strength: 1.8
clamp: 0.5
noise_type: pink
dc_preserve: 0.2
energy_compensate: False
hf_factor: 0.9
lf_factor: 0.3
transition: 2.0
schedule: flat
dct_basis_size: 4
seed: 0 (random)
saturation_mode: hard
```

**Step 3: Configure KSampler**
```
sampler_name: euler
scheduler: normal
steps: 4 (for FLUX2.Klein)
cfg: 1.0 (or model-recommended)
```

**Step 4: Write Prompt**
```
Positive: Professional headshot of a confident businesswoman, studio lighting, 
          neutral background, sharp focus, 85mm lens
  
Negative: blurry, deformed, low quality
```

**Step 5: Generate Multiple Seeds**
- Seed 100: Subject positioned left-third, slight turn to camera
- Seed 101: Subject centered, direct gaze
- Seed 102: Subject right-third, profile angle
- Seed 103: Subject lower-third, looking up

**Result**: Each seed produces a genuinely different composition, not just texture variations.

---

#### Example 2: Landscape Exploration with Second-Order Sampler

**Scenario**: You want to explore diverse landscape compositions using a higher-quality sampler.

**Step 1: Choose Sampler**
- Use **res_2m** or **heunpp2** (second-order samplers)

**Step 2: Configure Diversity Boost**
```
strength: 2.0
schedule: linear  ← Enables progressive HF decay
noise_type: pink
dct_basis_size: 3
dc_preserve: 0.0
hf_factor: 1.0
lf_factor: 0.5
transition: 2.0
```

**Step 3: Configure KSampler**
```
sampler_name: res_2m
steps: 6-8
```

**Step 4: Prompt**
```
Majestic mountain landscape at golden hour, alpine lake reflection, 
dramatic clouds, wilderness photography, wide angle
```

**Step 5: Batch Generate**
Generate 10-20 images with different seeds. Expect:
- Varied horizon heights (some sky-dominant, some land-dominant)
- Different mountain positions (left, right, center)
- Diverse cloud arrangements
- Varied lake positions and sizes

---

#### Example 3: A/B Testing with Fixed Seed

**Scenario**: You found a composition you like and want to test other parameters while keeping the composition constant.

**Step 1: Lock the DCT Seed**
```
seed: 12345  ← Fixed value
```

**Step 2: Generate Baseline**
- Keep all other parameters at defaults
- Note the composition

**Step 3: Test Parameter Variations**
- Test 1: Change `strength` from 2.0 to 1.5 (same seed = similar composition, weaker push)
- Test 2: Change `hf_factor` from 1.0 to 0.5 (same seed = similar composition, more detail)
- Test 3: Change `dct_basis_size` from 4 to 6 (same seed = similar overall layout, finer variations)

**Benefit**: Isolates the effect of individual parameters without composition changes confounding results.

---

#### Example 4: Combining with ControlNet

**Scenario**: You want composition diversity while maintaining some structural guidance.

**Workflow**:
```
[Checkpoint] → [ControlNet Loader] → [Apply ControlNet]
                      ↓
              [Diversity Boost (V3)] ← Applied AFTER ControlNet
                      ↓
                  [KSampler]
```

**Settings**:
```
strength: 1.0 – 1.5  ← Reduced to respect ControlNet guidance
schedule: flat
```

**Behavior**: 
- ControlNet provides structural guidance (edges, depth, pose)
- DiversityBoost adds variation within those constraints
- Result: Diverse interpretations of the same structural guide

**Tip**: Lower `strength` when using ControlNet to avoid fighting the guidance.

---

### Troubleshooting Common Issues

#### Issue 1: Images Are Too Dark or Too Bright

**Symptoms**: Generated images have incorrect overall brightness compared to expectations.

**Causes**:
- DC component being modified too aggressively
- Cumulative brightness shifts from multi-step application

**Solutions**:
1. Increase `dc_preserve` to 0.5 – 1.0
2. Ensure `schedule` is set to `flat` if using euler sampler
3. Enable `energy_compensate: True`

**Recommended Fix**:
```
dc_preserve: 0.7
energy_compensate: True
```

---

#### Issue 2: Visible Banding or Harsh Transitions

**Symptoms**: Smooth gradients show visible bands; transitions look artificial.

**Causes**:
- Hard clamping creating discontinuities
- Too aggressive HF attenuation without mid-frequency preservation

**Solutions**:
1. Change `saturation_mode` from `hard` to `tanh`
2. Increase `midfreq_preserve` to 0.3 – 0.5
3. Reduce `hf_factor` slightly (0.8 instead of 1.0)

**Recommended Fix**:
```
saturation_mode: tanh
midfreq_preserve: 0.4
hf_factor: 0.85
```

---

#### Issue 3: Incomplete Denoising / Artifacts at Step 1+

**Symptoms**: Images look unfinished, have residual noise, or show strange artifacts.

**Causes**:
- Using `linear` or `cosine` schedule with first-order samplers (euler)
- Effect being applied at steps where the sampler can't handle it

**Solutions**:
1. **Most likely**: Change `schedule` to `flat`
2. Switch to a second-order sampler (res_2m, heunpp2) if you want progressive decay
3. Reduce `strength` if problem persists

**Quick Diagnosis**:
- Using euler? → Must use `schedule: flat`
- Using res_2m? → Can use `schedule: linear` or `cosine`

---

#### Issue 4: Loss of Fine Details

**Symptoms**: Text becomes unreadable, facial features blur, fine textures disappear.

**Causes**:
- Overly aggressive HF attenuation destroying important high-frequency information
- No mid-frequency preservation enabled

**Solutions**:
1. Enable `midfreq_preserve: 0.4 – 0.6`
2. Adjust `midfreq_start` to 0.25 and `midfreq_end` to 0.75
3. Reduce `hf_factor` to 0.6 – 0.8

**Recommended Fix**:
```
midfreq_preserve: 0.5
midfreq_start: 0.25
midfreq_end: 0.75
hf_factor: 0.7
```

---

#### Issue 5: Compositions Still Look Similar

**Symptoms**: Different seeds still produce nearly identical layouts.

**Causes**:
- `strength` too low
- Model not sufficiently distilled (doesn't need DiversityBoost)
- Wrong parameter combination

**Solutions**:
1. Increase `strength` to 2.0 (maximum)
2. Verify you're using a distilled model (FLUX2.Klein, z-image-turbo, etc.)
3. Try `noise_type: blue` for different variation characteristics
4. Increase `dct_basis_size` to 5 or 6 for finer variations

**Debugging Steps**:
1. Set `strength: 2.0`
2. Set `dc_preserve: 0.0` (maximum diversity)
3. Generate 5-10 images with different seeds
4. If still similar, the model may not be distilled enough to benefit

---

#### Issue 6: Dark Patches or Dead Zones

**Symptoms**: Parts of the image are unnaturally dark or lack detail.

**Causes**:
- Multiplicative scale going too low in某些 areas
- Clamping not aggressive enough

**Solutions**:
1. Increase `clamp` value (allows higher upper bound, indirectly helps)
2. Set `asymmetric_clamp_min` to 0.15 – 0.2 (raises minimum scale)
3. Reduce `strength` slightly

**Recommended Fix**:
```
asymmetric_clamp_min: 0.2
clamp: 0.7
strength: 1.8
```

---

#### Issue 7: Color Shifts Between Seeds

**Symptoms**: Different seeds produce images with noticeably different color casts.

**Causes**:
- DC component modification affecting color channels differently
- Energy compensation disabled

**Solutions**:
1. Increase `dc_preserve` to 0.8 – 1.0
2. Enable `energy_compensate: True`
3. Consider using `saturation_mode: tanh` for smoother transitions

**Recommended Fix**:
```
dc_preserve: 0.9
energy_compensate: True
```

---

## Advanced Techniques

### Technique 1: Progressive Strength Scaling

For batch generation where you want a range of diversity levels:

**Setup**:
- Create multiple DiversityBoost nodes with different `strength` values
- Use a switch or batch processing to cycle through them

**Example**:
- Node A: `strength: 0.5` (subtle)
- Node B: `strength: 1.25` (moderate)
- Node C: `strength: 2.0` (strong)

**Use Case**: Exploring the diversity "spectrum" for a given prompt to find the optimal level.

---

### Technique 2: Seed Chaining for Controlled Variation

Combine fixed DiversityBoost seed with varying KSampler seeds:

**Setup**:
```
DiversityBoost seed: 12345 (fixed)
KSampler seed: [100, 101, 102, ...] (varying)
```

**Effect**:
- Composition framework stays similar (same DCT field)
- Details, textures, and fine elements vary (different sampler noise)

**Use Case**: Creating a series with consistent composition but varied execution.

---

### Technique 3: Resolution-Specific Tuning

Different resolutions may benefit from adjusted parameters:

**For High Resolution (1024×1024+)**:
```
dct_basis_size: 5 – 6  ← More modes for finer control
midfreq_preserve: 0.4  ← Preserve more detail
hf_factor: 0.9         ← Slightly less aggressive
```

**For Low Resolution (512×512)**:
```
dct_basis_size: 3 – 4  ← Fewer modes, coarser is fine
midfreq_preserve: 0.2  ← Less critical
hf_factor: 1.0         ← Full attenuation OK
```

**Rationale**: Higher resolutions have more tokens, allowing finer DCT basis without artifacts.

---

### Technique 4: Genre-Specific Presets

Create and save presets for different genres:

**Portrait Preset**:
```json
{
  "strength": 1.8,
  "schedule": "flat",
  "noise_type": "pink",
  "dct_basis_size": 4,
  "dc_preserve": 0.2,
  "hf_factor": 0.9,
  "midfreq_preserve": 0.3
}
```

**Landscape Preset**:
```json
{
  "strength": 2.0,
  "schedule": "linear",
  "noise_type": "pink",
  "dct_basis_size": 3,
  "dc_preserve": 0.0,
  "hf_factor": 1.0,
  "lf_factor": 0.5
}
```

**Architecture Preset**:
```json
{
  "strength": 1.8,
  "schedule": "flat",
  "noise_type": "pink",
  "dct_basis_size": 5,
  "dc_preserve": 0.4,
  "midfreq_preserve": 0.5
}
```

**Cleanup Preset**:
```json
{
  "strength": 0.0,
  "hf_factor": 0.6,
  "schedule": "flat"
}
```

---

### Technique 5: Combining with Other Model Patches

DiversityBoost operates on a different hook than most model patches, making it compatible with:

- **ControlNet**: Apply DiversityBoost after ControlNet
- **IPAdapter**: Fully compatible
- **LoRA**: Fully compatible
- **Other model merges**: Compatible

**Recommended Order**:
```
[Base Model] → [LoRA/ControlNet/IPAdapter] → [DiversityBoost] → [KSampler]
```

**Note**: If combining with other post-cfg functions, be aware of potential interactions. Test incrementally.

---

## Node Reference

### Diversity Boost (V3) — `DiversityBoostCoreV3`

**Category**: sampling

**Description**: Restore composition diversity for distilled models using polynomial HF modulation and DCT composition push.

**Inputs**:
- `model` (MODEL): Input model to modify

**Outputs**:
- `model` (MODEL): Modified model with diversity hook

**Parameters**: See [Complete Parameter Guide](#complete-parameter-guide)

---

### Diversity Boost (Legacy) — `DiversityBoostCore`

**Category**: sampling

**Description**: Legacy node using Butterworth LPF with `n_periods` parameter. Kept for backward compatibility.

**When to Use**: Only if you have existing workflows that depend on the old behavior.

**Key Difference**: Uses steep Butterworth filter instead of smooth polynomial modulation.

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| strength | 0.5 | Composition push amplitude |
| clamp | 1.0 | Scale factor upper bound |
| noise_type | pink | DCT coefficient spectrum |
| n_periods | 2 | Butterworth cutoff (spatial periods to preserve) |
| dc_preserve | 0.0 | DC amplitude preservation |
| energy_compensate | False | Rescale output RMS |

**Recommendation**: Use V3 for all new workflows.

---

## Tested Models

| Model | Status | Notes |
|-------|--------|-------|
| **FLUX2.[Klein] 9B** | ✅ Tested | Primary target, excellent results |
| **z-image-turbo** | ✅ Tested | Strong composition diversity restored |
| **Other distilled models** | 🟡 User-reported | Report your results! |

**What Makes a Model "Distilled"?**
- Few-step generation (1-8 steps)
- Trained via distillation from a larger teacher model
- Often has "turbo", "lightning", "distilled", or version numbers like "2." in the name

**If Your Model Isn't Distilled**: DiversityBoost may still work but effects will be subtler. The tool is specifically designed to fix composition collapse caused by distillation.

---

## Tips & Best Practices

### General Tips

1. **Start with V3 defaults** — They're tuned for strong diversity with minimal side effects
2. **Match schedule to sampler** — `flat` for euler, `linear`/`cosine` for res_2m/heunpp2
3. **Generate multiple seeds** — The whole point is to see variation across seeds
4. **Don't overthink it** — Defaults work well for most cases

### Workflow Tips

5. **Place node correctly** — Between model and KSampler, not after VAE
6. **Compatible with ControlNet** — But reduce `strength` to 1.0–1.5
7. **Batch test first** — Generate 10 seeds before committing to a parameter set
8. **Save presets** — Create genre-specific presets for quick switching

### Quality Tips

9. **Watch for brightness shifts** — Increase `dc_preserve` if needed
10. **Preserve details** — Use `midfreq_preserve` for text, faces, fine patterns
11. **Smooth artifacts** — Switch to `tanh` saturation mode if you see banding
12. **Second-order samplers** — Use res_2m for highest quality with progressive schedules

### Creative Tips

13. **Embrace randomness** — Leave `seed: 0` for maximum surprise
14. **Experiment with noise_type** — Blue noise for textured variations
15. **Combine with other tools** — Works with LoRA, IPAdapter, ControlNet
16. **Document what works** — Keep notes on successful parameter combinations

---

## Comparison: Before vs After

### Without DiversityBoost

```
Seed 100: Subject centered, horizon at 50%, clouds in upper-right
Seed 101: Subject centered, horizon at 50%, clouds in upper-right (slightly different cloud shape)
Seed 102: Subject centered, horizon at 50%, clouds in upper-right (different texture)
Seed 103: Subject centered, horizon at 50%, clouds in upper-right (minor variation)
```

**Problem**: Same composition, only texture differences.

### With DiversityBoost

```
Seed 100: Subject left-third, horizon at 40%, dramatic sky
Seed 101: Subject right-third, horizon at 60%, balanced composition
Seed 102: Subject centered, horizon at 30%, sky-dominant
Seed 103: Subject lower-left, horizon at 70%, land-dominant
```

**Result**: Genuinely different compositions, each seed produces unique framing.

---

## Performance Impact

- **Speed**: Negligible (<1% overhead)
- **Memory**: Minimal (cached frequency tensors)
- **Compatibility**: Works with all ComfyUI samplers (with correct schedule setting)

---

## FAQ

**Q: Do I need this if I'm not using distilled models?**  
A: Probably not. DiversityBoost specifically addresses composition collapse caused by distillation. Full-size models (non-distilled) typically already have good composition diversity.

**Q: Can I use this with img2img?**  
A: Yes, but effects may be subtler since you're starting from an existing image rather than pure noise.

**Q: Will this work with video generation?**  
A: The code includes video support (unpack/repack), but it's primarily tested for still images. Report your results!

**Q: How many steps should I use?**  
A: Use whatever steps your model recommends. DiversityBoost doesn't change optimal step counts. For FLUX2.Klein, that's typically 4 steps.

**Q: Can I automate preset switching?**  
A: Yes, use ComfyUI's workflow features or external automation to swap parameter sets between batches.

---

## License

MIT License — Free for personal and commercial use.

---

## Contributing

Feel free to:
- Report results with other distilled models
- Share successful parameter presets
- Submit bug reports or feature requests

---

## Acknowledgments

Built on the insight that distillation freezes spatial distributions, and that frequency-domain manipulation can restore diversity without model modification.

---

**Happy generating! May your compositions be ever diverse.** 🎨✨
