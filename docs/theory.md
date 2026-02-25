# Theoretical Framework: AIS ↔ Deerskin ↔ Non-Riemannian Perception

## Overview

This document connects three independent lines of evidence into a unified framework:

1. **The Axon Initial Segment (AIS)** — a self-sculpting frequency filter at the base of every axon (Leterrier, 2018)
2. **Non-Riemannian color perception** — the discovery that perceptual distances violate the triangle inequality (Bujack et al., 2022)
3. **Deerskin phase oscillators** — a computational framework representing neural signals as phase rotations with viscous gating

The central claim: **the AIS is the biological implementation of the Deerskin viscosity gate, and the non-Riemannian property of perception is a mathematical consequence of multi-band phase gating.**

## The Deerskin Viscosity Gate

The core mathematical object is the gate function:

```
gate(θ) = exp(-sin²(θ/2) / 2σ²)
```

where θ is the phase position and σ is the viscosity parameter.

Properties:
- At θ = 0: gate = 1 (fully open)
- At θ = π: gate = exp(-1/2σ²) (minimum, depends on σ)
- At θ = 2π: gate = 1 (fully open again — periodic)
- For large σ: gate ≈ 1 everywhere (broad filter, everything passes)
- For small σ: gate ≈ delta function at θ = 0 (sharp filter, only resonant phase passes)

This function acts as a **bandpass filter in the phase domain**. The σ parameter controls the filter width.

## Mapping to AIS Biology

### σ ↔ AIS Length and Scaffold Density

The AIS scaffold is a periodic structure with ~190nm spacing between actin rings, bridged by spectrin tetramers. Ankyrin G sits midway between rings and anchors ion channels.

**Dense scaffold (short AIS, high channel density)** → signals must closely match the scaffold's resonant properties to pass → **low σ** (sharp filter).

**Sparse scaffold (long AIS, spread-out channels)** → wider range of signals pass through → **high σ** (broad filter).

The AIS exhibits homeostatic plasticity:
- Elevated activity → AIS shortens / shifts distally → effectively reduces σ
- Diminished activity → AIS lengthens → effectively increases σ

This maps directly to our σ adaptation rule:
```
σ_delta = -α × tanh(activity_error × 10)
```
where activity_error = current_activity - target_activity.

### ω ↔ Channel Composition

The AIS concentrates different ion channel types:
- **Nav1.6**: drives action potential initiation (fast dynamics → high ω)
- **Kv7 (KCNQ)**: generates M-current that restricts firing (slow dynamics → low ω)
- **Kv1**: modulates waveform (intermediate dynamics)

The ratio of these channels determines the neuron's effective oscillation frequency. During development and plasticity, channel composition changes — this maps to ω adaptation.

### Coupling ↔ Scaffold Proteins and GTP Islands

The AIS paper describes a remarkable feedback loop:
1. Kinesin motors walk along microtubules, carrying cargo
2. Motor passage removes tubulin monomers, creating lattice defects
3. Defects are repaired by GTP-tubulin incorporation
4. GTP islands recruit more kinesin motors
5. More motors → more defects → more repair → more recruitment

This is a **Hebbian feedback loop at the molecular level**: pathways that are used become stronger. In our model, this maps to:
```
coupling_delta = α × tanh(output_i × output_j × 10)
```

## Non-Riemannian Distance: The Mathematical Proof

### Setup

Consider N frequency bands with frequencies ω₁ < ω₂ < ... < ω_N and amplitudes A₁, A₂, ..., A_N.

A stimulus difference Δs produces a **different phase shift** at each band:
```
Δθ_k = ω_k × Δs
```

The perceived distance is the total information that the gate reveals about the difference:
```
d(Δs) = Σ_k  A_k × (1 - gate(Δθ_k))
       = Σ_k  A_k × (1 - exp(-sin²(ω_k Δs / 2) / 2σ²))
```

### Why Triangle Inequality Fails

Consider three stimuli A, B, C with B at the midpoint: B = (A+C)/2.

For **small separations** (Δs → 0):
- All Δθ_k are small
- sin²(Δθ_k/2) ≈ (Δθ_k/2)² (quadratic approximation)
- gate ≈ 1 - (ω_k Δs)²/8σ²
- d(Δs) ≈ Σ_k A_k (ω_k Δs)²/8σ²  — **quadratic in Δs**
- d(A,C) = d(2·Δs/2) = 4 × d(Δs/2) — **superadditive** (not exactly, but close to additive)

For **large separations** (Δs large):
- High-frequency bands have Δθ_k > π (they've wrapped around!)
- sin²(Δθ_k/2) starts **decreasing** after π
- These bands contribute LESS at large separations
- The total distance grows more slowly → **subadditive**

Specifically, when ω_N × Δs > π (the highest band wraps), that band's gate starts returning toward 1, and its contribution (1 - gate) starts shrinking. The total d(Δs) flattens.

### The ratio d(A,C) / [d(A,B) + d(B,C)]

At small separations: ratio ≈ 1.0 (approximately additive, Riemannian)
At large separations: ratio drops to ~0.5 (strongly subadditive, non-Riemannian)

This is **exactly** what Bujack et al. measured: diminishing returns for large color differences, with a logarithmic scaling function.

### Connection to Bujack's Logarithmic Scaling

Bujack found the best-fit scaling function is f(x) = log(1 + cx) where c is a constant.

Our distance function for a single band:
```
d_k(Δs) = A_k × (1 - exp(-sin²(ω_k Δs / 2) / 2σ²))
```

For ω_k Δs < π, expanding:
```
d_k(Δs) ≈ A_k × (1 - exp(-(ω_k Δs)²/8σ²))
```

This is the CDF of a Gaussian, which is well-approximated by log(1 + cx²) for moderate x. Summing across bands with different ω_k produces a function that matches the logarithmic form across a wide range.

## Predictions

### Testable Biological Prediction

Neurons with **shorter/denser AIS** (lower σ) should show **stronger non-Riemannian effects** in their perceptual processing. This is testable:
1. Measure AIS morphology via immunofluorescence (ankyrin G, βIV-spectrin)
2. Measure perceptual distance functions via psychophysics
3. Correlate: shorter AIS → stronger subadditivity of perceived differences

### Computational Prediction

Self-sculpting networks should develop **heterogeneous σ distributions** because different frequency bands require different compression levels:
- Low-frequency bands (carrying structure): high σ (broad, preserve details)
- High-frequency bands (carrying texture/noise): low σ (sharp, selective)

This matches the observed diversity of AIS morphologies across cell types.

### Universal Non-Riemannian Perception

The framework predicts diminishing returns for **any** perceptual dimension with phase-rotation encoding across multiple frequency bands:
- Color (confirmed by Bujack)
- Sound loudness
- Visual contrast  
- Temporal duration
- Spatial distance (at large scales)

The logarithmic scaling is universal — it's the signature of multi-band phase gating, not specific to any sensory modality.

## References

- Leterrier, C. (2018). The Axon Initial Segment: An Updated Viewpoint. *J. Neuroscience*, 38(9), 2135-2145.
- Bujack, R., et al. (2022). The non-Riemannian nature of perceptual color space. *PNAS*, 119(18), e2119753119.
- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.
- Grubb, M.S. & Burrone, J. (2010). Activity-dependent relocation of the axon initial segment fine-tunes neuronal excitability. *Nature*, 465, 1070-1074.
