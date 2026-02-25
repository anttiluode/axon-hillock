"""
Non-Riemannian Distance
=======================

The viscosity gate exp(-sin²(Δθ/2) / 2σ²) applied across multiple frequency 
bands produces subadditive (non-Riemannian) perceptual distances.

In the Deerskin framework, stimuli are encoded as phase rotations. The
"distance" between two stimuli involves rotating through Δθ = ω × Δstimulus
at each frequency band, with each band contributing (1 - gate(Δθ_k)) to
the total perceived difference.

For small separations: all bands contribute linearly → Riemannian (additive).
For large separations: high-ω bands wrap around (Δθ > π), contributing LESS 
→ subadditive → non-Riemannian → Bujack's diminishing returns.

This module provides functions to compute and analyze these distances.
"""

import numpy as np
from typing import Optional, Tuple


def deerskin_perceptual_distance(delta_stim: float, sigma: float,
                                  n_bands: int = 8,
                                  omega_range: Tuple[float, float] = (0.5, 10.0),
                                  amplitudes: Optional[np.ndarray] = None) -> float:
    """
    Compute perceived distance for a stimulus difference through the Deerskin gate.
    
    Each frequency band k converts the stimulus difference into a phase difference:
        Δθ_k = ω_k × delta_stim
    
    The contribution of band k to perceived distance:
        contribution_k = (1 - gate(Δθ_k)) × amplitude_k
    
    where gate(Δθ) = exp(-sin²(Δθ/2) / 2σ²)
    
    Parameters
    ----------
    delta_stim : float
        Stimulus difference magnitude
    sigma : float
        Viscosity parameter
    n_bands : int
        Number of frequency bands
    omega_range : tuple
        (min_omega, max_omega) for band frequencies
    amplitudes : np.ndarray, optional
        Band amplitudes. Default: 1/f-like spectrum.
        
    Returns
    -------
    float
        Perceived distance (always ≥ 0)
    """
    omegas = np.linspace(omega_range[0], omega_range[1], n_bands)
    
    if amplitudes is None:
        amplitudes = 1.0 / (1 + np.arange(n_bands) * 0.3)
    
    total = 0.0
    for k in range(n_bands):
        delta_theta = omegas[k] * delta_stim
        gate = np.exp(-(np.sin(delta_theta / 2.0) ** 2) / (2 * sigma ** 2 + 1e-8))
        total += (1.0 - gate) * amplitudes[k]
    
    return total


def triangle_ratio(total_sep: float, sigma: float, **kwargs) -> float:
    """
    Compute d(A,C) / [d(A,B) + d(B,C)] for midpoint B.
    
    Ratio < 1.0 → subadditive (non-Riemannian)
    Ratio = 1.0 → additive (Riemannian)
    
    Parameters
    ----------
    total_sep : float
        Total separation between A and C
    sigma : float
        Viscosity parameter
    **kwargs
        Passed to deerskin_perceptual_distance
        
    Returns
    -------
    float
        Triangle inequality ratio
    """
    d_AC = deerskin_perceptual_distance(total_sep, sigma, **kwargs)
    d_AB = deerskin_perceptual_distance(total_sep / 2, sigma, **kwargs)
    d_BC = deerskin_perceptual_distance(total_sep / 2, sigma, **kwargs)
    return d_AC / (d_AB + d_BC + 1e-10)


def concavity_fraction(sigma: float, sep_range: Tuple[float, float] = (0.0, 3.0),
                       n_points: int = 50, **kwargs) -> float:
    """
    Fraction of the distance function that is concave (d²d/ds² < 0).
    
    Concavity → log-like → non-Riemannian → Bujack's result.
    
    Returns
    -------
    float
        Fraction in [0, 1]. Values > 0.5 indicate mostly concave (log-like).
    """
    seps = np.linspace(sep_range[0], sep_range[1], n_points)
    dists = [deerskin_perceptual_distance(s, sigma, **kwargs) for s in seps]
    d2 = np.diff(np.diff(dists))
    return float(np.mean(d2 < 0))


def scan_sigma_nonriemannian(sigma_range: Tuple[float, float] = (0.1, 5.0),
                              n_sigma: int = 20,
                              sep_range: Tuple[float, float] = (0.1, 3.0),
                              n_sep: int = 30,
                              **kwargs) -> dict:
    """
    Scan σ values and measure non-Riemannian strength at each.
    
    Returns
    -------
    dict with keys:
        'sigmas': array of σ values
        'ratio_small': mean triangle ratio at small separations
        'ratio_large': mean triangle ratio at large separations
        'ratio_drop': (ratio_small - ratio_large) / ratio_small
        'concavity': fraction of distance function that is concave
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    seps = np.linspace(sep_range[0], sep_range[1], n_sep)
    
    result = {
        'sigmas': sigmas,
        'ratio_small': [],
        'ratio_large': [],
        'ratio_drop': [],
        'concavity': []
    }
    
    for sigma in sigmas:
        ratios = [triangle_ratio(s, sigma, **kwargs) for s in seps]
        rs = np.mean(ratios[:n_sep // 4])
        rl = np.mean(ratios[-n_sep // 4:])
        
        result['ratio_small'].append(rs)
        result['ratio_large'].append(rl)
        result['ratio_drop'].append((rs - rl) / (rs + 1e-10))
        result['concavity'].append(concavity_fraction(sigma, sep_range, **kwargs))
    
    for k in ['ratio_small', 'ratio_large', 'ratio_drop', 'concavity']:
        result[k] = np.array(result[k])
    
    return result
