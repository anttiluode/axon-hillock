"""
AIS-Deerskin Neuron
===================

A phase oscillator with AIS-like self-sculpting plasticity.

The neuron maintains phase state and processes input through phase rotation 
gated by viscosity. Unlike vanilla Deerskin oscillators, ALL parameters adapt 
based on signal history.

Plasticity rules (derived from AIS biology):

1. σ adaptation (viscosity / AIS length):
   - High average activity → σ decreases (AIS shortens, sharper filter)
   - Low average activity → σ increases (AIS lengthens, broader filter)
   - Target activity is frequency-dependent (low ω → low target, high ω → high target)

2. ω adaptation (frequency / channel composition):
   - ω drifts toward the dominant frequency in input
   - Driven by instantaneous frequency estimation from phase velocity
   - Maps to AIS channel composition changes (Nav ↔ Kv balance)

3. Coupling (external, managed by Network):
   - Hebbian: co-active neurons strengthen connections
   - Maps to scaffold protein accumulation and GTP island feedback
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AISDeerskinNeuron:
    """
    A single self-sculpting phase oscillator.
    
    Parameters
    ----------
    idx : int
        Neuron index in the population
    omega : float
        Natural frequency (rad/s). Maps to AIS channel composition.
    phase : float
        Current phase (rad).
    sigma : float
        Viscosity parameter. Controls gate width. Maps to AIS length/density.
    sigma_adapt_rate : float
        How fast σ adapts. Default 0.01.
    omega_adapt_rate : float
        How fast ω drifts toward input frequency. Default 0.005.
    activity_tau : float
        EMA decay constant for activity trace. Default 0.95.
    sigma_min, sigma_max : float
        Bounds on σ adaptation. AIS can't grow/shrink infinitely.
    omega_min, omega_max : float
        Bounds on ω adaptation.
    """
    idx: int = 0
    omega: float = 1.0
    phase: float = 0.0
    sigma: float = 1.0
    
    # Plasticity state
    activity_trace: float = 0.0
    phase_velocity: float = 0.0
    total_signal_passed: float = 0.0
    adaptation_age: float = 0.0
    
    # Input frequency tracking for ω adaptation
    _prev_input: float = 0.0
    _input_freq_estimate: float = 0.0
    
    # Plasticity rates
    sigma_adapt_rate: float = 0.01
    omega_adapt_rate: float = 0.005
    activity_tau: float = 0.95
    
    # Bounds
    sigma_min: float = 0.05
    sigma_max: float = 5.0
    omega_min: float = 0.1
    omega_max: float = 20.0
    
    def tick(self, dt: float, input_signal: float, coupling_nudge: float = 0.0) -> float:
        """
        Process one timestep.
        
        Parameters
        ----------
        dt : float
            Timestep duration
        input_signal : float
            Scalar input amplitude for this neuron's frequency band
        coupling_nudge : float
            Kuramoto-style phase adjustment from coupled neighbors
            
        Returns
        -------
        float
            Gated output signal
        """
        self.adaptation_age += 1
        
        # --- Phase evolution ---
        self.phase_velocity = 0.85 * self.phase_velocity + 0.15 * coupling_nudge
        effective_omega = self.omega + self.phase_velocity
        self.phase += effective_omega * dt
        self.phase %= (2 * np.pi)
        
        # --- Viscosity gate ---
        # gate(θ) = exp(-sin²(θ/2) / 2σ²)
        # At θ=0: gate=1 (full pass). At θ=π: gate=exp(-1/2σ²) (minimum).
        gate = np.exp(-(np.sin(self.phase / 2.0) ** 2) / (2 * self.sigma ** 2 + 1e-8))
        
        # Output: input × gate × cos(phase)
        output = input_signal * gate * np.cos(self.phase)
        output_energy = output ** 2
        
        # --- Plasticity ---
        
        # 1. Activity trace (EMA)
        self.activity_trace = (self.activity_tau * self.activity_trace + 
                               (1 - self.activity_tau) * output_energy)
        
        # 2. σ adaptation (homeostatic)
        # Frequency-dependent target: higher ω → higher target → stabilizes at lower σ
        omega_norm = ((self.omega - self.omega_min) / 
                      (self.omega_max - self.omega_min + 1e-8))
        target_activity = 0.02 + 0.08 * omega_norm
        activity_error = self.activity_trace - target_activity
        sigma_delta = -self.sigma_adapt_rate * np.tanh(activity_error * 10.0)
        self.sigma = np.clip(self.sigma + sigma_delta, self.sigma_min, self.sigma_max)
        
        # 3. ω adaptation (frequency tracking via zero-crossing rate)
        # Estimate input frequency from sign changes
        sign_change = (input_signal * self._prev_input < 0)
        if sign_change:
            # Instantaneous frequency estimate from zero-crossing
            inst_freq = np.pi / (dt + 1e-8)  # half-period → frequency
            self._input_freq_estimate = (0.99 * self._input_freq_estimate + 
                                         0.01 * inst_freq)
        self._prev_input = input_signal
        
        # Pull ω toward estimated input frequency, modulated by input strength
        if self.adaptation_age > 100:  # wait for frequency estimate to stabilize
            input_strength = min(abs(input_signal), 2.0)
            freq_target = np.clip(self._input_freq_estimate, self.omega_min, self.omega_max)
            omega_delta = self.omega_adapt_rate * input_strength * (freq_target - self.omega)
            self.omega = np.clip(self.omega + omega_delta, self.omega_min, self.omega_max)
        
        # 4. Cumulative signal tracking
        self.total_signal_passed += abs(output) * dt
        
        return output
    
    def gate_value(self) -> float:
        """Current gate transparency (0=closed, 1=open)."""
        return np.exp(-(np.sin(self.phase / 2.0) ** 2) / (2 * self.sigma ** 2 + 1e-8))
    
    def state_dict(self) -> dict:
        """Full state for analysis."""
        return {
            'idx': self.idx,
            'omega': self.omega,
            'phase': self.phase,
            'sigma': self.sigma,
            'activity': self.activity_trace,
            'gate': self.gate_value(),
            'total_signal': self.total_signal_passed,
            'age': self.adaptation_age
        }
    
    def __repr__(self):
        return (f"AISDeerskinNeuron(idx={self.idx}, ω={self.omega:.2f}, "
                f"σ={self.sigma:.3f}, activity={self.activity_trace:.4f})")
