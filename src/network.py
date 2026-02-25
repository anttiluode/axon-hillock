"""
AIS-Deerskin Network
====================

A population of coupled AIS-Deerskin oscillators with:
- Kuramoto coupling (phase synchronization/desynchronization)
- Hebbian coupling plasticity (co-active neurons strengthen connections)
- Full history tracking for analysis
"""

import numpy as np
from typing import Tuple, Optional
from .neuron import AISDeerskinNeuron


class AISDeerskinNetwork:
    """
    Population of coupled self-sculpting phase oscillators.
    
    Parameters
    ----------
    n_neurons : int
        Number of oscillators. Default 8.
    omega_range : tuple
        (min_omega, max_omega) for initial frequency spacing.
    sigma_init : float
        Initial σ for all neurons.
    coupling_init : float
        Initial coupling strength between adjacent neurons.
    coupling_adapt_rate : float
        Hebbian coupling learning rate.
    """
    
    def __init__(self, n_neurons: int = 8,
                 omega_range: Tuple[float, float] = (0.5, 10.0),
                 sigma_init: float = 1.0,
                 coupling_init: float = 0.5,
                 coupling_adapt_rate: float = 0.001):
        
        self.n = n_neurons
        self.coupling_adapt_rate = coupling_adapt_rate
        self.coupling_max = 5.0
        
        # Initialize neurons with evenly spaced frequencies
        omegas = np.linspace(omega_range[0], omega_range[1], n_neurons)
        self.neurons = [
            AISDeerskinNeuron(
                idx=i,
                omega=omegas[i],
                phase=np.random.uniform(0, 2 * np.pi),
                sigma=sigma_init
            )
            for i in range(n_neurons)
        ]
        
        # Coupling matrix: neighbors attract, distant neurons weakly repel
        self.coupling = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    continue
                dist = abs(i - j)
                if dist <= 2:
                    self.coupling[i, j] = coupling_init / dist
                else:
                    self.coupling[i, j] = -coupling_init * 0.1
        
        # History
        self.history = {
            'omegas': [], 'sigmas': [], 'phases': [],
            'activities': [], 'sync': [], 'outputs': [],
            'coupling_strength': []
        }
    
    def compute_kuramoto_nudges(self) -> np.ndarray:
        """Kuramoto coupling: each neuron nudged by weighted sin(phase difference)."""
        phases = np.array([n.phase for n in self.neurons])
        nudges = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                nudges[i] += self.coupling[i, j] * np.sin(phases[j] - phases[i])
        return nudges
    
    def compute_sync(self) -> float:
        """Kuramoto order parameter: 0=desynchronized, 1=fully synchronized."""
        phases = np.array([n.phase for n in self.neurons])
        return float(abs(np.mean(np.exp(1j * phases))))
    
    def adapt_coupling(self, outputs: np.ndarray):
        """
        Hebbian coupling plasticity.
        Co-active neurons strengthen connections; anti-correlated weaken.
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):
                correlation = outputs[i] * outputs[j]
                delta = self.coupling_adapt_rate * np.tanh(correlation * 10)
                self.coupling[i, j] = np.clip(
                    self.coupling[i, j] + delta,
                    -self.coupling_max, self.coupling_max
                )
                self.coupling[j, i] = self.coupling[i, j]
    
    def step(self, input_bands: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """
        Process one timestep.
        
        Parameters
        ----------
        input_bands : np.ndarray
            Shape [n_neurons]. One signal value per frequency band.
        dt : float
            Timestep duration.
            
        Returns
        -------
        np.ndarray
            Shape [n_neurons]. Output from each oscillator.
        """
        assert len(input_bands) == self.n
        
        nudges = self.compute_kuramoto_nudges()
        
        outputs = np.zeros(self.n)
        for i, neuron in enumerate(self.neurons):
            outputs[i] = neuron.tick(dt, input_bands[i], nudges[i])
        
        self.adapt_coupling(outputs)
        
        # Record history
        self.history['omegas'].append([n.omega for n in self.neurons])
        self.history['sigmas'].append([n.sigma for n in self.neurons])
        self.history['phases'].append([n.phase for n in self.neurons])
        self.history['activities'].append([n.activity_trace for n in self.neurons])
        self.history['sync'].append(self.compute_sync())
        self.history['outputs'].append(outputs.copy())
        self.history['coupling_strength'].append(np.mean(np.abs(self.coupling)))
        
        return outputs
    
    def reconstruct(self, outputs: Optional[np.ndarray] = None) -> float:
        """Sum of outputs = reconstructed signal."""
        if outputs is None:
            outputs = self.history['outputs'][-1]
        return np.sum(outputs)
    
    def disable_plasticity(self):
        """Freeze all adaptation (for comparison experiments)."""
        for neuron in self.neurons:
            neuron.sigma_adapt_rate = 0.0
            neuron.omega_adapt_rate = 0.0
        self.coupling_adapt_rate = 0.0
    
    def __repr__(self):
        lines = [f"AISDeerskinNetwork(n={self.n}, sync={self.compute_sync():.3f})"]
        for n in self.neurons:
            lines.append(f"  {n}")
        return "\n".join(lines)
