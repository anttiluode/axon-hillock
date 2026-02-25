"""
axon-hillock: Self-sculpting phase oscillator networks
"""
from .neuron import AISDeerskinNeuron
from .network import AISDeerskinNetwork
from .distance import deerskin_perceptual_distance

__version__ = "0.1.0"
__all__ = ["AISDeerskinNeuron", "AISDeerskinNetwork", "deerskin_perceptual_distance"]
