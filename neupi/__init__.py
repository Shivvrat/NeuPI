# neupi/__init__.py
"""NeuPI: Neural-Probabilistic Inference Library"""

# Expose the version of the package
__version__ = "0.1.0"

# Expose the factory function as the main entry point
# Discretizers
from .discretize.threshold import ThresholdDiscretizer

# Inference modules
from .inference.itself import ITSELF_Engine
from .inference.single_pass import SinglePassInferenceEngine

# Loss Computation Functions
from .losses import mpe_log_likelihood_loss

# Neural models
from .models.nn import MLP

# Probabilistic models (as losses)
from .pm_ssl.nam.made import MADE
from .pm_ssl.pc.spn import SumProductNetwork
from .pm_ssl.pgm.mn import MarkovNetwork

# Registry
from .registry import get as factory

# Trainers
from .training.ssl_trainer import SelfSupervisedTrainer

# Control what 'from neupi import *' imports
__all__ = [
    "factory",
    "MarkovNetwork",
    "SumProductNetwork",
    "MADE",
    "MLP",
    "SelfSupervisedTrainer",
    "SinglePassInferenceEngine",
    "ITSELF_Engine",
    "ThresholdDiscretizer",
    "__version__",
]
