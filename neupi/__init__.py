# neupi/__init__.py
"""NeuPI: Neural-Probabilistic Inference Library"""

# Expose the version of the package
__version__ = "1.3.1"

# Expose the factory function as the main entry point
# Discretizers
from .discretize.combined import BestOfDiscretizer
from .discretize.kn import KNearestDiscretizer
from .discretize.oauai import OAUAI
from .discretize.threshold import ThresholdDiscretizer

# Embeddings
from .embedding.discrete import DiscreteEmbedder
from .embedding.identity import IdentityEmbedding

# Inference modules
from .inference.itself import ITSELF_Engine
from .inference.single_pass import SinglePassInferenceEngine

# Loss Computation Functions
from .losses import entropy_penalty, mpe_log_likelihood_loss

# Neural models
from .models.nn import MLP

# Registry
from .registry import get as factory

# Probabilistic models (as losses)
from .training.pm_ssl.nam.made import MADE
from .training.pm_ssl.pc.spn import SumProductNetwork
from .training.pm_ssl.pgm.mn import MarkovNetwork, ProbabilisticGraphicalModel

# Trainers
from .training.trainers.ssl_trainer import SelfSupervisedTrainer

# Control what 'from neupi import *' imports
__all__ = [
    "factory",
    "ProbabilisticGraphicalModel",
    "MarkovNetwork",
    "SumProductNetwork",
    "MADE",
    "MLP",
    "SelfSupervisedTrainer",
    "SinglePassInferenceEngine",
    "ITSELF_Engine",
    "KNearestDiscretizer",
    "OAUAI",
    "ThresholdDiscretizer",
    "BestOfDiscretizer",
    "DiscreteEmbedder",
    "IdentityEmbedding",
    "__version__",
    "mpe_log_likelihood_loss",
    "entropy_penalty",
]
