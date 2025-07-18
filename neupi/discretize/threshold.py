import torch

from neupi.core.discretizer import BaseDiscretizer
from neupi.registry import register


@register("discretizer")
class ThresholdDiscretizer(BaseDiscretizer):
    """
    Discretizes a tensor of probabilities to binary assignments based on a threshold.

    This class acts as a callable function. An instance can be passed to an
    inference engine and called to perform the discretization.

    Args:
        threshold (float): The threshold value. Values >= threshold will be 1,
                           and values < threshold will be 0. Defaults to 0.5.

    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. (2024). Learning to Solve the Constrained Most Probable Explanation Task in Probabilistic Graphical Models. Proceedings of The 27th International Conference on Artificial Intelligence and Statistics. International Conference on Artificial Intelligence and Statistics, PMLR, pp. 2791–2799. https://proceedings.mlr.press/v238/arya24b.html


    Arya, S., Rahman, T., & Gogate, V. (2024). Neural Network Approximators for Marginal MAP in Probabilistic Circuits. Proceedings of the AAAI Conference on Artificial Intelligence, 38(10), 10918–10926. https://doi.org/10.1609/aaai.v38i10.28966

    Arya, S., Rahman, T., & Gogate, V. G. (2024). A neural network approach for efficiently answering most probable explanation queries in probabilistic models. NeurIPS 2024. https://openreview.net/forum?id=ufPPf9ghzP
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")
        self.threshold = threshold

    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor = None,
        evidence_mask: torch.Tensor = None,
        unobs_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Converts a tensor of probabilities to binary assignments.

        Args:
            prob_outputs (torch.Tensor): A tensor of probabilities, typically the
                                         output of a sigmoid function.

        Returns:
            torch.Tensor: A tensor of binary assignments (0s and 1s) with the same
                          shape as the input, on the same device.
        """
        return (prob_outputs >= self.threshold).to(
            dtype=prob_outputs.dtype, device=prob_outputs.device
        )
