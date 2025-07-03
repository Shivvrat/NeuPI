import torch


class ThresholdDiscretizer(torch.nn.Module):
    """
    Discretizes a tensor of probabilities to binary assignments based on a threshold.

    This class acts as a callable function. An instance can be passed to an
    inference engine and called to perform the discretization.

    Args:
        threshold (float): The threshold value. Values >= threshold will be 1,
                           and values < threshold will be 0. Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prob_outputs: torch.Tensor) -> torch.Tensor:
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
