import torch


def apply_evidence(
    predictions: torch.Tensor, evidence_data: torch.Tensor, evidence_mask: torch.Tensor
) -> torch.Tensor:
    """
    Clamps the predictions to the ground truth values where evidence is present.

    This function handles both single instance (1D) and batched (2D) inputs.
    It is a common step in training neural solvers for probabilistic inference,
    where the network should not change the values of known evidence variables.
    The output is always guaranteed to be a 2D tensor.

    Args:
        predictions (torch.Tensor): The output from the neural network.
                                    Shape: (num_variables,) or (batch_size, num_variables).
        evidence_data (torch.Tensor): The tensor with the true evidence values.
                                     Must have the same shape as `predictions`. We only use the evidence data where the evidence mask is True.
        evidence_mask (torch.Tensor): A boolean tensor indicating which variables
                                      are evidence. Must have the same shape as `predictions`.

    Returns:
        torch.Tensor: A new 2D tensor of shape (batch_size, num_variables) where
                      evidence variables have been replaced by their ground truth values.
                      If the input was 1D, the output batch_size is 1.
    """
    # Create a clone to avoid in-place modification of the original predictions tensor
    processed_tensor = predictions.clone()

    # Use the evidence mask to overwrite predicted values with true evidence values.
    # This works for both 1D and 2D tensors.
    processed_tensor[evidence_mask] = evidence_data[evidence_mask]

    # Ensure the output is always a 2D tensor (batched)
    if processed_tensor.ndim == 1:
        return processed_tensor.unsqueeze(0)

    return processed_tensor
