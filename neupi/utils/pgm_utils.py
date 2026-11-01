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


def mark_unobserved(assignments: torch.Tensor, unobs_mask: torch.Tensor) -> torch.Tensor:
    """
    Sets unobserved variables to NaN, which evaluators that support marginalization
    (probabilistic circuits such as `SumProductNetwork`) sum out.

    Gradients do not flow to the replaced positions.

    Args:
        assignments (torch.Tensor): Shape (batch_size, num_variables).
        unobs_mask (torch.Tensor): Boolean mask of the unobserved variables. Same shape.

    Returns:
        torch.Tensor: A new tensor with NaN at the unobserved positions.
    """
    return torch.where(unobs_mask.bool(), torch.full_like(assignments, float("nan")), assignments)


def check_marginalization(pgm_evaluator: torch.nn.Module) -> None:
    """
    Raises a ValueError unless the evaluator can marginalize unobserved variables.

    MMAP (a non-empty unobs_mask) is only supported for probabilistic circuits.
    """
    if not getattr(pgm_evaluator, "supports_marginalization", False):
        raise ValueError(
            f"{type(pgm_evaluator).__name__} cannot marginalize unobserved variables. "
            "MMAP (a non-empty unobs_mask) is only supported for probabilistic circuits "
            "(SumProductNetwork)."
        )


def evaluate_log_likelihood(
    pgm_evaluator: torch.nn.Module, assignments: torch.Tensor, unobs_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Scores assignments with the evaluator, marginalizing out the unobserved variables.

    With no unobserved variables this is the MPE score `pgm_evaluator(assignments)`. With
    unobserved variables it is the MMAP score log p(e, q), which only evaluators with
    `supports_marginalization = True` (probabilistic circuits) can compute.

    Raises:
        ValueError: If there are unobserved variables and the evaluator cannot marginalize them.
    """
    if unobs_mask is None or not unobs_mask.any():
        return pgm_evaluator(assignments)
    check_marginalization(pgm_evaluator)
    return pgm_evaluator(mark_unobserved(assignments, unobs_mask))
