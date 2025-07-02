import torch


def mpe_log_likelihood_loss(
    predictions: torch.Tensor, pgm_evaluator: torch.nn.Module, aggregate: str = "avg"
) -> torch.Tensor:
    """
    Calculates the negative log-likelihood for a batch of MPE predictions.

    This loss function is designed for training neural networks to solve MPE inference.
    The goal is to maximize the log-likelihood of the predicted assignments. Since
    optimizers minimize loss, this function returns the *negative* of the summed
    log-likelihoods.

    Args:
        predictions (torch.Tensor): A batch of binary assignments (0s or 1s) predicted by
                                    a neural network. Shape: (batch_size, num_variables).
        pgm_evaluator (torch.nn.Module): An instantiated PGM evaluator object from `neupi.pgm`
                                         (e.g., MarkovNetwork, SumProductNetwork) that has a
                                         `forward` or `evaluate` method.

    Returns:
        torch.Tensor: A scalar tensor representing the total negative log-likelihood,
                      to be used for backpropagation.
    """
    # Ensure predictions are float for the evaluator
    float_predictions = predictions.float()

    # Get the log-likelihood for each prediction in the batch
    log_likelihoods = pgm_evaluator(float_predictions)

    # We want to MAXIMIZE the log-likelihood, so we MINIMIZE the negative log-likelihood.
    # We sum/average the likelihoods across the batch to get a single loss value.
    if aggregate == "avg":
        loss = -torch.mean(log_likelihoods)
    elif aggregate == "sum":
        loss = -torch.sum(log_likelihoods)
    else:
        raise ValueError(f"Invalid aggregate method: {aggregate}")

    return loss
