import torch


def entropy_penalty(
    predictions: torch.Tensor,
    query_mask: torch.Tensor,
    alpha: float = 1.0,
    aggregate: str = "avg",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Entropy penalty over the query variables, added to the negative log-likelihood loss.

    For each sample, l_E(q, alpha) = -alpha * sum_{j in Q} [q_j log(q_j) + (1 - q_j) log(1 - q_j)],
    where Q is the set of query variables. With alpha > 0 it pushes the relaxed predictions
    towards 0/1; alpha = 0 gives the plain continuous loss. A negative alpha is an entropy bonus
    that keeps the predictions soft.

    Args:
        predictions (torch.Tensor): Probabilities in [0, 1]. Shape: (batch_size, num_variables).
        query_mask (torch.Tensor): Boolean mask of the query variables. Same shape as `predictions`.
        alpha (float): Weight of the penalty; negative for a bonus.
        aggregate (str): "avg" or "sum" over the batch, as in `mpe_log_likelihood_loss`.
        eps (float): Predictions are clamped to [eps, 1 - eps] to keep the logarithms finite.

    Returns:
        torch.Tensor: A scalar tensor with the penalty.

    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. G. (2024). A neural network approach for efficiently answering most probable explanation queries in probabilistic models. NeurIPS 2024. https://openreview.net/forum?id=ufPPf9ghzP
    """
    q = predictions.float().clamp(eps, 1 - eps)
    plogp = q * torch.log(q) + (1 - q) * torch.log(1 - q)
    per_sample = -alpha * (plogp * query_mask.to(plogp.dtype)).sum(dim=-1)
    if aggregate == "avg":
        return per_sample.mean()
    elif aggregate == "sum":
        return per_sample.sum()
    raise ValueError(f"Invalid aggregate method: {aggregate}")


def mpe_log_likelihood_loss(
    predictions: torch.Tensor,
    pgm_evaluator: torch.nn.Module,
    aggregate: str = "avg",
) -> torch.Tensor:
    """
    Calculates the negative log-likelihood for a batch of MPE predictions.

    This loss function is designed for training neural networks to solve MPE inference.
    The goal is to maximize the log-likelihood of the predicted assignments. Since
    optimizers minimize loss, this function returns the *negative* of the summed
    log-likelihoods.

    Note: MMAP is only supported for probabilistic circuits (`SumProductNetwork`). For MMAP, set the unobserved variables to NaN (see `neupi.utils.pgm_utils.mark_unobserved`) and the evaluator marginalizes them out. `SelfSupervisedTrainer`, `ITSELF_Engine` and the discretizers do this automatically when `unobs_mask` is non-empty.

    Args:
        predictions (torch.Tensor): A batch of binary assignments (0s or 1s) predicted by
                                    a neural network. Shape: (batch_size, num_variables).
        pgm_evaluator (torch.nn.Module): An instantiated PGM evaluator object from `neupi.pgm`
                                         (e.g., ProbabilisticGraphicalModel, SumProductNetwork) that has a
                                         `forward` or `evaluate` method.
        aggregate (str): "avg" or "sum" over the batch.

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
