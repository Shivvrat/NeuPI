import torch
from torch import nn

from neupi.core.model import BaseProbModel
from neupi.registry import register
from ..io.uai_reader_cython import UAIParser


@register("prob_model")
class ProbabilisticGraphicalModel(BaseProbModel):
    """
    A PyTorch module to evaluate the log-likelihood of assignments in a binary probabilistic
    graphical model: a Markov network or a Bayesian network in UAI format.

    This class loads the model from a .uai file and provides an efficient, vectorized
    method to compute the log-likelihood for a batch of variable assignments. It supports
    both pairwise and higher-order factors, with or without unary factors. All variables
    must be binary.

    The network type is detected from the file's header (``MARKOV`` or ``BAYES``) and is
    available as ``network_type``; there is nothing to specify. A Bayesian network's joint
    distribution is the product of its CPTs, so both types are evaluated the same way.

    The log-likelihood is unnormalized (the log partition function is not subtracted).
    Zero table entries are replaced by 1e-10 so that relaxed assignments in [0, 1] have
    finite scores and gradients; an assignment that violates a zero entry gets a large
    penalty (log 1e-10 per violated entry) rather than -inf.

    Args:
        uai_file (str): Path to the .uai file (a MARKOV or BAYES network).
        device (str or torch.device): The device to perform computations on ('cpu' or 'cuda').

    Note: For now, models from the UAI format are supported. Other formats can be supported in by rewriting the UAIParser class.

    Example:
    ```
    from neupi import ProbabilisticGraphicalModel
    pgm = ProbabilisticGraphicalModel("path/to/model.uai")
    pgm.network_type  # "MARKOV" or "BAYES", read from the file
    ```
    """

    def __init__(self, uai_file: str, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        uai_file = str(uai_file)
        if not uai_file.lower().endswith(".uai"):
            raise ValueError("Only the .uai format is supported for probabilistic graphical models.")
        self.uai_file = uai_file

        with open(uai_file, "r") as f:
            file_content = f.read()

        # A BAYES file has the same layout, and the product of its CPTs is the joint
        # distribution, so the parser (which only accepts MARKOV) reads it as a Markov network
        stripped = file_content.lstrip()
        header = stripped.split(None, 1)[0].upper() if stripped else ""
        if header not in ("MARKOV", "BAYES"):
            raise ValueError(f"The .uai file must define a MARKOV or BAYES network, found {header!r}.")
        self.network_type = header
        if header == "BAYES":
            file_content = "MARKOV" + stripped[len(header) :]

        self.pgm = UAIParser(model_str=file_content, one_d_factors=0, device=self.device)
        self.num_variables = self.pgm.num_vars

        non_binary = [i for i, d in enumerate(self.pgm.domain_sizes.tolist()) if d != 2]
        if non_binary:
            raise ValueError(
                f"Only binary variables are supported, but {len(non_binary)} variable(s) have a "
                f"domain size other than 2 (first: variable {non_binary[0]} with "
                f"{int(self.pgm.domain_sizes[non_binary[0]])} values)."
            )

        # Select the appropriate evaluation method based on factor complexity
        if self.pgm.pairwise_only:
            self.evaluate = self._evaluate_pairwise
        else:
            self._precompute_for_higher_order()
            self.evaluate = self._evaluate_higher_order

    def _precompute_for_higher_order(self):
        """Precomputes binary combinations for efficient higher-order evaluation."""
        self.precomputed_data = {}
        for size, clique_class in self.pgm.clique_dict_class.items():
            binary_combinations = torch.tensor(
                [[(j >> k) & 1 for k in range(size - 1, -1, -1)] for j in range(2**size)],
                dtype=torch.float32,
                device=self.device,
            )
            self.precomputed_data[size] = {
                "binary_combinations": binary_combinations,
                "all_vars": clique_class.variables,
                "all_factors": clique_class.tables,
            }

    def _compute_clique_scores(self, x, binary_combinations, all_vars, all_factors):
        """Computes scores for a batch of assignments for a given clique size."""
        # Select the values of variables involved in the cliques
        all_values = x[:, all_vars.flatten()].view(x.shape[0], all_vars.shape[0], all_vars.shape[1])

        # Match assignments with binary combinations to find the right factor entry
        selected_values = all_values.unsqueeze(1) * binary_combinations.unsqueeze(0).unsqueeze(
            2
        ) + (1 - all_values.unsqueeze(1)) * (1 - binary_combinations.unsqueeze(0).unsqueeze(2))

        product_term = torch.prod(selected_values, dim=3)

        all_factors_flat = all_factors.view(all_factors.shape[0], -1)

        # Sum the log-potentials from the correct factor entries
        scores = torch.sum(product_term * all_factors_flat.permute(1, 0).unsqueeze(0), dim=1)
        return scores

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Moves x to the device as a float (batch_size, num_variables) tensor."""
        x = torch.as_tensor(x, device=self.device)
        if not torch.is_floating_point(x):
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.num_variables:
            raise ValueError(
                f"Expected assignments of shape (batch_size, {self.num_variables}), got {tuple(x.shape)}."
            )
        return x

    def _evaluate_higher_order(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates log-likelihood for models with higher-order factors.

        Args:
            x (torch.Tensor): Assignments in [0, 1] of shape (batch_size, num_variables)
                or (num_variables,).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,).
        """
        x = self._prepare_input(x)
        ll_scores = torch.zeros(x.shape[0], dtype=x.dtype, device=self.device)

        for size, data in self.precomputed_data.items():
            clique_scores = self._compute_clique_scores(
                x, data["binary_combinations"], data["all_vars"], data["all_factors"]
            )
            ll_scores += torch.sum(clique_scores, dim=1)
        return ll_scores

    def _evaluate_pairwise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates log-likelihood for models with only pairwise and unary factors.
        Either kind of factor may be absent.

        Args:
            x (torch.Tensor): Assignments in [0, 1] of shape (batch_size, num_variables)
                or (num_variables,).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,).
        """
        x = self._prepare_input(x)
        total_log_likelihood = torch.zeros(x.shape[0], dtype=x.dtype, device=self.device)

        # Unary factor contributions (the parser leaves the tables as None when a model has none)
        if self.pgm.univariate_tables is not None:
            x_uni = x[:, self.pgm.univariate_vars]
            univariate_contrib = (1 - x_uni) * self.pgm.univariate_tables[
                :, 0
            ] + x_uni * self.pgm.univariate_tables[:, 1]
            total_log_likelihood = total_log_likelihood + torch.sum(univariate_contrib, dim=1)

        # Pairwise factor contributions
        if self.pgm.bivariate_tables is not None:
            x_biv_0 = x[:, self.pgm.bivariate_vars[:, 0]]
            x_biv_1 = x[:, self.pgm.bivariate_vars[:, 1]]

            bivariate_contrib = (
                (1 - x_biv_0) * (1 - x_biv_1) * self.pgm.bivariate_tables[:, 0, 0].unsqueeze(0)
                + (1 - x_biv_0) * x_biv_1 * self.pgm.bivariate_tables[:, 0, 1].unsqueeze(0)
                + x_biv_0 * (1 - x_biv_1) * self.pgm.bivariate_tables[:, 1, 0].unsqueeze(0)
                + x_biv_0 * x_biv_1 * self.pgm.bivariate_tables[:, 1, 1].unsqueeze(0)
            )
            total_log_likelihood = total_log_likelihood + torch.sum(bivariate_contrib, dim=1)

        return total_log_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for the evaluate method.

        Args:
            x (torch.Tensor): Assignments in [0, 1] of shape (batch_size, num_variables)
                or (num_variables,).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,), on the
                device of x.
        """
        input_device = x.device if isinstance(x, torch.Tensor) else self.device
        return self.evaluate(x).to(input_device)


@register("prob_model")
class MarkovNetwork(ProbabilisticGraphicalModel):
    """
    The original name of `ProbabilisticGraphicalModel`, kept so that existing
    code keeps working. It is the same class: it also reads Bayesian networks.
    """
