import torch
from torch import nn

from neupi.core.model import BaseProbModel
from neupi.registry import register
from ..io.uai_reader_cython import UAIParser


@register("prob_model")
class MarkovNetwork(BaseProbModel):
    """
    A PyTorch module to evaluate the log-likelihood of assignments in a binary Markov Network.

    This class loads a Markov Network from a .uai file and provides an efficient,
    vectorized method to compute the log-likelihood for a batch of variable assignments.
    It supports both pairwise and higher-order factor models.

    Args:
        uai_file (str): Path to the .uai file defining the Markov Network.
        device (str or torch.device): The device to perform computations on ('cpu' or 'cuda').

    Note: For now, models from the UAI format are supported. Other formats can be supported in by rewriting the UAIParser class.

    Example:
    ```
    from neupi.pm.pgm.mn import MarkovNetwork
    mn = MarkovNetwork("path/to/mn.uai")
    ```
    """

    def __init__(self, uai_file: str, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        if not uai_file.endswith(".uai"):
            raise ValueError("Only .uai format is supported for Markov Networks.")
        self.uai_file = uai_file

        with open(uai_file, "r") as f:
            file_content = f.read()

        self.pgm = UAIParser(model_str=file_content, one_d_factors=0, device=self.device)
        self.num_variables = self.pgm.num_vars

        if self.pgm.network_type != "MARKOV":
            raise ValueError("The .uai file must define a MARKOV network.")

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

    def _evaluate_higher_order(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates log-likelihood for models with higher-order factors.

        Args:
            x (torch.Tensor): A binary tensor of assignments of shape (batch_size, num_variables).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,).
        """
        x = x.to(self.device)
        ll_scores = torch.zeros(x.shape[0], device=self.device)

        if x.shape[1] != self.pgm.num_vars:
            raise ValueError("Input dimension does not match the number of variables in the model.")

        for size, data in self.precomputed_data.items():
            clique_scores = self._compute_clique_scores(
                x, data["binary_combinations"], data["all_vars"], data["all_factors"]
            )
            ll_scores += torch.sum(clique_scores, dim=1)
        return ll_scores

    def _evaluate_pairwise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates log-likelihood for models with only pairwise and unary factors.

        Args:
            x (torch.Tensor): A binary tensor of assignments of shape (batch_size, num_variables).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,).
        """
        x = x.to(self.device)

        # Unary factor contributions
        univariate_contrib = (1 - x[:, self.pgm.univariate_vars]) * self.pgm.univariate_tables[
            :, 0
        ] + x[:, self.pgm.univariate_vars] * self.pgm.univariate_tables[:, 1]

        # Pairwise factor contributions
        x_biv_0 = x[:, self.pgm.bivariate_vars[:, 0]]
        x_biv_1 = x[:, self.pgm.bivariate_vars[:, 1]]

        bivariate_contrib = (
            (1 - x_biv_0) * (1 - x_biv_1) * self.pgm.bivariate_tables[:, 0, 0].unsqueeze(0)
            + (1 - x_biv_0) * x_biv_1 * self.pgm.bivariate_tables[:, 0, 1].unsqueeze(0)
            + x_biv_0 * (1 - x_biv_1) * self.pgm.bivariate_tables[:, 1, 0].unsqueeze(0)
            + x_biv_0 * x_biv_1 * self.pgm.bivariate_tables[:, 1, 1].unsqueeze(0)
        )

        total_log_likelihood = torch.sum(univariate_contrib, dim=1) + torch.sum(
            bivariate_contrib, dim=1
        )
        return total_log_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for the evaluate method.

        Args:
            x (torch.Tensor): A binary tensor of assignments of shape (batch_size, num_variables).

        Returns:
            torch.Tensor: A tensor of log-likelihood scores of shape (batch_size,).
        """
        return self.evaluate(x)
