import json

import torch
from neupi.core.model import BaseProbModel
from neupi.registry import register
from torch import nn

from .spn_utils import (
    BernoulliLeaf,
    ProductNode,
    SumNode,
    get_distributions,
    preprocess_links,
)


@register("prob_model")
class SumProductNetwork(BaseProbModel):
    # NaN inputs are summed out, so the SPN also scores MMAP solutions
    supports_marginalization = True

    def __init__(self, json_file, device="cpu"):
        """
        Initialize the SumProductNetwork with configuration from a JSON file.

        :param json_file: Path to a JSON file containing the model configuration.
        :param num_var: Number of variables in the model.
        :param device: Device to perform computations ('cpu' or 'cuda:0').
        :param depth_features: Depth of the features to be used for the input features of NN (defaults to 1).
        :param approx: Boolean flag for approximation method usage (defaults to False).

        Note: For now, models trained with DeeProb-kit are supported. (https://github.com/deeprob-org/deeprob-kit). Other libraries can be supported in by rewriting the get_distributions and preprocess_links functions.

        Example:
        ```
        from neupi.pm.pc.spn import SumProductNetwork
        spn = SumProductNetwork("path/to/spn.json")
        ```
        """
        super(SumProductNetwork, self).__init__()
        with open(json_file) as f:
            data = json.load(f)

        self.json_file = json_file
        self.num_var = 0
        self.num_nodes_in_spn: int = len(data["nodes"])
        self.all_nodes = torch.arange(self.num_nodes_in_spn, dtype=torch.long)
        self.eps = 1e-6
        self.device = device
        self.node_types = {"Sum": [], "Product": [], "Bernoulli": []}
        self.node_children, self.evaluation_order, self.root = preprocess_links(data)

        distributions = get_distributions(data, self)
        self.all_distributions = nn.ModuleList(list(distributions.values()))
        # Convert node types to tensors
        for each_node_type in self.node_types:
            self.node_types[each_node_type] = torch.tensor(
                self.node_types[each_node_type], dtype=torch.long, device=self.device
            )

    @property
    def num_variables(self) -> int:
        """Number of variables, named as in `ProbabilisticGraphicalModel`."""
        return self.num_var

    def _get_children_indices(self, node_idx: int):
        return self.node_children[node_idx]

    def _check_is_binary(self, tensor):
        # Check if all elements are either 0 or 1
        return ((tensor == 0) | (tensor == 1)).all()

    def evaluate(self, x):
        """
        Evaluate the SPN model on input data.

        :param x: Assignments in [0, 1] of shape (batch_size, num_var) or (num_var,);
            NaN marks a variable to sum out.
        :return: Function value at the root of the SPN.
        """
        x = torch.as_tensor(x, device=self.device)
        # At least float32: log-likelihoods are large, so half precision would lose whole nats
        x = x.to(torch.promote_types(x.dtype, torch.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.size(1) != self.num_var:
            raise ValueError(
                f"Expected assignments of shape (batch_size, {self.num_var}), got {tuple(x.shape)}."
            )
        # the shape of function_values_at_each_index is (num_nodes, batch_size)
        function_values_at_each_index = torch.empty(
            (self.num_nodes_in_spn, x.size(0)), dtype=x.dtype, device=self.device
        )
        # -1 is not a marginalization marker; use NaN for unobserved variables
        if (x == -1).any():
            raise ValueError("Input cannot contain -1s; mark unobserved variables with NaN.")
        # Evaluate every child before its parents
        for node_idx in self.evaluation_order:
            func_value = self._evaluate_node(
                node_idx,
                x,
                function_values_at_each_index,
            )
            function_values_at_each_index[node_idx] = func_value
        return function_values_at_each_index[self.root]

    def _get_node_type_and_children_indices(self, node_idx: int):
        """
        Get the type and children indices of a node.

        :param node_idx: Index of the node.
        :return: Tuple of node type and tensor of children indices.
        """
        children_indices = self._get_children_indices(node_idx)

        return children_indices

    def _evaluate_node(self, node_idx: int, x, function_values_at_each_index):
        """
        Evaluate a single node in the SPN.

        :param node_type: Type of the node (Sum, Product, Bernoulli).
        :param children_indices: Indices of the children nodes.
        :param node_idx: Index of the current node.
        :param x: Input data tensor.
        :param function_values_at_each_index: Tensor holding values for each node.
        :return: Evaluated value of the node.
        """
        node = self.all_distributions[node_idx]
        if isinstance(node, ProductNode):
            return node(function_values_at_each_index)
        elif isinstance(node, SumNode):
            return node(function_values_at_each_index)
        elif isinstance(node, BernoulliLeaf):
            return node(x)
        else:
            raise NotImplementedError("Unknown node type")

    def forward(self, x):
        """Log-likelihood of each assignment, shape (batch_size,), on the device of x."""
        input_device = x.device if isinstance(x, torch.Tensor) else self.device
        return self.evaluate(x).to(input_device)
