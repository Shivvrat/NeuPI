import json

import torch
from torch import nn

from .spn_node_classes import BernoulliLeaf, ProductNode, SumNode
from .spn_utils import get_distributions, preprocess_links


class SumProductNetwork(nn.Module):
    def __init__(self, json_file, device="cpu"):
        """
        Initialize the SumProductNetwork with configuration from a JSON file.

        :param json_file: Path to a JSON file containing the model configuration.
        :param num_var: Number of variables in the model.
        :param device: Device to perform computations ('cpu' or 'cuda:0').
        :param depth_features: Depth of the features to be used for the input features of NN (defaults to 1).
        :param approx: Boolean flag for approximation method usage (defaults to False).
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
        edge_parent, edge_child_start_index = preprocess_links(data)
        self.edge_parent = edge_parent.to(device)
        self.edge_child_start_index = edge_child_start_index.to(device)

        distributions = get_distributions(data, self)
        self.all_distributions = nn.ModuleList(list(distributions.values()))
        # Convert node types to tensors
        for each_node_type in self.node_types:
            self.node_types[each_node_type] = torch.tensor(
                self.node_types[each_node_type], dtype=torch.long, device=self.device
            )

    def _get_children_indices(self, node_idx: int):
        start_idx = self.edge_child_start_index[node_idx]
        end_idx = self.edge_child_start_index[node_idx + 1]
        return self.all_nodes[start_idx + 1 : end_idx + 1]

    def _check_is_binary(self, tensor):
        # Check if all elements are either 0 or 1
        return ((tensor == 0) | (tensor == 1)).all()

    def evaluate(self, x):
        """
        Evaluate the SPN model on input data.

        :param x: Tensor representing input data (batch_size, input_size).
        :return: Function value at the root of the SPN.
        """
        assert self.num_var == x.size(1), "Input size must match the number of variables"
        # the shape of function_values_at_each_index is (num_var, batch_size)
        function_values_at_each_index = torch.empty(
            (self.num_nodes_in_spn, x.size(0)), device=self.device
        )
        # Make sure there are no -1s in the input
        assert torch.sum(x == -1) == 0, "Input cannot contain -1s"
        # Evaluate each node in reverse order (from leaves to root)
        for node_idx in reversed(range(self.num_nodes_in_spn)):
            func_value = self._evaluate_node(
                node_idx,
                x,
                function_values_at_each_index,
            )
            function_values_at_each_index[node_idx] = func_value
        return function_values_at_each_index[0]

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
        return self.evaluate(x)
