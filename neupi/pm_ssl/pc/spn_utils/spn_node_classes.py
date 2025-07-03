import torch
import torch.nn as nn


class ProductNode(nn.Module):
    def __init__(self, node_idx, scope, children_indices):
        super(ProductNode, self).__init__()
        self.node_idx = node_idx
        self.scope = scope
        self.children_indices = children_indices

    def forward(self, function_values):
        """
        Perform the product operation.

        :param child_values: Tensor of values from child nodes.
        :return: Result of the product operation. (sum in log space)
        """
        child_values = function_values[self.children_indices]
        curr_node_value = torch.sum(child_values, dim=0)
        return curr_node_value


class SumNode(nn.Module):
    def __init__(self, node_idx, weights, scope, children_indices, device, eps=1e-6):
        super(SumNode, self).__init__()
        self.node_idx = node_idx
        self.weights = torch.FloatTensor(weights).to(device)  # Make weights a learnable parameter
        self.log_weights = torch.log(self.weights + eps).unsqueeze(1)
        self.scope = scope
        self.children_indices = children_indices
        self.eps = eps

    def forward(self, function_values):
        """
        Perform the sum operation in log space.

        :param child_values: Tensor of values from child nodes.
        :return: Result of the sum operation in log space.
        """
        child_values = function_values[self.children_indices]
        weighted_child_values = child_values + self.log_weights
        curr_node_value = torch.logsumexp(weighted_child_values, dim=0)
        return curr_node_value


class BernoulliLeaf(torch.nn.Module):
    def __init__(self, node_idx, p, scope, device):
        super(BernoulliLeaf, self).__init__()
        self.node_idx = node_idx
        self.weights = torch.tensor([p], dtype=torch.float, device=device)
        self.eps = 1e-6
        self.scope = scope
        self.device = device

        # Precompute operations
        self.precomputed_values = self.precompute()

    def precompute(self):
        # Precompute log weights and related values
        log_weights = torch.log(self.weights + self.eps).to(self.device)
        log_one_minus_weights = torch.log(1 - self.weights + self.eps).to(self.device)

        # Precompute results for NaN cases
        result_for_nan = torch.logsumexp(torch.stack([log_weights, log_one_minus_weights]), dim=0)

        # Precompute results for input cases
        if self.weights > 0.5:
            result_for_query = log_weights
        else:
            result_for_query = log_one_minus_weights

        return {
            "log_weights": log_weights,
            "log_one_minus_weights": log_one_minus_weights,
            "result_for_nan": result_for_nan,
            "result_for_query": result_for_query,
        }

    def forward(self, inp_x):
        """
        Evaluate the Bernoulli leaf node.

        :param inp_x: Tensor representing input data (batch_size, input_size).
        :return: Result of the Bernoulli leaf node.
        """
        # if value is -1, it is a query variable - used when we want to find inputs for NN
        # if value is nan, it is a unobs value

        # This reduces a dimension since we are only interested in one variable for each bernoulli node

        if inp_x.ndimension() == 1:
            # add a first dimension to x - if it is 1d - this is one example and not a batch
            inp_x = inp_x.unsqueeze(0)
        inp_x = inp_x[:, self.scope]

        # Handling NaN values
        unobs_indices = torch.isnan(inp_x)
        if torch.any(unobs_indices):
            x = inp_x.clone()
            x[unobs_indices] = 0  # Temporarily replace NaNs for computation
        else:
            x = inp_x
        if x.ndimension() == 1:
            # Log probabilities calculation using precomputed values
            log_prob_one_minus_x = torch.log(1 - x + self.eps)
            log_prob_x = torch.log(x + self.eps)
        elif x.ndimension() == 2:
            log_prob_one_minus_x = torch.log(x[:, 0] + self.eps)
            log_prob_x = torch.log(x[:, 1] + self.eps)
            unobs_indices = unobs_indices[:, 0]
        result = torch.logsumexp(
            torch.stack(
                [
                    log_prob_x + self.precomputed_values["log_weights"],
                    log_prob_one_minus_x + self.precomputed_values["log_one_minus_weights"],
                ]
            ),
            dim=0,
        )

        # Replace results for NaN values
        result_in_log_space = torch.where(
            unobs_indices, self.precomputed_values["result_for_nan"], result
        )
        # query_result is equal to log_weight if self.weight > 0.5 else log_one_minus_weight
        query_results = self.precomputed_values["result_for_query"]

        # query vars are made -1
        if len(x.shape) == 1:
            result_in_log_space = torch.where(x != -1, result_in_log_space, query_results)
        elif len(x.shape) == 2:
            result_in_log_space = torch.where(x[:, 0] != -1, result_in_log_space, query_results)
        return result_in_log_space
