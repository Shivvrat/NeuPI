import torch

from .spn_node_classes import BernoulliLeaf, ProductNode, SumNode


def preprocess_links(data):
    num_nodes = len(data["nodes"])
    num_links = len(data["links"])

    # Tensors to store edges and their start indices
    edge_parent = torch.empty(num_links, dtype=torch.long)
    edge_child_start_index = torch.zeros(num_nodes + 1, dtype=torch.long)

    for idx, link in enumerate(data["links"]):
        link_idx = idx
        target = link["target"]
        edge_parent[link_idx] = target
        edge_child_start_index[target + 1] += 1

    # Convert to cumulative start indices
    edge_child_start_index = torch.cumsum(edge_child_start_index, 0)

    return edge_parent, edge_child_start_index


def get_distributions(data, spn_class):
    all_distributions = {}
    # Initialize nodes based on their type (Bernoulli, Sum, Product)
    # We take reversed order because we want to evaluate the nodes from leaves to root
    for idx, each_node_dict in enumerate(data["nodes"]):
        node_class = each_node_dict["class"]
        spn_class.node_types[node_class].append(each_node_dict["id"])
        scope = torch.LongTensor([int(s) for s in each_node_dict.get("scope", [])])
        if node_class == "Bernoulli":
            params = each_node_dict["params"]["p"]
            all_distributions[idx] = BernoulliLeaf(idx, params, scope[0], spn_class.device)
            spn_class.num_var += 1
        elif node_class == "Product":
            children_indices = spn_class._get_children_indices(idx)
            all_distributions[idx] = ProductNode(idx, scope, children_indices)
        elif node_class in ["Sum"]:
            children_indices = spn_class._get_children_indices(idx)
            all_distributions[idx] = SumNode(
                idx,
                each_node_dict["weights"],
                scope,
                children_indices,
                spn_class.device,
                spn_class.eps,
            )
        else:
            raise NotImplementedError(f"Unknown node class {node_class}")
    return all_distributions
