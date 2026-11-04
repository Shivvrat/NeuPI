import torch

from .spn_node_classes import BernoulliLeaf, ProductNode, SumNode


def preprocess_links(data):
    """
    Reads the children of every node from the links of a DeeProb-kit JSON graph.

    In the JSON, a link goes from a child (``source``) to its parent (``target``), and
    ``idx`` is the child's position among its parent's children (the order of the sum
    weights). Nodes may be listed in any order and children may be shared (a DAG).

    Returns:
        children (list of LongTensor): The children of each node, as node positions.
        order (list of int): The node positions with every child before its parents.
        root (int): The position of the root node.
    """
    nodes = data["nodes"]
    num_nodes = len(nodes)
    position = {node.get("id", i): i for i, node in enumerate(nodes)}
    if len(position) != num_nodes:
        raise ValueError("SPN node ids must be unique.")

    child_lists = [[] for _ in range(num_nodes)]
    has_parent = [False] * num_nodes
    for link_order, link in enumerate(data["links"]):
        try:
            parent, child = position[link["target"]], position[link["source"]]
        except KeyError as e:
            raise ValueError(f"SPN link {link} refers to an unknown node id {e}.") from None
        child_lists[parent].append((link.get("idx", link_order), link_order, child))
        has_parent[child] = True

    children = [
        torch.tensor([c for _, _, c in sorted(cl)], dtype=torch.long) for cl in child_lists
    ]

    roots = [i for i in range(num_nodes) if not has_parent[i]]
    if len(roots) != 1:
        raise ValueError(f"An SPN must have exactly one root node, found {len(roots)}.")
    root = roots[0]

    # Post-order traversal from the root (iterative, so deep SPNs do not hit the recursion limit)
    order, state = [], [0] * num_nodes  # 0 = unvisited, 1 = on the stack, 2 = done
    stack = [(root, iter(children[root].tolist()))]
    state[root] = 1
    while stack:
        node, it = stack[-1]
        child = next(it, None)
        if child is None:
            stack.pop()
            state[node] = 2
            order.append(node)
        elif state[child] == 1:
            raise ValueError("The SPN graph has a cycle.")
        elif state[child] == 0:
            state[child] = 1
            stack.append((child, iter(children[child].tolist())))
    if len(order) != num_nodes:
        raise ValueError(f"{num_nodes - len(order)} SPN node(s) are not reachable from the root.")
    return children, order, root


def get_distributions(data, spn_class):
    all_distributions = {}
    # Initialize nodes based on their type (Bernoulli, Sum, Product)
    # We take reversed order because we want to evaluate the nodes from leaves to root
    all_vars_in_spn = set()
    for idx, each_node_dict in enumerate(data["nodes"]):
        node_class = each_node_dict["class"]
        if node_class not in spn_class.node_types:
            raise NotImplementedError(
                f"Unknown node class {node_class}; only Sum, Product and Bernoulli nodes are supported."
            )
        spn_class.node_types[node_class].append(idx)
        scope = [int(s) for s in each_node_dict.get("scope", [])]
        all_vars_in_spn.update(scope)
        scope = torch.LongTensor(scope)
        if node_class == "Bernoulli":
            if len(scope) != 1:
                raise ValueError(f"Bernoulli node {idx} must have a scope of one variable.")
            params = each_node_dict["params"]["p"]
            all_distributions[idx] = BernoulliLeaf(
                idx, params, scope[0], spn_class.device
            )
        elif node_class == "Product":
            children_indices = spn_class._get_children_indices(idx)
            all_distributions[idx] = ProductNode(idx, scope, children_indices)
        elif node_class in ["Sum"]:
            children_indices = spn_class._get_children_indices(idx)
            if len(each_node_dict["weights"]) != len(children_indices):
                raise ValueError(
                    f"Sum node {idx} has {len(each_node_dict['weights'])} weights but "
                    f"{len(children_indices)} children."
                )
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
    # Variables are indexed from 0, so a variable that no leaf covers still gets a column
    spn_class.num_var = max(all_vars_in_spn) + 1 if all_vars_in_spn else 0
    return all_distributions
