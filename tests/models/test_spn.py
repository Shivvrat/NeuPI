from pathlib import Path

import numpy as np
import pytest
import torch
from neupi import SumProductNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def spn_model():
    """Pytest fixture to load the SumProductNetwork once for all tests."""
    json_path = Path(__file__).parent.parent / "networks/spn/nltcs/spn.json"
    return SumProductNetwork(json_file=str(json_path), device=DEVICE)


def get_baseline_ll(assignment: torch.Tensor, spn_json_file: str) -> float:
    """
    Compute the log-likelihood of the given assignment under the SPN defined by the JSON file.

    Args:
        assignment (torch.Tensor): Tensor of shape (1, num_variables) with binary values (0 or 1).
        spn_json_file (str): Path to the SPN JSON file.

    Returns:
        float: The log-likelihood of the assignment.
    """
    from deeprob.spn.algorithms.inference import log_likelihood
    from deeprob.spn.structure.io import load_spn_json

    # Load the SPN from the JSON file
    spn = load_spn_json(spn_json_file)

    x = assignment.detach().clone().cpu().to(torch.float32).numpy()
    log_ll = np.mean(
        log_likelihood(
            spn,
            x,  # shape: (batch_size, num_variables)
            return_results=False,
        )
    )
    return float(log_ll)


def test_spn_evaluation(spn_model):
    """
    Tests the log-likelihood evaluation of the SumProductNetwork module
    against a known, manually calculated value.
    """
    # Test assignment:
    num_variables = spn_model.num_var
    batch_size = 10
    assignment = torch.randint(0, 2, (batch_size, num_variables), device=DEVICE)
    expected_ll = get_baseline_ll(assignment, spn_model.json_file)

    # Evaluate the model
    calculated_ll = spn_model(assignment).mean()

    assert calculated_ll.numel() == 1
    assert torch.isclose(
        calculated_ll, torch.tensor(expected_ll, device=DEVICE), atol=1e-1
    ), f"Expected {expected_ll}, got {calculated_ll}"
