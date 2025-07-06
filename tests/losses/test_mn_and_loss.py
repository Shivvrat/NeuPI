import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from neupi import MarkovNetwork, mpe_log_likelihood_loss

# Define the device for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def pairwise_mn():
    """Pytest fixture to load the pairwise Markov Network once for all tests."""
    # Path to the test data file, relative to this test file
    uai_path = Path(__file__).parent.parent.parent / "examples/networks/mn/Segmentation_12.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


def get_baseline_ll(data, mn_path):
    """
    Get the baseline log-likelihood for a given assignment.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as new_file:
        for row in data:
            new_file.write(",".join(map(str, map(int, row))) + "\n")
        data_path = new_file.name

    command = ["libra", "mscore", "-m", mn_path, "-i", data_path]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(f"Output: {output}")

    match = re.search(r"avg\s*=\s*(-?\d+\.\d+)", output)
    if not match:
        raise ValueError(f"Could not extract log-likelihood from output: {output}")
    ll_score_log_e = float(match.group(1))
    return ll_score_log_e


def test_mn_evaluation(pairwise_mn):
    """
    Tests the log-likelihood evaluation of the MarkovNetwork module
    against a known, pre-calculated value.
    """
    num_variables = pairwise_mn.num_variables
    # generate random assignment with (10, num_variables)
    assignment = torch.randint(0, 2, (10, num_variables), device=DEVICE)

    # Pre-calculated log-likelihood for this assignment:
    expected_ll = get_baseline_ll(assignment, pairwise_mn.uai_file)

    # Evaluate the model
    calculated_ll = pairwise_mn(assignment).mean()

    assert calculated_ll.numel() == 1
    assert torch.isclose(
        calculated_ll, torch.tensor(expected_ll, device=DEVICE), atol=1e-4
    ), f"Expected {expected_ll}, got {calculated_ll}"


def test_mpe_loss_function(pairwise_mn):
    """
    Tests the mpe_log_likelihood_loss function using the MN evaluator.
    """
    # Batch of two assignments
    num_variables = pairwise_mn.num_variables
    assignments = torch.randint(0, 2, (10, num_variables), device=DEVICE)

    # Expected LLs:
    expected_ll = get_baseline_ll(assignments, pairwise_mn.uai_file)

    # Calculate loss using the library function
    loss = mpe_log_likelihood_loss(assignments, pairwise_mn)

    assert loss.numel() == 1
    assert torch.isclose(
        -loss, torch.tensor(expected_ll, device=DEVICE), atol=1e-4
    ), f"Expected {expected_ll}, got {loss}"
