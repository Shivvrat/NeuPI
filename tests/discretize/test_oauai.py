from pathlib import Path

import pytest
import torch
from neupi import OAUAI, ThresholdDiscretizer

# Import components from your NeuPI library structure
from neupi.training.pm_ssl.pgm.mn import MarkovNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Fixtures ---


@pytest.fixture(scope="module")
def mn_evaluator():
    """Loads a Markov Network evaluator for all tests in this module."""
    uai_path = Path(__file__).parent.parent.parent / "examples/networks/mn/Grids_17.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


@pytest.fixture(scope="module")
def dummy_inference_data(mn_evaluator):
    """Creates a dummy batch of inference data (probabilities and masks)."""
    num_samples = 4
    num_vars = mn_evaluator.num_variables

    prob_outputs = torch.rand(num_samples, num_vars, device=DEVICE, dtype=torch.float32)

    # Define a query mask
    query_mask = torch.zeros(num_samples, num_vars, device=DEVICE, dtype=torch.bool)
    # Make the first 10 variables query variables for each sample
    query_mask[:, :10] = True

    # Other masks are not used by this discretizer but are needed for the interface
    evidence_mask = torch.zeros_like(query_mask)
    unobs_mask = torch.zeros_like(query_mask)

    return prob_outputs, evidence_mask, query_mask, unobs_mask


# --- Test Cases ---


def test_uncertainty_discretizer_produces_valid_output(mn_evaluator, dummy_inference_data):
    """
    Tests that the HighUncertaintyDiscretizer runs and returns a valid binary tensor.
    """
    prob_outputs, ev_mask, q_mask, un_mask = dummy_inference_data

    # 1. Initialize the discretizer
    uncertainty_discretizer = OAUAI(
        pgm_evaluator=mn_evaluator, k=3  # Search over the 3 most uncertain variables
    )

    # 2. Call the discretizer
    discrete_assignments = uncertainty_discretizer(prob_outputs, ev_mask, q_mask, un_mask)

    # 3. Validate the output
    assert discrete_assignments is not None
    assert discrete_assignments.shape == prob_outputs.shape
    assert torch.all(
        (discrete_assignments == 0) | (discrete_assignments == 1)
    ), "Output assignments are not binary"


def test_uncertainty_discretizer_improves_score(mn_evaluator, dummy_inference_data):
    """
    Tests that the HighUncertaintyDiscretizer finds assignments with scores
    better than or equal to simple thresholding.
    """
    prob_outputs, ev_mask, q_mask, un_mask = dummy_inference_data

    # 1. Get baseline score from simple thresholding
    threshold_discretizer = ThresholdDiscretizer(threshold=0.5)
    thresholded_assignments = threshold_discretizer(prob_outputs)
    with torch.no_grad():
        baseline_scores = mn_evaluator(thresholded_assignments)

    # 2. Get scores from the HighUncertaintyDiscretizer
    uncertainty_discretizer = OAUAI(
        pgm_evaluator=mn_evaluator, k=4  # Use a slightly larger k for a better search
    )
    oauai_assignments = uncertainty_discretizer(prob_outputs, ev_mask, q_mask, un_mask)
    with torch.no_grad():
        oauai_scores = mn_evaluator(oauai_assignments)

    # 3. Assert that the search scores are at least as good as the baseline
    print(f"Threshold Avg Score: {baseline_scores.mean().item():.4f}")
    print(f"OAUAI Avg Score: {oauai_scores.mean().item():.4f}")
    assert torch.all(
        oauai_scores >= baseline_scores
    ), "OAUAI produced a worse score than simple thresholding"
