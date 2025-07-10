from pathlib import Path

import pytest
import torch

# Import components
from neupi import (
    DiscreteEmbedder,
    KNearestDiscretizer,
    MarkovNetwork,
    SinglePassInferenceEngine,
    ThresholdDiscretizer,
)
from neupi.models.nn import MLP
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def mn_evaluator():
    """Loads a Markov Network evaluator."""
    uai_path = Path(__file__).parent.parent.parent / "examples/networks/mn/Grids_17.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


@pytest.fixture(scope="module")
def random_model(mn_evaluator):
    """Provides a simple, untrained model."""
    num_vars = mn_evaluator.num_variables
    embedding = DiscreteEmbedder(num_vars)
    return MLP(hidden_sizes=[16], output_size=num_vars, embedding=embedding).to(DEVICE)


@pytest.fixture(scope="module")
def dummy_dataloader(mn_evaluator):
    """Creates a dummy DataLoader for training and inference."""
    num_samples = 8
    num_vars = mn_evaluator.num_variables
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), device=DEVICE, dtype=torch.float32)
    evidence_mask = torch.rand(num_samples, num_vars, device=DEVICE) > 0.5
    query_mask = ~evidence_mask
    unobs_mask = torch.zeros_like(evidence_mask, dtype=torch.bool)
    dataset = TensorDataset(evidence_data, evidence_mask, query_mask, unobs_mask)
    return DataLoader(dataset, batch_size=4)


def test_knn_discretizer_integration(random_model, mn_evaluator, dummy_dataloader):
    """
    Tests that the KNearestDiscretizer can be used within an inference engine.
    """
    num_vars = mn_evaluator.num_variables

    # 1. Initialize the KNN Discretizer
    # It requires the PGM evaluator to act as its scoring function.
    knn_discretizer = KNearestDiscretizer(pgm_evaluator=mn_evaluator, k=5)

    # 2. Initialize and run the inference engine with the new discretizer
    # The engine now needs to be aware of the query_mask
    knn_engine = SinglePassInferenceEngine(
        model=random_model, discretizer=knn_discretizer, device=DEVICE
    )
    knn_results = knn_engine.run(dummy_dataloader)
    with torch.no_grad():
        knn_ll = mn_evaluator(knn_results["final_assignments"]).mean()
    assert "final_assignments" in knn_results
    num_samples = len(dummy_dataloader.dataset)
    num_vars = mn_evaluator.num_variables
    assert knn_results["final_assignments"].shape == (num_samples, num_vars)
    # assert that the final assignments are binary
    assert torch.all(
        (knn_results["final_assignments"] == 0) | (knn_results["final_assignments"] == 1)
    ), "Final assignments are not binary"

    # use the threshold discretizer and check that the results are the same
    threshold_discretizer = ThresholdDiscretizer(threshold=0.5)
    threshold_engine = SinglePassInferenceEngine(
        model=random_model, discretizer=threshold_discretizer, device=DEVICE
    )
    threshold_results = threshold_engine.run(dummy_dataloader)
    assert "final_assignments" in threshold_results
    with torch.no_grad():
        threshold_ll = mn_evaluator(threshold_results["final_assignments"]).mean()
    # make sure that the log likelihood is better for the knn discretizer
    print(f"KNN Avg Log-Likelihood: {knn_ll.item():.4f}")
    print(f"Threshold Avg Log-Likelihood: {threshold_ll.item():.4f}")
    assert (
        knn_ll >= threshold_ll
    ), "KNN threshold discretizer failed to improve the average log-likelihood over the threshold discretizer ."
