from pathlib import Path

import pytest
import torch
from neupi.discretize.threshold import ThresholdDiscretizer
from neupi.inference.itself import ITSELF_Engine  # Used for comparison
from neupi.inference.single_pass import SinglePassInferenceEngine  # The class to test
from neupi.losses import mpe_log_likelihood_loss
from neupi.models.nn import MLP

# Import components from your NeuPI library structure
# Using 'pm_ssl' and 'models.nn' as per your provided file structure
from neupi.pm_ssl import MarkovNetwork
from neupi.training.ssl_trainer import SelfSupervisedTrainer
from torch.utils.data import DataLoader, TensorDataset

# Define the device for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Fixtures ---


@pytest.fixture(scope="module")
def mn_evaluator():
    """Loads a Markov Network evaluator for all tests in this module."""
    uai_path = Path(__file__).parent.parent / "networks/mn/Grids_17.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


@pytest.fixture(scope="module")
def dummy_dataloader(mn_evaluator):
    """Creates a dummy DataLoader for training and inference."""
    num_samples = 8
    num_vars = mn_evaluator.num_variables
    inputs = torch.rand(num_samples, num_vars, device=DEVICE)
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), device=DEVICE, dtype=torch.float32)
    evidence_mask = torch.rand(num_samples, num_vars, device=DEVICE) > 0.5
    dataset = TensorDataset(inputs, evidence_data, evidence_mask)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture(scope="module")
def random_model(mn_evaluator):
    """Provides a randomly initialized MLP model."""
    num_vars = mn_evaluator.num_variables
    return MLP(input_size=num_vars, hidden_sizes=[16], output_size=num_vars).to(DEVICE)


@pytest.fixture(scope="module")
def discretizer():
    """Provides a discretizer."""
    return ThresholdDiscretizer(threshold=0.5)


@pytest.fixture(scope="module")
def pretrained_model(mn_evaluator, dummy_dataloader):
    """Provides a model that has been pre-trained for a few epochs."""
    num_vars = mn_evaluator.num_variables
    model = MLP(input_size=num_vars, hidden_sizes=[16], output_size=num_vars).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SelfSupervisedTrainer(
        model=model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=optimizer,
        device=DEVICE,
    )
    # Train for 2 epochs to get a 'pre-trained' state
    final_model = trainer.fit(dummy_dataloader, num_epochs=2)
    return final_model


# --- Test Cases ---


def test_itself_from_random_init(random_model, mn_evaluator, dummy_dataloader, discretizer):
    """
    Tests if the ITSELF_Engine can run starting from a randomly initialized model.
    """
    engine = ITSELF_Engine(
        model=random_model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=discretizer,
        refinement_lr=1e-3,
        refinement_steps=3,
        device=DEVICE,
    )
    results = engine.run(dummy_dataloader)

    assert "final_assignments" in results
    num_samples = len(dummy_dataloader.dataset)
    num_vars = mn_evaluator.num_variables
    assert results["final_assignments"].shape == (num_samples, num_vars)
    # assert that the final assignments are binary
    assert torch.all(
        (results["final_assignments"] == 0) | (results["final_assignments"] == 1)
    ), "Final assignments are not binary"


def test_itself_from_pretrained(pretrained_model, mn_evaluator, dummy_dataloader, discretizer):
    """
    Tests if the ITSELF_Engine can run starting from a pre-trained model.
    """
    engine = ITSELF_Engine(
        model=pretrained_model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=discretizer,
        refinement_lr=1e-3,
        refinement_steps=3,
        device=DEVICE,
    )
    results = engine.run(dummy_dataloader)

    assert "final_assignments" in results
    num_samples = len(dummy_dataloader.dataset)
    num_vars = mn_evaluator.num_variables
    assert results["final_assignments"].shape == (num_samples, num_vars)
    # assert that the final assignments are binary
    assert torch.all((results["final_assignments"] == 0) | (results["final_assignments"] == 1))


def test_itself_improves_score(pretrained_model, mn_evaluator, dummy_dataloader, discretizer):
    """
    Tests the core hypothesis of ITSELF: that test-time refinement improves
    the log-likelihood of the final assignments compared to a simple forward pass.
    """
    # 1. Get initial scores from a simple forward pass (no refinement)
    simple_inference_engine = SinglePassInferenceEngine(
        model=pretrained_model, discretizer=discretizer, device=DEVICE
    )
    initial_results = simple_inference_engine.run(dummy_dataloader)
    initial_assignments = initial_results["final_assignments"].to(DEVICE)
    assert torch.all(
        (initial_assignments == 0) | (initial_assignments == 1)
    ), "Initial assignments are not binary"

    with torch.no_grad():
        initial_ll = mn_evaluator(initial_assignments).mean()

    # 2. Get refined scores using the ITSELF engine
    itself_engine = ITSELF_Engine(
        model=pretrained_model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=discretizer,
        refinement_lr=1e-3,
        refinement_steps=5,  # Use a few more steps to ensure improvement
        device=DEVICE,
    )
    refined_results = itself_engine.run(dummy_dataloader)
    refined_assignments = refined_results["final_assignments"].to(DEVICE)
    assert torch.all(
        (refined_assignments == 0) | (refined_assignments == 1)
    ), "Final assignments are not binary"

    with torch.no_grad():
        refined_ll = mn_evaluator(refined_assignments).mean()

    # 3. Assert that the refined log-likelihood is better (less negative)
    print(f"Initial Avg Log-Likelihood: {initial_ll.item():.4f}")
    print(f"Refined Avg Log-Likelihood: {refined_ll.item():.4f}")
    assert (
        refined_ll > initial_ll
    ), "ITSELF refinement failed to improve the average log-likelihood."
