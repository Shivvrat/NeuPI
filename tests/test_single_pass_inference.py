from pathlib import Path

import pytest
import torch
from neupi.discretize.threshold import ThresholdDiscretizer
from neupi.inference.single_pass import SinglePassInferenceEngine
from neupi.losses import mpe_log_likelihood_loss
from neupi.models.nn import MLP

# Import components from your NeuPI library structure
from neupi.pm_ssl import MarkovNetwork
from neupi.training.ssl_trainer import SelfSupervisedTrainer
from torch.utils.data import DataLoader, TensorDataset

# Define the device for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def discretizer():
    """Provides a discretizer."""
    return ThresholdDiscretizer(threshold=0.5)


@pytest.fixture(scope="module")
def trained_model():
    """Pytest fixture to train a model for a few epochs and return it."""
    # 1. Setup PGM evaluator and dummy data
    uai_path = Path(__file__).parent / "networks/mn/Grids_17.uai"
    mn_evaluator = MarkovNetwork(uai_file=str(uai_path), device=DEVICE)

    num_samples = 16
    num_vars = mn_evaluator.num_variables
    inputs = torch.rand(num_samples, num_vars, device=DEVICE)
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), device=DEVICE, dtype=torch.float32)
    evidence_mask = torch.rand(num_samples, num_vars, device=DEVICE) > 0.5
    dataset = TensorDataset(inputs, evidence_data, evidence_mask)
    dataloader = DataLoader(dataset, batch_size=4)

    # 2. Setup model and trainer
    model = MLP(input_size=num_vars, hidden_sizes=[16], output_size=num_vars).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = SelfSupervisedTrainer(
        model=model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 3. Train for a couple of epochs and get the trained model
    final_model = trainer.train(dataloader, num_epochs=2)
    return final_model


def test_inference_engine_output(trained_model, discretizer):
    """
    Tests that the InferenceEngine runs and returns outputs of the correct shape and type.
    """
    # 1. Create a dummy dataloader for inference
    num_samples = 10
    # The number of variables is the output size of the trained model's final layer
    num_vars = trained_model.output_layer.out_features
    inputs = torch.rand(num_samples, num_vars, device=DEVICE)
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), device=DEVICE, dtype=torch.float32)
    evidence_mask = torch.rand(num_samples, num_vars, device=DEVICE) > 0.5
    dataset = TensorDataset(inputs, evidence_data, evidence_mask)
    dataloader = DataLoader(dataset, batch_size=5)

    # 2. Initialize and run the inference engine
    inference_engine = SinglePassInferenceEngine(
        model=trained_model, discretizer=discretizer, device=DEVICE
    )
    results = inference_engine.run(dataloader)

    # 3. Validate the results
    assert isinstance(results, dict)
    assert "raw_outputs" in results
    assert "prob_outputs" in results
    assert "final_assignments" in results

    # Check shapes
    assert results["raw_outputs"].shape == (num_samples, num_vars)
    assert results["prob_outputs"].shape == (num_samples, num_vars)
    assert results["final_assignments"].shape == (num_samples, num_vars)
    # assert that the final assignments are binary
    assert torch.all(
        (results["final_assignments"] == 0) | (results["final_assignments"] == 1)
    ), "Final assignments are not binary"

    # Check that probabilities are in the correct range [0, 1]
    assert torch.all(results["prob_outputs"] >= 0) and torch.all(results["prob_outputs"] <= 1)
