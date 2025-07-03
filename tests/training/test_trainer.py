from pathlib import Path

import pytest
import torch
from neupi import MLP, MarkovNetwork, SelfSupervisedTrainer, mpe_log_likelihood_loss
from torch.utils.data import DataLoader, TensorDataset

# Define the device for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def mn_evaluator():
    """Pytest fixture to load a Markov Network evaluator for the tests."""
    # Using a test file from your provided structure
    uai_path = Path(__file__).parent.parent / "networks/mn/Grids_17.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


@pytest.fixture(scope="module")
def dummy_dataloader(mn_evaluator):
    """Pytest fixture to create a dummy DataLoader for training."""
    num_samples = 16
    num_vars = mn_evaluator.num_variables

    # Create random data for the test
    # In a real scenario, inputs would be features derived from the PGM state.
    # For this test, we use random tensors as placeholders.
    inputs = torch.rand(num_samples, num_vars, device=DEVICE)

    # The evidence data assignments (e.g., from sampled data)
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), device=DEVICE, dtype=torch.float32)

    # A random mask where ~50% of variables are evidence
    evidence_mask = torch.rand(num_samples, num_vars, device=DEVICE) > 0.5

    dataset = TensorDataset(inputs, evidence_data, evidence_mask)
    return DataLoader(dataset, batch_size=4)


def test_trainer_decreases_loss(mn_evaluator, dummy_dataloader):
    """
    Tests the end-to-end training pipeline.

    This test verifies that after a few training steps, the loss decreases,
    indicating that the model is learning and the backpropagation is working.
    """
    num_vars = mn_evaluator.num_variables

    # 1. Initialize the Model (Neural Network)
    model = MLP(
        input_size=num_vars,
        hidden_sizes=[32, 16],
        output_size=num_vars,
        hidden_activation="relu",
        use_batchnorm=True,
    ).to(DEVICE)

    # 2. Initialize the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Initialize the Trainer
    # Using the mpe_log_likelihood_loss from your library
    trainer = SelfSupervisedTrainer(
        model=model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 4. Perform a few training steps
    num_steps = 100
    initial_loss = None
    final_loss = None

    # Get a single batch to run multiple steps on
    initial_batch = next(iter(dummy_dataloader))

    for i in range(num_steps):
        loss = trainer.step(initial_batch)
        if i == 0:
            initial_loss = loss
        if i == num_steps - 1:
            final_loss = loss
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}")

    # 5. Assert that the training process is working
    assert initial_loss is not None
    assert final_loss is not None
    # A key check: the loss should decrease after several updates.
    assert final_loss < initial_loss, "Loss did not decrease after training steps."
    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
