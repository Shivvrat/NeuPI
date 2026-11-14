from pathlib import Path

import pytest
import torch
from neupi import (
    MLP,
    DiscreteEmbedder,
    GUIDETrainer,
    ITSELF_Engine,
    KNearestDiscretizer,
    OAUAI,
    ProbabilisticGraphicalModel,
    SelfSupervisedTrainer,
    ThresholdDiscretizer,
    make_query_masks,
    mpe_log_likelihood_loss,
    default_hparams,
)
from torch.utils.data import DataLoader, TensorDataset

GRID = str(Path(__file__).parent.parent.parent / "examples/networks/mn/Grids_17.uai")


@pytest.mark.parametrize("method", ["ssmp", "guide"])
def test_default_hparams_builds_and_trains(method):
    cfg = default_hparams(method)
    pgm = ProbabilisticGraphicalModel(GRID)
    n = pgm.num_variables
    x = torch.randint(0, 2, (16, n)).float()
    e, q, u = make_query_masks(16, n, query_ratio=0.5, per_example=True, seed=0)
    loader = DataLoader(TensorDataset(x * e, e, q, u), batch_size=8)
    embedding = DiscreteEmbedder(n, **cfg["embedding"])
    model = MLP(output_size=n, embedding=embedding, **cfg["model"])
    optimizer = torch.optim.Adam(model.parameters(), **cfg["optimizer"])
    trainer_cls = GUIDETrainer if method == "guide" else SelfSupervisedTrainer
    trainer = trainer_cls(model, pgm, mpe_log_likelihood_loss, optimizer, **cfg["trainer"])
    fit = dict(cfg["fit"], num_epochs=1)
    trainer.fit(
        loader,
        val_dataloader=loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg["scheduler"]),
        verbose=False,
        **fit,
    )
    itself = dict(cfg["itself"], refinement_steps=2)
    engine = ITSELF_Engine(
        trainer.model, pgm, mpe_log_likelihood_loss, torch.optim.Adam, ThresholdDiscretizer(), **itself
    )
    assert engine.run(loader)["final_assignments"].shape == (16, n)
    KNearestDiscretizer(pgm, **cfg["discretizers"]["knn"])
    OAUAI(pgm, **cfg["discretizers"]["oauai"])
    assert cfg["batch_size"] > 0


def test_default_hparams_is_a_copy_and_validates():
    cfg = default_hparams("ssmp", "pc")
    assert cfg["model"]["hidden_sizes"] == [128, 256, 512]
    cfg["model"]["hidden_sizes"].append(1)
    cfg["optimizer"]["lr"] = 1.0
    assert default_hparams("ssmp", "pc")["model"]["hidden_sizes"] == [128, 256, 512]
    assert default_hparams("ssmp")["optimizer"]["lr"] == 1e-3
    with pytest.raises(ValueError, match="method"):
        default_hparams("sine")
    with pytest.raises(ValueError, match="model_type"):
        default_hparams("ssmp", "bn")
