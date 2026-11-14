import copy
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from neupi.discretize.threshold import ThresholdDiscretizer
from neupi.inference.itself import ITSELF_Engine
from neupi.registry import register
from neupi.training.trainers.ssl_trainer import SelfSupervisedTrainer
from neupi.utils.pgm_utils import evaluate_log_likelihood


class _Indexed(Dataset):
    """Yields (index, *example), so that each training query has its own database entry."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return (i, *self.dataset[i])


def _index_collate(collate_fn):
    """Collates (index, *example) items with the user's collate function for the examples."""

    def collate(items):
        index = torch.tensor([item[0] for item in items])
        return (index, *collate_fn([item[1:] for item in items]))

    return collate


@register("trainer")
class GUIDETrainer(SelfSupervisedTrainer):
    """
    Teacher-student training (GUIDE: GUided Iterative Dual LEarning with Self-supervised Teacher).

    The trained network (the student S) learns from a teacher network T of the same
    structure. For every mini-batch:

    1. T is fitted to the mini-batch with ITSELF (`teacher_steps` gradient steps on the
       self-supervised loss, from a fresh optimizer);
    2. a forward pass of T gives an MPE assignment for each query (thresholded at 0.5), which
       replaces the query's entry in a database DB if it has a lower loss (higher
       log-likelihood);
    3. S takes `student_steps` gradient steps on the binary cross-entropy between its outputs
       for the query variables and the DB assignments;
    4. T is set to S.

    At the start T and S are initialized independently and DB is empty. The database
    entries are per training query, so each query keeps its evidence across epochs: use a
    dataset of fixed queries, e.g. `QueryDataset` or a `TensorDataset`.

    `fit`, `evaluate`, model selection, the scheduler, `history` and the callback work as
    in `SelfSupervisedTrainer`; the training loss reported per epoch is the student's
    binary cross-entropy, and the validation loss is the self-supervised loss. The
    `optimizer` updates the student; `model` is the student and the returned network.
    The best DB assignments and their losses are kept in `db_assignments` and `db_loss`.

    Args:
        model (torch.nn.Module): The student network.
        pgm_evaluator (torch.nn.Module): The probabilistic model evaluator.
        loss_fn (callable): The self-supervised loss used by ITSELF on the teacher.
        optimizer (torch.optim.Optimizer): The student's optimizer.
        device (str): The device to train on.
        teacher_steps (int): ITSELF steps on the teacher per mini-batch (the maximum if
            `tol` is set). Defaults to 100, the ITSELF limit in the paper.
        teacher_lr (float): The teacher's ITSELF learning rate. Defaults to 1e-3.
        teacher_optimizer_cls: The teacher's optimizer class. Defaults to Adam.
        tol (float, optional): Stop ITSELF on a mini-batch once the loss changes by less
            than `tol` for `patience` steps (the paper's "until convergence"). None runs all
            `teacher_steps`.
        patience (int): See `tol`. Defaults to 1.
        student_steps (int): Gradient steps of the student per mini-batch. Defaults to 1.
        entropy_weight (float): Weight of the entropy penalty in the teacher's
            self-supervised loss (and in the validation loss); negative values give an
            entropy bonus. Defaults to 0.0.
        entropy_aggregate (str): "avg" or "sum" for the entropy penalty.
        teacher (torch.nn.Module, optional): The initial teacher. Defaults to a copy of the
            student with freshly initialized parameters.

    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. G. (2024). A neural network approach for efficiently
    answering most probable explanation queries in probabilistic models. NeurIPS 2024,
    Algorithm 1. https://openreview.net/forum?id=ufPPf9ghzP
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pgm_evaluator: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        teacher_steps: int = 100,
        teacher_lr: float = 1e-3,
        teacher_optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        tol: Optional[float] = None,
        patience: int = 1,
        student_steps: int = 1,
        entropy_weight: float = 0.0,
        entropy_aggregate: str = "avg",
        teacher: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            model, pgm_evaluator, loss_fn, optimizer, device, entropy_weight, entropy_aggregate
        )
        if teacher_steps < 1 or student_steps < 1:
            raise ValueError("teacher_steps and student_steps must be at least 1.")
        if teacher is None:
            teacher = copy.deepcopy(model)
            # Line 4 of Algorithm 1: T and S are initialized randomly (and independently)
            for module in teacher.modules():
                if module is not teacher and hasattr(module, "reset_parameters"):
                    module.reset_parameters()  # type: ignore[operator]
            if hasattr(teacher, "initialize_weights"):
                teacher.initialize_weights()  # type: ignore[operator]
        self.teacher = teacher.to(device)
        self.student_steps = student_steps
        self.teacher_engine = ITSELF_Engine(
            self.teacher,
            self.pgm_evaluator,
            loss_fn,
            teacher_optimizer_cls,
            ThresholdDiscretizer(),
            refinement_lr=teacher_lr,
            refinement_steps=teacher_steps,
            device=device,
            tol=tol,
            patience=patience,
            entropy_weight=entropy_weight,
            entropy_aggregate=entropy_aggregate,
        )
        self.db_assignments: Optional[torch.Tensor] = None
        self.db_loss: Optional[torch.Tensor] = None

    def step(self, batch_data):
        """One GUIDE iteration (lines 6-13 of Algorithm 1) on a mini-batch (index, *query)."""
        assert self.db_assignments is not None and self.db_loss is not None
        index, evidence_data, evidence_mask, query_mask, unobs_mask = batch_data
        index = torch.as_tensor(index).cpu()
        batch = tuple(
            t.to(self.device) for t in (evidence_data, evidence_mask, query_mask, unobs_mask)
        )
        evidence_data, evidence_mask, query_mask, unobs_mask = batch

        # Lines 7 and 9: fit T to the mini-batch with ITSELF, then its (thresholded) assignments
        assignments, _, _ = self.teacher_engine._process_batch(batch, trained_model=self.teacher)
        with torch.no_grad():
            loss = -evaluate_log_likelihood(
                self.pgm_evaluator, assignments, unobs_mask if unobs_mask.any() else None
            )
            loss = loss.detach().cpu().to(self.db_loss.dtype)
            # Line 10: keep the better assignment for each query
            better = loss < self.db_loss[index]
            self.db_assignments[index[better]] = (
                assignments.detach().cpu().to(self.db_assignments.dtype)[better]
            )
            self.db_loss[index] = torch.where(better, loss, self.db_loss[index])

        # Line 12: supervised update of S with the database labels on the query variables
        self.model.train()
        if evidence_data.shape[0] == 1:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        labels = self.db_assignments[index].to(
            self.device, evidence_data.dtype if evidence_data.is_floating_point() else torch.float32
        )
        query = query_mask.bool()
        if not query.any():
            # No query variables in this mini-batch: nothing to learn
            self.teacher.load_state_dict(self.model.state_dict())
            return 0.0
        for _ in range(self.student_steps):
            logits = self.model(evidence_data, evidence_mask, query_mask, unobs_mask)
            student_loss = F.binary_cross_entropy_with_logits(
                logits[query], labels[query].to(logits.dtype)
            )
            self.optimizer.zero_grad()
            student_loss.backward()
            self.optimizer.step()

        # Line 13: T <- S
        self.teacher.load_state_dict(self.model.state_dict())
        return student_loss.item()

    def fit(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        scheduler=None,
        verbose: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
        selection: Union[str, Callable[[torch.nn.Module, DataLoader], float]] = "assignment",
    ):
        """
        Trains the student with GUIDE; the arguments are those of `SelfSupervisedTrainer.fit`.
        `dataloader` must iterate over a dataset of fixed queries (its batch size and
        shuffling are kept).
        """
        dataset = dataloader.dataset
        num_examples = len(dataset)  # type: ignore[arg-type]
        if self.db_assignments is None or len(self.db_assignments) != num_examples:
            num_vars = dataset[0][0].shape[-1]
            self.db_assignments = torch.zeros(num_examples, num_vars)
            self.db_loss = torch.full((num_examples,), float("inf"), dtype=torch.float64)
        indexed = DataLoader(
            _Indexed(dataset),
            batch_size=dataloader.batch_size,
            shuffle=isinstance(dataloader.sampler, RandomSampler),
            drop_last=dataloader.drop_last,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            generator=dataloader.generator,
            collate_fn=_index_collate(dataloader.collate_fn),
        )
        model = super().fit(
            indexed, num_epochs, val_dataloader, scheduler, verbose, callback, selection
        )
        self.teacher.load_state_dict(self.model.state_dict())
        return model
