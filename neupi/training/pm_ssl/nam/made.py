import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
from neupi.core.model import BaseProbModel
from neupi.registry import register


class MaskedLinear(nn.Linear):
    """same as Linear except has a configurable mask on the weights"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


@register("prob_model")
class MADE(BaseProbModel):
    """
    A MADE (masked autoencoder for distribution estimation) over binary variables.

    Like the other evaluators, ``forward(x)`` returns the per-sample log-likelihood
    log p(x), so a trained MADE can be passed to the trainers and inference engines.
    The network outputs (one logit per variable) are available from ``get_logits(x)``. To
    train the MADE itself on data, minimize ``-made(x).mean()``.
    """

    def __init__(self, nin, hidden_sizes, nout=None, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs (defaults to nin, which is what the log-likelihood needs), which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        if nout is None:
            nout = nin
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.sigmoid = nn.Sigmoid()
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            low = self.m[l - 1].min()
            # randint's upper bound is exclusive; with one input the only label is 0
            self.m[l] = rng.randint(low, max(self.nin - 1, low + 1), size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    @property
    def num_variables(self) -> int:
        return self.nin

    @property
    def num_var(self) -> int:
        return self.nin

    def get_logits(self, x):
        """The network outputs: one Bernoulli logit per variable (for nout == nin)."""
        return self.net(x)

    def forward(self, x):
        """Alias for `evaluate`: the log-likelihood of each assignment, shape (batch_size,),
        on the device of x."""
        input_device = x.device if isinstance(x, torch.Tensor) else next(self.parameters()).device
        return self.evaluate(x).to(input_device)

    def evaluate(self, x):
        """
        Log-likelihood of assignments in [0, 1] of shape (batch_size, nin) or (nin,).
        """
        if self.nout != self.nin:
            raise ValueError("The log-likelihood needs nout == nin (one logit per variable).")
        param = next(self.parameters())
        x = torch.as_tensor(x, device=param.device)
        x = x.to(param.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.nin:
            raise ValueError(f"Expected assignments of shape (batch_size, {self.nin}), got {tuple(x.shape)}.")
        distributions = self.get_logits(x)
        # This is a more numerically stable implementation of the log likelihood - uses logsumexp trick
        log_likelihoods = -(F.binary_cross_entropy_with_logits(distributions, x, reduction="none"))
        log_likelihoods = torch.sum(log_likelihoods, dim=1)
        return log_likelihoods
