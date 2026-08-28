"""The liquid time-constant recurrent cell.

An LTC unit is a continuous-time state whose *time constant depends on the
input*. The dynamics are

    dx/dt = -[1/tau + f(x, I)] * x + f(x, I) * g(x, I)

with a learnable positive tau, a conductance

    f(x, I) = sigmoid(W_f I + U_f x + b_f)   in (0, 1)

and a signed target

    g(x, I) = W_g I + U_g x + b_g.

``f`` sets *how fast* the state moves and ``g`` sets *where toward*. Both depend
on the input. Splitting them is a small departure from writing the target as a
per-unit constant A: with a constant target the input can only modulate the rate,
the state relaxes to the same attractor whatever the sequence was, and the batch
variance of the final state collapses to roughly 1e-6, which is a model that
cannot be trained. In the original per-synapse formulation the reversal
potentials are summed with the synaptic activations, so the target is already
input-dependent; ``g`` recovers that in the dense form.

The bracketed term is the effective decay rate, and it is a function of the
input. That is the whole difference from a gated RNN. A GRU interpolates between
carrying and replacing its state; an LTC changes how fast its state evolves. The
system therefore has a time constant that varies per unit and per timestep,
bounded by

    1 / (1/tau + 1) <= tau_effective <= tau

which ``effective_tau_bounds`` returns.

Integration uses the fused semi-implicit Euler step from Hasani et al.,

    x_{k+1} = (x_k + delta * f * A) / (1 + delta * (1/tau + f))

which is unconditionally stable for positive tau and f, unlike explicit Euler.
Explicit Euler on a stiff LTC will diverge at exactly the input scales the model
is supposed to be good at, which is the failure mode worth knowing about.

Reference: R. Hasani, M. Lechner, A. Amini, D. Rus and R. Grosu,
"Liquid Time-constant Networks," AAAI 2021, vol. 35, no. 9, pp. 7657-7666.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """One LTC layer, stepped over a single timestep.

    Parameters
    ----------
    input_size:
        Feature width of the input at each timestep.
    hidden_size:
        Number of LTC units.
    ode_unfolds:
        Solver substeps per timestep. More substeps track fast dynamics more
        accurately at linear cost. Six is the value used in the reference
        implementation and is a reasonable default.
    tau_init:
        Initial time constant, in units of the timestep. Set it near the
        timescale you expect to matter; the model learns from there. For a
        sequence spanning T units of time, ``tau_init`` on the order of T is a
        better starting point than 1.0.
    gate_bias_init:
        Initial bias on the conductance. Negative values start the units in the
        long-memory regime. See :meth:`reset_parameters`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ode_unfolds: int = 6,
        tau_init: float = 1.0,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        if ode_unfolds < 1:
            raise ValueError("ode_unfolds must be at least 1")
        if tau_init <= 0:
            raise ValueError("tau_init must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds

        self.conductance_in = nn.Linear(input_size, hidden_size, bias=False)
        self.conductance_rec = nn.Linear(hidden_size, hidden_size, bias=True)
        self.target_in = nn.Linear(input_size, hidden_size, bias=False)
        self.target_rec = nn.Linear(hidden_size, hidden_size, bias=True)

        # tau is parameterised through softplus so it stays positive without a
        # clamp that would zero the gradient at the boundary.
        inv = torch.expm1(torch.tensor(tau_init)).clamp_min(1e-6).log()
        self.tau_raw = nn.Parameter(torch.full((hidden_size,), float(inv)))
        self.gate_bias_init = gate_bias_init

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.conductance_in, self.target_in):
            nn.init.xavier_uniform_(linear.weight)
        for linear in (self.conductance_rec, self.target_rec):
            nn.init.orthogonal_(linear.weight)
        # A negative conductance bias starts f near zero, so the effective decay
        # rate starts near 1/tau and the units begin in the long-memory regime.
        # Training shortens the ones that need to be short. Starting at f = 0.5
        # instead puts a floor of 0.5 on the decay rate whatever tau is, which
        # caps the memory horizon below most useful sequence lengths.
        nn.init.constant_(self.conductance_rec.bias, self.gate_bias_init)
        nn.init.zeros_(self.target_rec.bias)

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw) + 1e-6

    def effective_tau_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-unit (min, max) effective time constant.

        f is a sigmoid so it lies in (0, 1), which bounds the effective decay
        rate between 1/tau and 1/tau + 1. Inspecting these after training shows
        which units specialised to fast structure and which to slow.
        """
        tau = self.tau
        return 1.0 / (1.0 / tau + 1.0), tau

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        delta_t: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Advance ``state`` by ``delta_t``.

        ``delta_t`` may be a scalar or a per-sample tensor of shape (batch, 1).
        Per-sample values are what let the cell consume irregularly sampled
        sequences directly, without resampling them onto a uniform grid.
        """
        if x.dim() != 2 or state.dim() != 2:
            raise ValueError("expected x and state of shape (batch, features)")
        if x.shape[0] != state.shape[0]:
            raise ValueError("x and state disagree on batch size")

        if not torch.is_tensor(delta_t):
            delta_t = torch.as_tensor(delta_t, dtype=x.dtype, device=x.device)
        step = delta_t / self.ode_unfolds

        conductance_drive = self.conductance_in(x)
        target_drive = self.target_in(x)
        tau_inv = 1.0 / self.tau

        for _ in range(self.ode_unfolds):
            f = torch.sigmoid(conductance_drive + self.conductance_rec(state))
            g = target_drive + self.target_rec(state)
            numerator = state + step * f * g
            denominator = 1.0 + step * (tau_inv + f)
            state = numerator / denominator
        return state

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        reference = self.tau_raw
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"ode_unfolds={self.ode_unfolds}"
        )
