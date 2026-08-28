"""Sequence encoder and classifier built on :class:`LTCCell`.

Defaults follow the LTNN configuration described in Section IV-B of
"AudioFacialMatrix: Dataset for Voice and Face AI": a 128-unit recurrent layer
feeding a fully connected classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cell import LTCCell


class LTCEncoder(nn.Module):
    """Run an LTC cell over a sequence and return the states.

    Parameters
    ----------
    return_sequence:
        If ``True``, return every state, shape (batch, time, hidden). Otherwise
        return the final state, shape (batch, hidden).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        ode_unfolds: int = 6,
        tau_init: float = 1.0,
        gate_bias_init: float = -2.0,
        return_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size, ode_unfolds, tau_init, gate_bias_init)
        self.return_sequence = return_sequence

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor | float = 1.0,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of sequences.

        Parameters
        ----------
        x:
            Shape (batch, time, input_size).
        delta_t:
            Scalar, or shape (batch, time) for irregular sampling. Variable step
            sizes are handled natively, which is the practical reason to reach
            for an LTC over a discrete RNN on unevenly sampled data.
        lengths:
            Optional true lengths, shape (batch,). When given, the returned
            final state is the state at each sequence's own last valid step
            rather than at the padded end.
        """
        if x.dim() != 3:
            raise ValueError("expected x of shape (batch, time, input_size)")
        batch, steps, _ = x.shape

        if torch.is_tensor(delta_t) and delta_t.dim() == 2:
            if delta_t.shape != (batch, steps):
                raise ValueError("delta_t of shape (batch, time) must match x")
            per_step = [delta_t[:, t : t + 1] for t in range(steps)]
        else:
            per_step = [delta_t] * steps

        state = self.cell.init_state(batch, x.device, x.dtype)
        states = []
        for t in range(steps):
            state = self.cell(x[:, t], state, per_step[t])
            states.append(state)

        stacked = torch.stack(states, dim=1)
        if self.return_sequence:
            return stacked

        if lengths is None:
            return stacked[:, -1]
        idx = (lengths.to(stacked.device).long() - 1).clamp_min(0)
        return stacked[torch.arange(batch, device=stacked.device), idx]


class LTCClassifier(nn.Module):
    """LTC encoder plus a linear head, returning logits.

    Logits rather than probabilities, so the caller pairs this with
    ``nn.CrossEntropyLoss`` and no softmax is applied twice.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        ode_unfolds: int = 6,
        tau_init: float = 1.0,
        gate_bias_init: float = -2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = LTCEncoder(input_size, hidden_size, ode_unfolds, tau_init, gate_bias_init)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor | float = 1.0,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.dropout(self.encoder(x, delta_t, lengths)))

    def timescale_report(self) -> dict[str, float]:
        """Summary of what timescales the units settled on after training.

        Useful as a sanity check: if every unit converged to the same effective
        tau, the liquid part is not earning its cost and a GRU would do.
        """
        with torch.no_grad():
            lo, hi = self.encoder.cell.effective_tau_bounds()
        return {
            "tau_min": float(hi.min()),
            "tau_max": float(hi.max()),
            "tau_median": float(hi.median()),
            "tau_spread": float(hi.max() - hi.min()),
            "effective_floor_min": float(lo.min()),
        }
