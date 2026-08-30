"""Where a liquid time constant actually buys something.

Run with:  PYTHONPATH=src python examples/irregular_sampling_benchmark.py

The task is deliberately chosen so that the elapsed time between observations
carries the signal. Each sequence is a sinusoid at one of four frequencies,
sampled at random, uneven intervals. The sampled values alone are close to
uninformative about the frequency; the values together with their spacing are
not.

A GRU has no notion of elapsed time, so the baseline is given delta_t as an extra
input channel, which is the fair comparison and the usual workaround. The LTC
consumes delta_t in its dynamics instead, where it changes the integration step
rather than being another number to learn a function of.

This is a small synthetic task on CPU. It is here to make the mechanism visible,
not to claim a benchmark result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ltc import LTCClassifier

SEED = 0
FREQS = (0.5, 1.5, 4.0, 9.0)
N_TRAIN, N_TEST, STEPS = 1200, 400, 40
HIDDEN, EPOCHS, BATCH = 32, 25, 64
# sequences span roughly sum(gaps) ~ 7.4 time units, so start the units with a
# memory horizon of that order rather than the default 1.0
TAU_INIT = 8.0


def make_split(n: int, rng: np.random.Generator):
    labels = rng.integers(0, len(FREQS), size=n)
    gaps = rng.uniform(0.02, 0.35, size=(n, STEPS))
    times = np.cumsum(gaps, axis=1)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    values = np.sin(2 * np.pi * np.array(FREQS)[labels][:, None] * times + phase)
    values += rng.normal(0, 0.05, size=values.shape)
    return (
        torch.tensor(values[..., None], dtype=torch.float32),
        torch.tensor(gaps, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


class GRUBaseline(nn.Module):
    """GRU over [value, delta_t], the standard way to feed a discrete RNN time."""

    def __init__(self, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, delta_t):
        return self.head(self.rnn(torch.cat([x, delta_t[..., None]], dim=-1))[0][:, -1])


def run(model, train, test, tag: str, history: list | None = None) -> float:
    """Train and return final test accuracy. If `history` is given, appends
    (epoch, test_acc) to it every 5 epochs -- used by scripts/render_visuals.py
    to plot real training curves; unused by the default CLI run."""
    x, dt, y = train
    xt, dtt, yt = test
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(y))
        for i in range(0, len(y), BATCH):
            idx = order[i : i + BATCH]
            opt.zero_grad()
            loss_fn(model(x[idx], dt[idx]), y[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(xt, dtt).argmax(-1) == yt).float().mean().item()
            print(f"  {tag}  epoch {epoch + 1:>3}  test acc {acc:.3f}")
            if history is not None:
                history.append((epoch + 1, acc))

    model.eval()
    with torch.no_grad():
        return (model(xt, dtt).argmax(-1) == yt).float().mean().item()


def main() -> None:
    rng = np.random.default_rng(SEED)
    train, test = make_split(N_TRAIN, rng), make_split(N_TEST, rng)

    torch.manual_seed(SEED)
    ltc = LTCClassifier(
        input_size=1, num_classes=len(FREQS), hidden_size=HIDDEN, tau_init=TAU_INIT
    )
    print("LTC (delta_t drives the integration step)")
    ltc_acc = run(ltc, train, test, "ltc")

    torch.manual_seed(SEED)
    gru = GRUBaseline(HIDDEN, len(FREQS))
    print("\nGRU (delta_t as an extra input channel)")
    gru_acc = run(gru, train, test, "gru")

    n_ltc = sum(p.numel() for p in ltc.parameters())
    n_gru = sum(p.numel() for p in gru.parameters())
    print(
        f"\nfinal: LTC {ltc_acc:.3f} ({n_ltc} params)   "
        f"GRU {gru_acc:.3f} ({n_gru} params)"
    )

    report = ltc.timescale_report()
    print(
        "\nlearned timescales: "
        f"min {report['tau_min']:.3f}, median {report['tau_median']:.3f}, "
        f"max {report['tau_max']:.3f}, spread {report['tau_spread']:.3f}"
    )
    print(
        "A wide spread means units specialised to different rates. A spread near "
        "zero would mean the liquid part is not earning its cost here."
    )


if __name__ == "__main__":
    main()
