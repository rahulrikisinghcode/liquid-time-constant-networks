"""Render the README visuals from the library's own output.

Not part of the installed package: this only exists to regenerate docs/img/.
Every number here comes from actually running the code -- the same training
loop as examples/irregular_sampling_benchmark.py (imported directly, not
duplicated), and the real ltc.features.mfcc() on a synthetic signal. Nothing
here is invented independently of the library.

Run with:  PYTHONPATH=src:examples python scripts/render_visuals.py
Needs:     pip install matplotlib (not a runtime dependency of the library
           itself; the benchmark's torch/numpy/scipy already are)
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "examples"))

import irregular_sampling_benchmark as bench  # noqa: E402

from ltc.features import MFCCConfig, mfcc  # noqa: E402

LTC_COLOR = "#1e8449"
GRU_COLOR = "#8e44ad"
GRID_COLOR = "#e5e8ea"


def style(ax):
    ax.grid(color=GRID_COLOR, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def run_benchmark():
    """Train both models exactly as the CLI benchmark does, capturing curves."""
    rng = np.random.default_rng(bench.SEED)
    train, test = bench.make_split(bench.N_TRAIN, rng), bench.make_split(bench.N_TEST, rng)

    torch.manual_seed(bench.SEED)
    ltc = bench.LTCClassifier(input_size=1, num_classes=len(bench.FREQS),
                              hidden_size=bench.HIDDEN, tau_init=bench.TAU_INIT)
    ltc_history: list[tuple[int, float]] = []
    bench.run(ltc, train, test, "ltc", history=ltc_history)

    torch.manual_seed(bench.SEED)
    gru = bench.GRUBaseline(bench.HIDDEN, len(bench.FREQS))
    gru_history: list[tuple[int, float]] = []
    bench.run(gru, train, test, "gru", history=gru_history)

    lo, hi = ltc.encoder.cell.effective_tau_bounds()
    tau_per_unit = hi.detach().numpy()
    return ltc_history, gru_history, tau_per_unit


def render_benchmark_panels(ax_curves, ax_tau, ltc_history, gru_history, tau_per_unit):
    ltc_x, ltc_y = zip(*ltc_history)
    gru_x, gru_y = zip(*gru_history)
    ax_curves.plot(ltc_x, ltc_y, marker="o", ms=4, color=LTC_COLOR, label="LTC (delta_t drives the dynamics)")
    ax_curves.plot(gru_x, gru_y, marker="o", ms=4, color=GRU_COLOR, label="GRU (delta_t as an extra input)")
    style(ax_curves)
    ax_curves.set_xlabel("epoch")
    ax_curves.set_ylabel("test accuracy")
    ax_curves.set_title("Irregular-sampling benchmark\n(examples/irregular_sampling_benchmark.py)",
                       fontsize=10.5, fontweight="bold")
    ax_curves.legend(fontsize=8, frameon=False)

    ax_tau.hist(tau_per_unit, bins=12, color=LTC_COLOR, alpha=0.75, edgecolor="white")
    style(ax_tau)
    ax_tau.set_xlabel("learned effective tau (time units)")
    ax_tau.set_ylabel("units")
    spread = tau_per_unit.max() - tau_per_unit.min()
    ax_tau.set_title(f"What each of the {len(tau_per_unit)} units settled on\n"
                    f"(spread {spread:.2f}: units specialised, not collapsed)",
                    fontsize=10.5, fontweight="bold")


def render_mfcc_panel(ax):
    """Same synthetic clip at two very different recording levels; the
    coefficients should land almost on top of each other. This is what
    test_mfcc_is_invariant_to_recording_level checks numerically -- this
    panel is the same claim, made visible."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    voice_like = (
        0.6 * np.sin(2 * np.pi * 180 * t)
        + 0.3 * np.sin(2 * np.pi * 540 * t)
        + 0.15 * np.sin(2 * np.pi * 900 * t)
    )
    rng = np.random.default_rng(0)
    voice_like += rng.normal(0, 0.01, size=voice_like.shape)

    loud = mfcc(voice_like, sr, MFCCConfig(include_deltas=False))
    quiet = mfcc(voice_like * 0.0025, sr, MFCCConfig(include_deltas=False))  # 400x quieter

    frame = loud.shape[0] // 2
    coeffs = np.arange(1, loud.shape[1])  # skip c0, the energy term, by convention
    ax.plot(coeffs, loud[frame, 1:], marker="o", ms=4, color="#2c3e50",
          label="1x gain (loud)")
    ax.plot(coeffs, quiet[frame, 1:], marker="x", ms=6, color="#c0392b", ls="--",
          label="1/400x gain (quiet)")
    style(ax)
    ax.set_xlabel("MFCC coefficient index")
    ax.set_ylabel("value")
    max_abs_diff = float(np.abs(loud[:, 1:] - quiet[:, 1:]).max())
    ax.set_title(f"MFCC shape is gain-invariant\n(max |diff| across the clip: {max_abs_diff:.2e})",
               fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)


def render():
    ltc_history, gru_history, tau_per_unit = run_benchmark()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    render_benchmark_panels(axes[0], axes[1], ltc_history, gru_history, tau_per_unit)
    render_mfcc_panel(axes[2])
    fig.suptitle("What the LTC actually does, from a live run of this repository's own code",
               fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = OUT / "ltc_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
