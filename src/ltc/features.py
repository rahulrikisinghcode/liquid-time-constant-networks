"""MFCC front end matching the AudioFacialMatrix audio preprocessing.

Section IV-B of the paper resamples every clip to 16 kHz, extracts MFCCs from
overlapping windows, and normalises the feature vectors across the dataset.
This module implements that with numpy and scipy only, so the feature path has
no heavyweight audio dependency and stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample_poly

TARGET_SR = 16_000


def resample(signal: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample to ``target_sr`` using a polyphase filter."""
    if sr <= 0:
        raise ValueError("sr must be positive")
    if sr == target_sr:
        return signal.astype(np.float64, copy=False)
    g = np.gcd(int(sr), int(target_sr))
    return resample_poly(signal, target_sr // g, sr // g).astype(np.float64)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank, shape (n_filters, n_fft // 2 + 1)."""
    if fmax > sr / 2:
        raise ValueError("fmax cannot exceed the Nyquist frequency")
    points = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2))
    bins = np.floor((n_fft + 1) * points / sr).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre > left:
            fb[i, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            fb[i, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return fb


@dataclass(frozen=True)
class MFCCConfig:
    """Window and filterbank settings.

    ``win_length_s`` of 25 ms with a 10 ms hop is the standard speech
    configuration and gives the overlapping windows the paper describes.
    """

    sample_rate: int = TARGET_SR
    win_length_s: float = 0.025
    hop_length_s: float = 0.010
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 20.0
    fmax: float = 8000.0
    preemphasis: float = 0.97
    include_deltas: bool = True

    @property
    def win_length(self) -> int:
        return int(round(self.win_length_s * self.sample_rate))

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_length_s * self.sample_rate))

    @property
    def n_fft(self) -> int:
        return int(2 ** np.ceil(np.log2(self.win_length)))

    @property
    def n_features(self) -> int:
        return self.n_mfcc * (3 if self.include_deltas else 1)


def _frame(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(signal) < win:
        signal = np.pad(signal, (0, win - len(signal)))
    n = 1 + (len(signal) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def _deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    weights = np.arange(-width, width + 1)
    norm = 2.0 * (weights ** 2).sum()
    out = np.zeros_like(features)
    for i, w in enumerate(weights):
        out += w * padded[i : i + len(features)]
    return out / norm


def mfcc(signal: np.ndarray, sr: int, config: MFCCConfig | None = None) -> np.ndarray:
    """MFCC matrix of shape (frames, n_features).

    With ``include_deltas`` the first and second order differences are appended,
    which is what makes the representation carry the rate of spectral change
    rather than only its instantaneous value.
    """
    config = config or MFCCConfig()
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError("expected a mono signal")
    if signal.size == 0:
        raise ValueError("signal is empty")

    signal = resample(signal, sr, config.sample_rate)
    signal = np.append(signal[0], signal[1:] - config.preemphasis * signal[:-1])

    frames = _frame(signal, config.win_length, config.hop_length) * np.hamming(config.win_length)
    power = np.abs(np.fft.rfft(frames, n=config.n_fft)) ** 2 / config.n_fft

    fb = mel_filterbank(config.n_mels, config.n_fft, config.sample_rate, config.fmin, config.fmax)
    energies = power @ fb.T
    # The log floor is relative to the clip's own peak, not absolute. An absolute
    # floor clips more bins in a quiet recording than a loud one, which destroys
    # the gain invariance of coefficients 1..n and makes recording level leak into
    # the features. AFM clips come from arbitrary YouTube sources at arbitrary
    # levels, so this matters.
    energies = np.maximum(energies, max(energies.max() * 1e-10, 1e-30))
    coeffs = dct(np.log(energies), type=2, axis=1, norm="ortho")[:, : config.n_mfcc]

    if not config.include_deltas:
        return coeffs
    d1 = _deltas(coeffs)
    return np.hstack([coeffs, d1, _deltas(d1)])


def normalize(features: np.ndarray, mean: np.ndarray | None = None,
              std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise per coefficient, returning the statistics used.

    Fit the statistics on training data and pass them back in for validation and
    test. Recomputing them per split leaks information across the split, which is
    an easy mistake to make with a per-utterance normalisation helper.
    """
    mean = features.mean(axis=0) if mean is None else mean
    std = features.std(axis=0) if std is None else std
    return (features - mean) / np.maximum(std, 1e-8), mean, std
