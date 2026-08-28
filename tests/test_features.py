import numpy as np
import pytest

from ltc.features import MFCCConfig, mel_filterbank, mfcc, normalize, resample


def tone(freq, sr=16000, seconds=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_resample_changes_length_proportionally():
    out = resample(tone(440, sr=44100), 44100, 16000)
    assert abs(len(out) - 8000) < 50


def test_resample_is_identity_at_target_rate():
    x = tone(440)
    assert np.allclose(resample(x, 16000, 16000), x)


def test_filterbank_shape_and_nonnegativity():
    fb = mel_filterbank(40, 512, 16000, 20.0, 8000.0)
    assert fb.shape == (40, 257)
    assert (fb >= 0).all()


def test_filterbank_rejects_fmax_above_nyquist():
    with pytest.raises(ValueError):
        mel_filterbank(40, 512, 16000, 20.0, 9000.0)


def test_mfcc_shape_matches_config():
    cfg = MFCCConfig()
    out = mfcc(tone(440), 16000, cfg)
    assert out.shape[1] == cfg.n_features == 39
    assert np.isfinite(out).all()


def test_mfcc_without_deltas():
    cfg = MFCCConfig(include_deltas=False)
    assert mfcc(tone(440), 16000, cfg).shape[1] == cfg.n_mfcc


def test_mfcc_separates_different_pitches():
    a = mfcc(tone(220), 16000).mean(axis=0)
    b = mfcc(tone(1800), 16000).mean(axis=0)
    assert np.linalg.norm(a - b) > 1.0


@pytest.mark.parametrize("gain", [0.5, 0.1, 0.01, 4.0])
def test_mfcc_is_invariant_to_recording_level(gain):
    # a gain change shifts c0 by a constant and must leave the shape
    # coefficients untouched; an absolute log floor silently breaks this
    quiet = mfcc(tone(440) * gain, 16000)
    loud = mfcc(tone(440), 16000)
    assert np.allclose(quiet[:, 1:13], loud[:, 1:13], atol=1e-9)


def test_short_signal_is_padded_not_rejected():
    assert mfcc(np.ones(100), 16000).shape[0] >= 1


def test_empty_signal_rejected():
    with pytest.raises(ValueError):
        mfcc(np.array([]), 16000)


def test_normalize_returns_reusable_statistics():
    train = mfcc(tone(440), 16000)
    _, mean, std = normalize(train)
    test = mfcc(tone(450), 16000)
    applied, m2, s2 = normalize(test, mean, std)
    assert np.allclose(m2, mean) and np.allclose(s2, std)
    assert applied.shape == test.shape
