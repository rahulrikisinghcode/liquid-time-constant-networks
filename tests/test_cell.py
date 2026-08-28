import pytest
import torch

from ltc import LTCCell, LTCClassifier, LTCEncoder


def test_state_shape_and_finiteness():
    cell = LTCCell(4, 8)
    state = cell.init_state(3)
    out = cell(torch.randn(3, 4), state)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_solver_is_stable_on_large_inputs():
    # explicit Euler diverges here; the semi-implicit step must not
    cell = LTCCell(4, 16, ode_unfolds=6)
    state = cell.init_state(2)
    for _ in range(200):
        state = cell(torch.randn(2, 4) * 100.0, state, delta_t=5.0)
    assert torch.isfinite(state).all()
    assert state.abs().max() < 1e3


def test_tau_stays_positive_after_hostile_updates():
    cell = LTCCell(3, 6)
    with torch.no_grad():
        cell.tau_raw.fill_(-50.0)
    assert (cell.tau > 0).all()
    lo, hi = cell.effective_tau_bounds()
    assert (lo > 0).all() and (lo <= hi).all()


def test_delta_t_changes_the_trajectory():
    torch.manual_seed(0)
    cell = LTCCell(3, 6)
    x, state = torch.randn(2, 3), cell.init_state(2)
    assert not torch.allclose(cell(x, state, 0.1), cell(x, state, 2.0))


def test_per_sample_delta_t():
    cell = LTCCell(3, 6)
    x, state = torch.randn(4, 3), cell.init_state(4)
    dt = torch.tensor([[0.1], [0.5], [1.0], [2.0]])
    out = cell(x, state, dt)
    assert out.shape == (4, 6)
    assert not torch.allclose(out[0], out[3])


def test_more_unfolds_converge():
    torch.manual_seed(0)
    coarse = LTCCell(3, 6, ode_unfolds=32)
    fine = LTCCell(3, 6, ode_unfolds=64)
    fine.load_state_dict(coarse.state_dict())
    x, state = torch.randn(2, 3), coarse.init_state(2)
    assert torch.allclose(coarse(x, state, 1.0), fine(x, state, 1.0), atol=1e-3)


def test_gradients_reach_every_parameter():
    model = LTCClassifier(5, 3, hidden_size=12)
    logits = model(torch.randn(4, 15, 5))
    torch.nn.functional.cross_entropy(logits, torch.randint(0, 3, (4,))).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_encoder_respects_lengths():
    enc = LTCEncoder(4, 8)
    x = torch.randn(3, 10, 4)
    lengths = torch.tensor([10, 5, 1])
    seq = LTCEncoder(4, 8, return_sequence=True)
    seq.load_state_dict(enc.state_dict())
    full = seq(x)
    picked = enc(x, lengths=lengths)
    for i, L in enumerate(lengths.tolist()):
        assert torch.allclose(picked[i], full[i, L - 1], atol=1e-6)


def test_classifier_returns_logits_not_probabilities():
    logits = LTCClassifier(4, 3, hidden_size=8)(torch.randn(2, 6, 4))
    assert logits.shape == (2, 3)
    assert not torch.allclose(logits.exp().sum(-1), torch.ones(2))


def test_timescale_report_keys():
    report = LTCClassifier(4, 2, hidden_size=8).timescale_report()
    assert set(report) == {
        "tau_min", "tau_max", "tau_median", "tau_spread", "effective_floor_min"
    }
    assert report["tau_min"] > 0


@pytest.mark.parametrize(
    "kwargs", [dict(input_size=0, hidden_size=4), dict(input_size=4, hidden_size=4, ode_unfolds=0),
               dict(input_size=4, hidden_size=4, tau_init=0.0)]
)
def test_constructor_validates(kwargs):
    with pytest.raises(ValueError):
        LTCCell(**kwargs)


def test_encoder_rejects_wrong_rank():
    with pytest.raises(ValueError):
        LTCEncoder(4, 8)(torch.randn(3, 4))


def test_final_state_actually_depends_on_the_sequence():
    # a per-unit constant target makes every sequence relax to the same
    # attractor; guard the property that made the model trainable
    torch.manual_seed(0)
    enc = LTCEncoder(1, 32, tau_init=8.0)
    x = torch.randn(16, 40, 1)
    dt = torch.rand(16, 40) * 0.33 + 0.02
    with torch.no_grad():
        final = enc(x, dt)
    assert float(final.var(dim=0).mean()) > 1e-4


def test_conductance_and_target_have_independent_effects():
    torch.manual_seed(0)
    cell = LTCCell(3, 8)
    x, state = torch.randn(2, 3), cell.init_state(2)
    before = cell(x, state, 1.0)
    with torch.no_grad():
        cell.target_in.weight.mul_(-1.0)
    assert not torch.allclose(before, cell(x, state, 1.0))


def test_negative_gate_bias_lengthens_initial_memory():
    torch.manual_seed(0)
    long_mem = LTCCell(2, 16, tau_init=8.0, gate_bias_init=-4.0)
    short_mem = LTCCell(2, 16, tau_init=8.0, gate_bias_init=2.0)
    x = torch.zeros(1, 2)
    s_long = torch.ones(1, 16)
    s_short = torch.ones(1, 16)
    for _ in range(20):
        s_long = long_mem(x, s_long, 0.2)
        s_short = short_mem(x, s_short, 0.2)
    assert s_long.abs().mean() > s_short.abs().mean()
