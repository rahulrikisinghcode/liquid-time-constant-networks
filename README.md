# liquid-time-constant-networks

Liquid time-constant (LTC) recurrent networks in PyTorch, plus the MFCC front end
they were used with in the AudioFacialMatrix audio experiments.

An LTC unit is a continuous-time state whose time constant is a function of its
input. A gated RNN decides *whether* to update its state; an LTC decides *how
fast* the state evolves. That distinction is the whole library, and it buys two
concrete things: sequences sampled at uneven intervals can be consumed directly
without resampling, and different units can settle on different timescales in the
same layer.

Implements the formulation in R. Hasani, M. Lechner, A. Amini, D. Rus and
R. Grosu, "Liquid Time-constant Networks," AAAI 2021, vol. 35, no. 9,
pp. 7657-7666. Defaults follow the LTNN configuration in Section IV-B of
"AudioFacialMatrix: Dataset for Voice and Face AI" (Singh and Singh): a 128-unit
recurrent layer into a fully connected head.

## The dynamics

```
dx/dt = -[1/tau + f(x, I)] * x  +  f(x, I) * g(x, I)

f(x, I) = sigmoid(W_f I + U_f x + b_f)     conductance, in (0, 1)
g(x, I) = W_g I + U_g x + b_g              signed target
```

`f` sets how fast the state moves and `g` sets where toward. The bracketed decay
rate depends on the input, which is what makes the time constant liquid, and it is
bounded per unit:

```
1 / (1/tau + 1)  <=  tau_effective  <=  tau
```

`LTCCell.effective_tau_bounds()` returns that interval, and
`LTCClassifier.timescale_report()` summarises it after training.

Integration uses the fused semi-implicit Euler step

```
x_{k+1} = (x_k + dt * f * g) / (1 + dt * (1/tau + f))
```

which is unconditionally stable for positive `tau` and `f`. Explicit Euler is not,
and it diverges at exactly the stiff input scales an LTC is supposed to handle well.
`tests/test_cell.py::test_solver_is_stable_on_large_inputs` pins this down.

## Two implementation notes that matter

Both of these are the difference between a model that trains and one that does not,
and neither is obvious from the paper.

**The target has to depend on the input.** Written with a per-unit constant reversal
potential `A` in place of `g`, the input can only modulate the *rate*, so every
sequence relaxes to the same attractor and the batch variance of the final state
collapses to around 1e-6. On the benchmark below that model sits at chance for
every epoch. In the original per-synapse formulation the reversal potentials are
summed against the synaptic activations, so the target is already input-dependent;
`g` recovers that in the dense form.
Guarded by `test_final_state_actually_depends_on_the_sequence`.

**The conductance bias should start negative.** `f` is a sigmoid, so at `b_f = 0` it
starts near 0.5 and puts a floor of 0.5 on the decay rate no matter how large `tau`
is. The memory horizon is then under two time units whatever you asked for.
`gate_bias_init = -2.0` starts `f` near zero, so units begin in the long-memory
regime and training shortens the ones that need to be short. Set `tau_init` on the
order of the sequence duration rather than leaving it at 1.0.

## Install

```bash
git clone https://github.com/rahulrikisinghcode/liquid-time-constant-networks.git
cd liquid-time-constant-networks
pip install -e ".[dev]"
pytest
```

## Use

```python
import torch
from ltc import LTCClassifier

model = LTCClassifier(input_size=39, num_classes=8, hidden_size=128, tau_init=8.0)

x = torch.randn(16, 300, 39)          # (batch, frames, mfcc features)
logits = model(x)                     # (16, 8)
```

Irregularly sampled input, with a per-sample gap at every step:

```python
delta_t = torch.rand(16, 300) * 0.4 + 0.02
logits = model(x, delta_t)
```

Variable-length batches, taking each sequence's own last state rather than the
padded end:

```python
logits = model(x, delta_t, lengths=torch.randint(50, 300, (16,)))
```

After training, check that the units did not all collapse onto one timescale:

```python
model.timescale_report()
# {'tau_min': 7.50, 'tau_max': 8.32, 'tau_median': 7.95, 'tau_spread': 0.82, ...}
```

A spread near zero means the liquid part is not earning its cost and a GRU would do.

## Audio front end

`ltc.features` implements the AFM preprocessing with numpy and scipy only: resample
to 16 kHz, pre-emphasis, 25 ms windows at a 10 ms hop, a 40-band mel filterbank,
13 MFCCs, and first and second order deltas, for 39 features per frame.

```python
from ltc.features import MFCCConfig, mfcc, normalize

features = mfcc(signal, sample_rate)              # (frames, 39)
train, mean, std = normalize(features)
test, _, _ = normalize(test_features, mean, std)  # reuse the training statistics
```

The log floor is relative to each clip's own peak rather than absolute. An absolute
floor clips more mel bins in a quiet recording than a loud one, which breaks the
gain invariance of the shape coefficients and lets recording level leak into the
features. With clips drawn from arbitrary sources at arbitrary levels, that is a
real leak and not a rounding detail.
`test_mfcc_is_invariant_to_recording_level` covers it across a 400x gain range.

`normalize` returns the statistics it used so you can fit on train and apply to
validation and test. Recomputing them per split leaks across the split.

## Benchmark

`examples/irregular_sampling_benchmark.py` builds a task where the elapsed time
between observations carries the signal: sinusoids at four frequencies, sampled at
random uneven intervals. The values alone are close to uninformative; the values
with their spacing are not. The GRU baseline gets `delta_t` as an extra input
channel, which is the usual workaround and the fair comparison.

```
final: LTC 0.743 (2340 params)   GRU 0.702 (3588 params)
learned timescales: min 7.501, median 7.948, max 8.323, spread 0.821
```

Four points of accuracy at two thirds of the parameters, on one small synthetic task
on CPU with a single seed. That is a demonstration of the mechanism, not a benchmark
result, and it should not be cited as one.

## Scope

This is the cell, the encoder, the classifier head and the feature front end. There
is no training loop, no data loader and no checkpointing, because those are project
specific and wrapping them here would only get in the way.

## Citation

```bibtex
@inproceedings{hasani2021liquid,
  author    = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander
               and Rus, Daniela and Grosu, Radu},
  title     = {Liquid Time-constant Networks},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {9},
  pages     = {7657--7666},
  year      = {2021}
}
```

The 128-unit / MFCC configuration this repo defaults to follows Section IV-B of:

```bibtex
@misc{singh_afm,
  author = {Singh, Rahul and Singh, Rita},
  title  = {AudioFacialMatrix: Dataset for Voice and Face AI},
  note   = {Carnegie Mellon University, Language Technologies Institute}
}
```

## License

MIT. See [LICENSE](LICENSE).
