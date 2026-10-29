# Vendored from dysts 0.96 (William Gilpin), Apache-2.0

Files copied verbatim from the PyPI wheel `dysts-0.96-py3-none-any.whl`
(sha256 `17d0106a33e424cd43692a2c7cab72da8784a8726cd277656ac81d9a0ea530a2`):

- `flows.py` (from `dysts/flows.py`): right-hand-side definitions.
- `chaotic_attractors.json` (from `dysts/data/`): published parameters, initial conditions, time scales.
- `LICENSE.md`.

The package itself is not installed. `experiments/accumulating_learner/systems.py` executes `flows.py`
against a stub `base` module (`staticjit` = `staticmethod`) so that only the vector fields are used.
