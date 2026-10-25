# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent short-chain recurrence and transformation regression tests."""
from unittest.mock import patch
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax_md import simulate, space, quantity

from nhc_test_util import (
  DTYPES, all_close, outer, reference_factory, reference_substep,
)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('chain_steps', [1, 2, 4])
@pytest.mark.parametrize('sy_steps', [1, 3, 5, 7])
def test_chain_recurrence(dtype, length, chain_steps, sy_steps):
  dt, kT = dtype(.01), dtype(1)
  p = jnp.arange(12, dtype=dtype).reshape(4, 3) * dtype(.1)
  factory = simulate.nose_hoover_chain(dt, length, chain_steps, sy_steps, dtype(1))
  chain = factory.initialize(12, jnp.sum(p*p)/2, kT).set(
    position=jnp.arange(length, dtype=dtype)*dtype(.01),
    momentum=jnp.arange(length, dtype=dtype)*dtype(.02)+dtype(.03))
  single = simulate.nose_hoover_chain(dt, length, 1, 1, dtype(1)).half_step
  tol = dict(atol=2e-10, rtol=2e-9) if dtype == jnp.float64 else {}
  all_close((*jax.jit(single)(p, chain, kT), kT),
            jax.jit(reference_substep)(dt, p, chain, kT), **tol)
  all_close(jax.jit(factory.half_step)(p, chain, kT),
            jax.jit(reference_factory(dt, length, chain_steps, sy_steps, dtype(1)).half_step)(p, chain, kT), **tol)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('sy_steps', [3, 7])
@pytest.mark.parametrize('mass_kind', ['scalar', 'vector', 'column'])
def test_full_state_and_transformations(dtype, sy_steps, mass_kind):
  _, shift = space.free()
  energy = lambda r, **kwargs: jnp.sum(r*r)/2
  init, step = simulate.nvt_nose_hoover(energy, shift, .01, dtype(1), tau=1, sy_steps=sy_steps)
  with patch.object(simulate, 'nose_hoover_chain', reference_factory):
    _, reference = simulate.nvt_nose_hoover(energy, shift, .01, dtype(1), tau=1, sy_steps=sy_steps)
  r = jnp.arange(12, dtype=dtype).reshape(4, 3)*dtype(.03)
  mass = dtype(1.3) if mass_kind == 'scalar' else jnp.linspace(dtype(.8), dtype(1.5), 4)
  if mass_kind == 'column':
    mass = mass[:, None]
  state = init(jax.random.PRNGKey(7), r, mass=mass)
  for n in [1, 50]:
    all_close(outer(step, n)(state), outer(reference, n)(state), atol=1e-5, rtol=1e-4)
  all_close(jax.jit(lambda s: step(s, kT=dtype(1.2)))(state),
            jax.jit(lambda s: reference(s, kT=dtype(1.2)))(state))
  batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
  all_close(jax.jit(jax.vmap(step))(batch), jax.jit(jax.vmap(reference))(batch))
  objective = lambda fn, p: jnp.sum(fn(state.set(momentum=p)).momentum**2)
  all_close(jax.jit(jax.grad(lambda p: objective(step, p)))(state.momentum),
            jax.jit(jax.grad(lambda p: objective(reference, p)))(state.momentum))


@pytest.mark.parametrize('sy_steps', [3, 7])
def test_harmonic_invariant_long_run(sy_steps):
  _, shift = space.free()
  energy = lambda r, **kwargs: jnp.sum(r*r)/2
  kwargs = dict(dt=.01, kT=jnp.float32(1), tau=1, sy_steps=sy_steps)
  init, step = simulate.nvt_nose_hoover(energy, shift, **kwargs)
  with patch.object(simulate, 'nose_hoover_chain', reference_factory):
    _, reference = simulate.nvt_nose_hoover(energy, shift, **kwargs)
  r = jax.random.normal(jax.random.PRNGKey(13), (16, 3), dtype=jnp.float32)
  state = init(jax.random.PRNGKey(17), r, mass=jnp.linspace(.8, 1.5, 16, dtype=jnp.float32))
  initial = simulate.nvt_nose_hoover_invariant(energy, state, kwargs['kT'])
  # Baseline float32 accumulated drift is measured rather than silently tightened.
  for n in [1000, 10000]:
    result = outer(step, n)(state)
    ref = outer(reference, n)(state)
    drift = abs(float(simulate.nvt_nose_hoover_invariant(energy, result, kwargs['kT']) - initial))
    baseline_drift = abs(float(simulate.nvt_nose_hoover_invariant(energy, ref, kwargs['kT']) - initial))
    assert drift <= baseline_drift + 5e-4*abs(float(initial))
    assert drift/abs(float(initial)) < .02
    assert .5 < float(quantity.temperature(momentum=result.momentum, mass=result.mass)) < 1.5


@pytest.mark.parametrize('sy_steps', [1, 3, 5, 7])
@pytest.mark.parametrize('length', [3, 5])
def test_npt_chain_consumer(sy_steps, length):
  box = jnp.eye(3, dtype=jnp.float32) * 4
  _, shift = space.periodic_general(box)
  energy = lambda r, box, **kwargs: jnp.sum(space.transform(box, r)**2)/2
  nhc = dict(chain_length=length, chain_steps=2, sy_steps=sy_steps, tau=.1)
  kwargs = dict(dt=.001, pressure=jnp.float32(.1), kT=jnp.float32(1),
                barostat_kwargs=nhc, thermostat_kwargs=nhc)
  init, step = simulate.npt_nose_hoover(energy, shift, **kwargs)
  with patch.object(simulate, 'nose_hoover_chain', reference_factory):
    _, reference = simulate.npt_nose_hoover(energy, shift, **kwargs)
  r = jnp.arange(18, dtype=jnp.float32).reshape(6, 3) * .01 + .1
  state = init(jax.random.PRNGKey(21), r, box, mass=jnp.linspace(.8, 1.5, 6, dtype=jnp.float32))
  for n in [1, 50]:
    all_close(outer(step, n)(state), outer(reference, n)(state), atol=1e-5, rtol=1e-4)
