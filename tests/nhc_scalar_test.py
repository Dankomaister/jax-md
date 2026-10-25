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

"""Scalar-chain compatibility with the independent original recurrence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import simulate
import nhc_test_util as reference

from nhc_test_util import DTYPES, close


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('kind', ['scalar', 'array', 'pytree'])
def test_scalar_index_recurrence(dtype, length, kind):
  p = dtype(.7) if kind == 'scalar' else jnp.arange(8, dtype=dtype).reshape(4, 2)*dtype(.1)
  if kind == 'pytree':
    p = {'a': p, 'b': (jnp.ones((2,), dtype)*dtype(.3),)}
  ke = sum(jnp.sum(x*x)/2 for x in jax.tree.leaves(p))
  dof = sum(x.size for x in jax.tree.leaves(p))
  dt, kT = dtype(.01), dtype(1)
  factory = simulate.nose_hoover_chain(dt, length, 2, 7, dtype(1))
  chain = factory.initialize(dof, ke, kT).set(
    position=jnp.arange(length, dtype=dtype)*dtype(.01),
    momentum=jnp.arange(length, dtype=dtype)*dtype(.02)+dtype(.03))
  single = simulate.nose_hoover_chain(dt, length, 1, 1, dtype(1)).half_step
  close((*jax.jit(single)(p, chain, kT), kT),
        jax.jit(reference.reference_substep)(dt, p, chain, kT), dtype)
  ref_half = reference.reference_factory(dt, length, 2, 7, dtype(1)).half_step
  close(jax.jit(factory.half_step)(p, chain, kT), jax.jit(ref_half)(p, chain, kT), dtype)
  if length in (1, 5) and kind == 'scalar':
    def objective(fn, momentum):
      # Include KE's dependence on the scalar barostat momentum.
      c = chain.set(kinetic_energy=momentum*momentum/2)
      return fn(momentum, c, kT)[0]
    close(jax.jit(jax.grad(lambda x: objective(factory.half_step, x)))(p),
          jax.jit(jax.grad(lambda x: objective(ref_half, x)))(p), dtype)
    close(jax.jit(lambda x: jax.jvp(lambda y: objective(factory.half_step, y),
                                   (x,), (jnp.ones_like(x),)))(p),
          jax.jit(lambda x: jax.jvp(lambda y: objective(ref_half, y),
                                   (x,), (jnp.ones_like(x),)))(p), dtype)
    if length == 1:
      # The legacy clipped neighbor read must also preserve its gradient.
      def chain_objective(fn, chain_momentum):
        return fn(p, chain.set(momentum=chain_momentum), kT)[0]
      close(jax.jit(jax.grad(lambda x: chain_objective(factory.half_step, x)))(chain.momentum),
            jax.jit(jax.grad(lambda x: chain_objective(ref_half, x)))(chain.momentum), dtype)


@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_short_chain_has_no_intermediate_scatter(length):
  """Compiler diagnostic only: this does not establish physics correctness."""
  factory = simulate.nose_hoover_chain(jnp.float32(.01), length, 1, 1, jnp.float32(1))
  p = jnp.ones((4, 3), jnp.float32)
  chain = factory.initialize(12, jnp.float32(6), jnp.float32(1))
  text = str(jax.make_jaxpr(factory.half_step)(p, chain, jnp.float32(1)))
  assert ('scatter' in text) == (length > 8)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kind', ['zero', 'near_zero', 'large_finite', 'overflow_control'])
def test_scalar_chain_numerical_limits(dtype, kind):
  amplitude = {'zero': 0., 'near_zero': 1e-15, 'large_finite': 1e6,
               'overflow_control': 1e15}[kind]
  p = jnp.ones((4, 3), dtype)*dtype(amplitude)
  ke = jnp.sum(p*p)/2
  kT = dtype(1) if amplitude <= 1 else dtype(amplitude*amplitude)
  factory = simulate.nose_hoover_chain(dtype(.01), 5, 2, 7, dtype(1))
  chain = factory.initialize(12, ke, kT).set(momentum=jnp.ones(5, dtype)*dtype(.03))
  actual = jax.jit(factory.half_step)(p, chain, kT)
  expected = jax.jit(reference.reference_factory(dtype(.01), 5, 2, 7, dtype(1)).half_step)(p, chain, kT)
  if kind == 'overflow_control' and dtype == jnp.float32:
    assert not all(np.isfinite(x).all() for x in jax.tree.leaves(expected))
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
      np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
      np.testing.assert_array_equal(np.isinf(x), np.isinf(y))
  else:
    close(actual, expected, dtype)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('subdivisions,order', [(1, 1), (2, 7)])
def test_length_one_nonzero_position(dtype, subdivisions, order):
  """Preserve the legacy clipped recurrence with a nonzero sole xi."""
  from unittest.mock import patch
  from jax_md import space

  dt, kT = dtype(.01), dtype(1)
  p = jnp.arange(8, dtype=dtype).reshape(4, 2)*dtype(.1)
  factory = simulate.nose_hoover_chain(dt, 1, subdivisions, order, dtype(1))
  old = reference.reference_factory(dt, 1, subdivisions, order, dtype(1))
  chain = factory.initialize(8, jnp.sum(p*p)/2, kT).set(
    position=jnp.array([.02], dtype), momentum=jnp.array([.03], dtype))
  close(jax.jit(factory.half_step)(p, chain, kT),
        jax.jit(old.half_step)(p, chain, kT), dtype)

  _, shift = space.free()
  energy = lambda r, **kwargs: jnp.sum(r*r)/dtype(2)
  kwargs = dict(dt=dt, kT=kT, tau=dtype(1), chain_length=1,
                chain_steps=subdivisions, sy_steps=order)
  init, step = simulate.nvt_nose_hoover(energy, shift, **kwargs)
  with patch.object(simulate, 'nose_hoover_chain', reference.reference_factory):
    _, ref_step = simulate.nvt_nose_hoover(energy, shift, **kwargs)
  r = jnp.arange(8, dtype=dtype).reshape(4, 2)*dtype(.03)
  state = init(jax.random.PRNGKey(7), r, mass=dtype(1), momenta=p)
  state = state.set(chain=chain)
  for n in [1, 50, 1000]:
    close(reference.outer(step, n)(state),
          reference.outer(ref_step, n)(state), dtype)
