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

"""NPT dynamic-temperature compatibility with an independent NHC recurrence."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import quantity, simulate, space
import nhc_test_util as reference


from nhc_test_util import DTYPES, assert_state_close


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('dimension', [2, 3])
def test_npt_dynamic_temperature_and_derivatives(dtype, dimension):
  """Both real NPT chain consumers retain states, batching and derivatives."""
  box = jnp.eye(dimension, dtype=dtype) * dtype(4)
  _, shift = space.periodic_general(
    box, fractional_coordinates=False, wrapped=False
  )

  def energy(r, box, perturbation=1.0, **kwargs):
    return jnp.sum(r * r) * perturbation**2 / dtype(2)

  nhc = dict(chain_length=4, chain_steps=4, sy_steps=7, tau=dtype(1))
  kwargs = dict(
    dt=dtype(.001),
    pressure=dtype(.1),
    kT=dtype(1),
    thermostat_kwargs=nhc,
    barostat_kwargs=nhc,
  )
  init, step = simulate.npt_nose_hoover(energy, shift, **kwargs)
  with patch.object(simulate, 'nose_hoover_chain', reference.reference_factory):
    _, old_step = simulate.npt_nose_hoover(energy, shift, **kwargs)

  r = jnp.arange(4 * dimension, dtype=dtype).reshape(4, dimension) * dtype(.03)
  state = init(
    jax.random.PRNGKey(21),
    r,
    box,
    mass=jnp.linspace(dtype(.8), dtype(1.5), 4),
    momenta=r * dtype(3),
  )
  p_box = dtype(.01)
  state = state.set(
    box_momentum=p_box,
    thermostat=state.thermostat.set(
      position=jnp.arange(4, dtype=dtype) * dtype(.01) + dtype(.02),
      momentum=jnp.arange(4, dtype=dtype) * dtype(.02) + dtype(.03),
    ),
    barostat=state.barostat.set(
      position=jnp.arange(4, dtype=dtype) * dtype(.01) + dtype(.02),
      momentum=jnp.arange(4, dtype=dtype) * dtype(.02) + dtype(.03),
      kinetic_energy=quantity.kinetic_energy(
        momentum=p_box, mass=state.box_mass
      ),
    ),
  )
  assert state.box_momentum.shape == ()
  for n in [1, 50]:
    assert_state_close(
      reference.outer(step, n)(state),
      reference.outer(old_step, n)(state),
      dtype,
      n,
    )

  targets = jnp.array([1.1, 1.3], dtype=dtype)

  def batched(fn):
    return jax.jit(jax.vmap(lambda target: fn(state, kT=target)))(targets)

  assert_state_close(batched(step), batched(old_step), dtype)

  def observable(fn, target):
    s = fn(state, kT=target)
    return (
      jnp.sum(s.momentum**2)
      + s.box_momentum**2
      + dtype(.001)
      * (jnp.sum(s.thermostat.position**2) + jnp.sum(s.barostat.position**2))
    )

  target = dtype(1.2)
  assert_state_close(
    jax.jit(jax.grad(lambda t: observable(step, t)))(target),
    jax.jit(jax.grad(lambda t: observable(old_step, t)))(target),
    dtype,
  )
  assert_state_close(
    jax.jit(
      lambda t: jax.jvp(lambda x: observable(step, x), (t,), (dtype(1),))
    )(target),
    jax.jit(
      lambda t: jax.jvp(lambda x: observable(old_step, x), (t,), (dtype(1),))
    )(target),
    dtype,
  )
