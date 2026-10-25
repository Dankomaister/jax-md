# Copyright 2019 Google LLC
# Portions of the reference recurrence and mass updates are adapted from
# jax_md/simulate.py.
#
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

"""Independent original NHC recurrence and shared regression-test utilities.

The sequential array updates below follow the original JAX-MD recurrence.
Neither the reference factory nor its initialization/mass updates call the
optimized production nose_hoover_chain helper. Only public state types and
Suzuki-Yoshida weights are shared. Tests can be collected independently.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

DTYPES = [jnp.float32] + ([jnp.float64] if jax.config.x64_enabled else [])


def reference_substep(delta, momentum, chain, kT):
  """Original sequential array recurrence, independent of scan lowering."""
  xi, p, q, tau, ke, dof = simulate.dataclasses.astuple(chain)
  d2 = delta / jnp.float32(2)
  d4, d8 = d2 / jnp.float32(2), d2 / jnp.float32(4)
  last = p.shape[0] - 1
  g = p[last - 1] ** jnp.float32(2) / q[last - 1] - kT
  p = p.at[last].add(d4 * g)
  for m in range(last - 1, 0, -1):
    g = p[m - 1] ** 2 / q[m - 1] - kT
    scale = jnp.exp(-d8 * p[m + 1] / q[m + 1])
    p = p.at[m].set(scale * (scale * p[m] + d4 * g))
  # Baseline length 1 clamps index 1; it has no special one-chain algorithm.
  neighbor = min(1, last)
  g = jnp.float32(2) * ke - dof * kT
  scale = jnp.exp(-d8 * p[neighbor] / q[neighbor])
  p = p.at[0].set(scale * (scale * p[0] + d4 * g))
  scale = jnp.exp(-d2 * p[0] / q[0])
  ke = ke * scale ** jnp.float32(2)
  momentum = jax.tree.map(lambda x: x * scale, momentum)
  xi = xi + d2 * p / q
  g = jnp.float32(2) * ke - dof * kT
  for m in range(last):
    scale = jnp.exp(-d8 * p[m + 1] / q[m + 1])
    update = scale * (scale * p[m] + d4 * g)
    p = p.at[m].set(update)
    g = update ** 2 / q[m] - kT
  p = p.at[last].add(d4 * g)
  return momentum, simulate.NoseHooverChain(xi, p, q, tau, ke, dof), kT


def reference_factory(dt, chain_length, chain_steps, sy_steps, tau):
  # Keep initialization and mass updates independent of the optimized helper.
  def initialize(degrees_of_freedom, KE, kT):
    xi = jnp.zeros(chain_length, KE.dtype)
    p_xi = jnp.zeros(chain_length, KE.dtype)
    Q = kT * tau ** jnp.float32(2) * jnp.ones(chain_length, dtype=jnp.float32)
    Q = Q.at[0].multiply(degrees_of_freedom)
    return simulate.NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

  def update_mass(state, kT):
    Q = kT * state.tau ** jnp.float32(2) * jnp.ones(chain_length, dtype=jnp.float32)
    Q = Q.at[0].multiply(state.degrees_of_freedom)
    return state.set(mass=Q)

  def half_step(p, chain, kT):
    if chain_steps == 1 and sy_steps == 1:
      p, chain, _ = reference_substep(dt, p, chain, kT)
      return p, chain
    weights = jnp.array(simulate.SUZUKI_YOSHIDA_WEIGHTS[sy_steps])
    for _ in range(chain_steps):
      for i in range(sy_steps):
        delta = jnp.float32((dt / chain_steps) * weights[i])
        p, chain, _ = reference_substep(delta, p, chain, kT)
    return p, chain

  return simulate.NoseHooverChainFns(initialize, half_step, update_mass)


def all_close(a, b, atol=2e-6, rtol=2e-5):
  assert jax.tree.structure(a) == jax.tree.structure(b)
  xs, ys = jax.tree.leaves(a), jax.tree.leaves(b)
  assert len(xs) == len(ys)
  for x, y in zip(xs, ys):
    assert x.shape == y.shape and x.dtype == y.dtype
    assert np.isfinite(np.asarray(x)).all()
    assert np.isfinite(np.asarray(y)).all()
    np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)


def close(a, b, dtype):
  """Original dtype-specific single-step tolerances."""
  atol, rtol = (2e-10, 2e-9) if dtype == jnp.float64 else (2e-6, 2e-5)
  all_close(a, b, atol=atol, rtol=rtol)


def assert_state_close(actual, expected, dtype, steps=1):
  """Original NPT transformation tolerances, including multi-step float32."""
  atol, rtol = (
    (2e-10, 2e-9)
    if dtype == jnp.float64
    else (2e-6, 2e-5) if steps == 1 else (1e-5, 1e-4)
  )
  all_close(actual, expected, atol=atol, rtol=rtol)


def outer(step, n):
  return jax.jit(lambda state: jax.lax.fori_loop(0, n, lambda _, s: step(s), state))
