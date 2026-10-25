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

"""Cyclic stage order, odd scan remainders and dynamic-temperature gradients."""
import jax
import jax.numpy as jnp
import pytest
from jax_md import simulate
import nhc_test_util as reference
from nhc_test_util import close

from nhc_test_util import DTYPES


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('length,subdivisions,order', [
  (1, 1, 1), (5, 1, 3), (6, 1, 5), (5, 1, 7),
  (2, 2, 3), (5, 2, 7), (5, 4, 1), (6, 4, 5),
  (4, 2, 7), (4, 16, 1), (8, 2, 7), (8, 16, 1),
  (9, 2, 7), (9, 16, 1)])
def test_stage_sequence(dtype, length, subdivisions, order):
  dt, kT = dtype(.01), dtype(1.2)
  factory = simulate.nose_hoover_chain(dt, length, subdivisions, order, dtype(1))
  old = reference.reference_factory(dt, length, subdivisions, order, dtype(1))
  p = jnp.arange(12, dtype=dtype).reshape(4, 3)*dtype(.1)
  chain = factory.initialize(12, jnp.sum(p*p)/2, dtype(1)).set(
    position=jnp.arange(length, dtype=dtype)*dtype(.01),
    momentum=jnp.arange(length, dtype=dtype)*dtype(.02)+dtype(.03))

  def run(fn, target):
    c = factory.update_mass(chain, target)
    return fn(p, c, target)

  close(jax.jit(lambda t: run(factory.half_step, t))(kT),
        jax.jit(lambda t: run(old.half_step, t))(kT), dtype)
  if length == 5 and order in (3, 7):
    targets = jnp.stack([kT, dtype(1.3)])
    close(jax.jit(jax.vmap(lambda t: run(factory.half_step, t)))(targets),
          jax.jit(jax.vmap(lambda t: run(old.half_step, t)))(targets), dtype)
    objective = lambda fn, t: jnp.sum(run(fn, t)[0]**2)
    close(jax.jit(jax.grad(lambda t: objective(factory.half_step, t)))(kT),
          jax.jit(jax.grad(lambda t: objective(old.half_step, t)))(kT), dtype)
