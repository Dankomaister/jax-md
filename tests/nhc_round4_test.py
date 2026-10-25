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

"""Extended scalar chain/fallback and cyclic bounded-stage regression coverage."""
from unittest.mock import patch
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax_md import simulate,space,quantity
import nhc_test_util as reference

from nhc_test_util import DTYPES

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('length',[6,7,8,9])
@pytest.mark.parametrize('subdivisions',[1,2,4])
@pytest.mark.parametrize('order',[1,3,5,7])
def test_extended_scalar_chain(dtype,length,subdivisions,order):
  dt,kT=dtype(.01),dtype(1)
  p=jnp.arange(12,dtype=dtype).reshape(4,3)*dtype(.1)
  factory=simulate.nose_hoover_chain(dt,length,subdivisions,order,dtype(1))
  old=reference.reference_factory(dt,length,subdivisions,order,dtype(1))
  chain=factory.initialize(12,jnp.sum(p*p)/2,kT).set(
    position=jnp.arange(length,dtype=dtype)*dtype(.01)+dtype(.02),
    momentum=jnp.arange(length,dtype=dtype)*dtype(.02)+dtype(.03))
  tol=dict(atol=2e-10,rtol=2e-9) if dtype==jnp.float64 else dict(atol=2e-6,rtol=2e-5)
  single=simulate.nose_hoover_chain(dt,length,1,1,dtype(1)).half_step
  reference.all_close((*jax.jit(single)(p,chain,kT),kT),
                      jax.jit(reference.reference_substep)(dt,p,chain,kT),**tol)
  reference.all_close(jax.jit(factory.half_step)(p,chain,kT),
                      jax.jit(old.half_step)(p,chain,kT),**tol)
  if subdivisions==2 and order==7:
    def objective(fn,q):return jnp.sum(fn(p,chain.set(momentum=q),kT)[0]**2)
    reference.all_close(jax.jit(jax.grad(lambda q:objective(factory.half_step,q)))(chain.momentum),
                        jax.jit(jax.grad(lambda q:objective(old.half_step,q)))(chain.momentum),**tol)
    reference.all_close(jax.jit(lambda q:jax.jvp(lambda z:objective(factory.half_step,z),(q,),(jnp.ones_like(q),)))(chain.momentum),
                        jax.jit(lambda q:jax.jvp(lambda z:objective(old.half_step,z),(q,),(jnp.ones_like(q),)))(chain.momentum),**tol)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('length,subdivisions,order',[
  (1,1,1),(4,1,1),(4,4,7),(5,2,3),(5,2,7),(6,1,5),(8,4,1),(9,1,3)])
def test_cyclic_stage_and_dynamic_temperature(dtype,length,subdivisions,order):
  dt,kT=dtype(.01),dtype(1.2)
  factory=simulate.nose_hoover_chain(dt,length,subdivisions,order,dtype(1))
  old=reference.reference_factory(dt,length,subdivisions,order,dtype(1))
  p=jnp.arange(12,dtype=dtype).reshape(4,3)*dtype(.1)
  chain=factory.initialize(12,jnp.sum(p*p)/2,dtype(1)).set(
    position=jnp.arange(length,dtype=dtype)*dtype(.01)+dtype(.02),
    momentum=jnp.arange(length,dtype=dtype)*dtype(.02)+dtype(.03))
  def run(fn,t):return fn(p,factory.update_mass(chain,t),t)
  tol=dict(atol=2e-10,rtol=2e-9) if dtype==jnp.float64 else dict(atol=2e-6,rtol=2e-5)
  targets=jnp.stack([kT,dtype(1.3)])
  reference.all_close(jax.jit(jax.vmap(lambda t:run(factory.half_step,t)))(targets),
                      jax.jit(jax.vmap(lambda t:run(old.half_step,t)))(targets),**tol)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('length',[8,9])
def test_npt_extended_chain(dtype,length):
  box=jnp.eye(3,dtype=dtype)*4
  _,shift=space.periodic_general(box)
  def energy(r,box,perturbation=1.,**kwargs):
    return jnp.sum(space.transform(box,r)**2)*perturbation**2/dtype(2)
  nhc=dict(chain_length=length,chain_steps=2,sy_steps=7,tau=dtype(1))
  kwargs=dict(dt=dtype(.001),pressure=dtype(.1),kT=dtype(1),
              thermostat_kwargs=nhc,barostat_kwargs=nhc)
  init,step=simulate.npt_nose_hoover(energy,shift,**kwargs)
  with patch.object(simulate,'nose_hoover_chain',reference.reference_factory):
    _,old=simulate.npt_nose_hoover(energy,shift,**kwargs)
  r=jnp.arange(18,dtype=dtype).reshape(6,3)*dtype(.01)+dtype(.1)
  s=init(jax.random.PRNGKey(21),r,box,mass=jnp.linspace(dtype(.8),dtype(1.5),6))
  s=s.set(box_momentum=dtype(.01),
          thermostat=s.thermostat.set(position=jnp.ones(length,dtype)*dtype(.02),
                                     momentum=jnp.ones(length,dtype)*dtype(.03)),
          barostat=s.barostat.set(position=jnp.ones(length,dtype)*dtype(.02),
                                 momentum=jnp.ones(length,dtype)*dtype(.03),
                                 kinetic_energy=quantity.kinetic_energy(
                                   momentum=dtype(.01),mass=s.box_mass)))
  for n in [1,50,1000]:
    tol=dict(atol=2e-10,rtol=2e-9) if dtype==jnp.float64 else dict(atol=5e-5,rtol=5e-4)
    reference.all_close(reference.outer(step,n)(s),reference.outer(old,n)(s),**tol)

  reference.all_close(
    jax.jit(lambda state: step(state,kT=dtype(1.2)))(s),
    jax.jit(lambda state: old(state,kT=dtype(1.2)))(s),
    atol=2e-10 if dtype==jnp.float64 else 2e-6,
    rtol=2e-9 if dtype==jnp.float64 else 2e-5)
