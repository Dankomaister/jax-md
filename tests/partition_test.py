# Copyright 2019 Google LLC
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

"""Tests for google3.third_party.py.jax_md.mapping."""

import itertools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized

from functools import partial

import numpy as onp

import jax

from jax import random
import jax.numpy as np

from jax import grad

from jax import jit, vmap

from jax_md import dataclasses, smap, partition, space, energy, quantity
from jax_md.util import *
from jax_md import test_util
from jax_md import util

jax.config.parse_flags_with_absl()


PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 10
SPATIAL_DIMENSION = [2, 3]

if util.x64_enabled():
  POSITION_DTYPE = [f32, f64]
else:
  POSITION_DTYPE = [f32]


class CellListTest(test_util.JAXMDTestCase):
  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}'.format(dtype.__name__),
        'dtype': dtype,
      }
      for dtype in POSITION_DTYPE
    )
  )
  def test_cell_list_emplace_2d(self, dtype):
    box_size = np.array([8.65, 8.0], f32)
    cell_size = f32(1.0)

    R = np.array(
      [[0.25, 0.25], [8.5, 1.95], [8.1, 1.5], [3.7, 7.9]], dtype=dtype
    )

    cell_fn = partition.cell_list(box_size, cell_size)

    cell_list = cell_fn.allocate(R)
    self.assertEqual(cell_list.id_buffer.dtype, jnp.int32)

    self.assertAllClose(R[0], cell_list.position_buffer[0, 0, 0])
    self.assertAllClose(R[1], cell_list.position_buffer[1, 8, 1])
    self.assertAllClose(R[2], cell_list.position_buffer[1, 8, 0])
    self.assertAllClose(R[3], cell_list.position_buffer[7, 3, 1])

    self.assertEqual(0, cell_list.id_buffer[0, 0, 0])
    self.assertEqual(1, cell_list.id_buffer[1, 8, 1])
    self.assertEqual(2, cell_list.id_buffer[1, 8, 0])
    self.assertEqual(3, cell_list.id_buffer[7, 3, 1])

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, 2))

    R_out = np.zeros((5, 2), dtype)
    R_out = R_out.at[id_flat].set(R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_cell_list_random_emplace(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = f32(9.0)
    cell_size = f32(1.0)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim))

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = R_out.at[id_flat].set(R_flat)[:-1]

    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_cell_list_random_emplace_rect(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = np.array([9.0, 3.0, 7.25]) if dim == 3 else np.array([9.0, 3.25])
    cell_size = f32(1.0)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim))

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim))
    R_out = R_out.at[id_flat].set(R_flat)[:-1]
    self.assertAllClose(R_out, R)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_cell_list_random_emplace_side_data(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32)
      if dim == 3
      else np.array([9.0, 4.25], f32)
    )
    cell_size = f32(1.23)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    side_data_dim = 2
    side_data = random.normal(key, (PARTICLE_COUNT, side_data_dim), dtype=dtype)

    cell_fn = partition.cell_list(box_size, cell_size)
    cell_list = cell_fn.allocate(R, side_data=side_data)

    id_flat = np.reshape(cell_list.id_buffer, (-1,))
    R_flat = np.reshape(cell_list.position_buffer, (-1, dim))
    R_out = np.zeros((PARTICLE_COUNT + 1, dim), dtype)
    R_out = R_out.at[id_flat].set(R_flat)[:-1]

    side_data_flat = np.reshape(
      cell_list.kwarg_buffers['side_data'], (-1, side_data_dim)
    )
    side_data_out = np.zeros((PARTICLE_COUNT + 1, side_data_dim), dtype)
    side_data_out = side_data_out.at[id_flat].set(side_data_flat)[:-1]

    self.assertAllClose(R_out, R)
    self.assertAllClose(side_data_out, side_data)


class NeighborListTest(test_util.JAXMDTestCase):
  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_neighbor_list_build(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32)
      if dim == 3
      else np.array([9.0, 4.25], f32)
    )
    cutoff = f32(1.23)

    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_fn = partition.neighbor_list(
      displacement, box_size, cutoff, 0.0, 1.1
    )

    idx = neighbor_fn.allocate(R).idx
    R_neigh = R[idx]
    mask = idx < N

    d = vmap(vmap(metric, (None, 0)))
    dR = d(R, R_neigh)

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, f32(0)) * mask
    mask_exact = 1.0 - np.eye(dR_exact.shape[0])
    dR_exact = np.where(dR_exact < cutoff, dR_exact, f32(0)) * mask_exact

    dR = np.sort(dR, axis=1)
    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(dR.shape[0]):
      dR_row = dR[i]
      dR_row = dR_row[dR_row > 0.0]

      dR_exact_row = dR_exact[i]
      dR_exact_row = np.array(dR_exact_row[dR_exact_row > 0.0], dtype)

      self.assertAllClose(dR_row, dR_exact_row)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_neighbor_list_build_sparse(self, dtype, dim):
    key = random.PRNGKey(1)

    box_size = (
      np.array([9.0, 4.0, 7.25], f32)
      if dim == 3
      else np.array([9.0, 4.25], f32)
    )
    cutoff = f32(1.23)

    displacement, _ = space.periodic(box_size)
    metric = space.metric(displacement)

    R = box_size * random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_fn = partition.neighbor_list(
      displacement, box_size, cutoff, 0.0, 1.1, format=partition.Sparse
    )

    nbrs = neighbor_fn.allocate(R)
    mask = partition.neighbor_list_mask(nbrs)

    d = space.map_bond(metric)
    dR = d(R[nbrs.idx[0]], R[nbrs.idx[1]])

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, f32(0)) * mask
    mask_exact = 1.0 - np.eye(dR_exact.shape[0])
    dR_exact = np.where(dR_exact < cutoff, dR_exact, f32(0)) * mask_exact

    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(N):
      dR_row = dR[nbrs.idx[0] == i]
      dR_row = dR_row[dR_row > 0.0]
      dR_row = np.sort(dR_row)

      dR_exact_row = dR_exact[i]
      dR_exact_row = np.array(dR_exact_row[dR_exact_row > 0.0], dtype)

      self.assertAllClose(dR_row, dR_exact_row)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dtype={}_dim={}'.format(dtype.__name__, dim),
        'dtype': dtype,
        'dim': dim,
      }
      for dtype in POSITION_DTYPE
      for dim in SPATIAL_DIMENSION
    )
  )
  def test_neighbor_list_build_time_dependent(self, dtype, dim):
    key = random.PRNGKey(1)

    if dim == 2:
      box_fn = lambda t: np.array([[9.0, t], [0.0, 3.75]], f32)
    elif dim == 3:
      box_fn = lambda t: np.array(
        [[9.0, 0.0, t], [0.0, 4.0, 0.0], [0.0, 0.0, 7.25]]
      )
    min_length = np.min(np.diag(box_fn(0.0)))
    cutoff = f32(1.23)
    # TODO(schsam): Get cell-list working with anisotropic cell sizes.
    cell_size = cutoff / min_length

    displacement, _ = space.periodic_general(box_fn(0.0))
    metric = space.metric(displacement)

    R = random.uniform(key, (PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]
    neighbor_list_fn = partition.neighbor_list(
      metric, 1.0, cutoff, 0.0, 1.1, cell_size=cell_size, t=np.array(0.0)
    )

    idx = neighbor_list_fn.allocate(R, box=box_fn(np.array(0.25))).idx
    R_neigh = R[idx]
    mask = idx < N

    metric = partial(metric, box=box_fn(0.25))
    d = vmap(vmap(metric, (None, 0)))
    dR = d(R, R_neigh)

    d_exact = space.map_product(metric)
    dR_exact = d_exact(R, R)

    dR = np.where(dR < cutoff, dR, 0) * mask
    dR_exact = np.where(dR_exact < cutoff, dR_exact, 0)

    dR = np.sort(dR, axis=1)
    dR_exact = np.sort(dR_exact, axis=1)

    for i in range(dR.shape[0]):
      dR_row = dR[i]
      dR_row = dR_row[dR_row > 0.0]

      dR_exact_row = dR_exact[i]
      dR_exact_row = dR_exact_row[dR_exact_row > 0.0]

      self.assertAllClose(dR_row, dR_exact_row)

  def test_cell_list_overflow(self):
    displacement_fn, shift_fn = space.free()

    box = 100.0
    r_cutoff = 3.0
    dr_threshold = 0.0

    neighbor_list_fn = partition.neighbor_list(
      displacement_fn,
      box=box,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
    )

    # all far from eachother
    R = jnp.array(
      [
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
        [50.0, 50.0],
      ]
    )
    neighbors = neighbor_list_fn.allocate(R)
    self.assertEqual(neighbors.idx.dtype, jnp.int32)

    # two first point are close to eachother
    R = jnp.array(
      [
        [20.0, 20.0],
        [20.0, 20.0],
        [40.0, 40.0],
        [50.0, 50.0],
      ]
    )

    neighbors = neighbors.update(R)
    self.assertTrue(neighbors.did_buffer_overflow)
    self.assertEqual(neighbors.idx.dtype, jnp.int32)

  def test_custom_mask_function(self):
    displacement_fn, shift_fn = space.free()

    box = 1.0
    r_cutoff = 3.0
    dr_threshold = 0.0
    n_particles = 10
    R = jnp.broadcast_to(jnp.zeros(3), (n_particles, 3))

    def acceptable_id_pair(id1, id2):
      """
      Don't allow particles to have an interaction when their id's
      are closer than 3 (eg disabling 1-2 and 1-3 interactions)
      """
      return jnp.abs(id1 - id2) > 3

    def mask_id_based(
      idx: Array, ids: Array, mask_val: int, _acceptable_id_pair: Callable
    ) -> Array:
      """
      _acceptable_id_pair mapped to act upon the neighbor list where:
          - index of particle 1 is in index in the first dimension of array
          - index of particle 2 is given by the value in the array
      """

      @partial(vmap, in_axes=(0, 0, None))
      def acceptable_id_pair(idx, id1, ids):
        id2 = ids.at[idx].get()
        return vmap(_acceptable_id_pair, in_axes=(None, 0))(id1, id2)

      mask = acceptable_id_pair(idx, ids, ids)
      return jnp.where(mask, idx, mask_val)

    ids = jnp.arange(n_particles)  # id is just particle index here.
    mask_val = n_particles
    custom_mask_function = partial(
      mask_id_based,
      ids=ids,
      mask_val=mask_val,
      _acceptable_id_pair=acceptable_id_pair,
    )

    neighbor_list_fn = partition.neighbor_list(
      displacement_fn,
      box=box,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
      custom_mask_function=custom_mask_function,
    )

    neighbors = neighbor_list_fn.allocate(R)
    neighbors = neighbors.update(R)
    """
    Without masking it's 9 neighbors (with mask self) -> 90 neighbors.
    With masking -> 42.
    """
    self.assertEqual(42, (neighbors.idx != mask_val).sum())

  def test_issue191_1(self):
    box_vector = jnp.ones(3) * 3

    r_cut = 0.1
    _positions = jnp.linspace(0.5, 0.7, 20)
    positions = jnp.stack([_positions, _positions, _positions], axis=1)

    displacement, _ = space.periodic_general(box_vector)

    neighbor_fn = partition.neighbor_list(
      displacement, box_vector, r_cut, 0.1 * r_cut, fractional_coordinates=True
    )

    neighbor2_fn = partition.neighbor_list(
      displacement,
      box_vector[0],
      r_cut,
      0.1 * r_cut,
      fractional_coordinates=True,
      disable_cell_list=True,
    )

    nbrs = neighbor_fn.allocate(positions)
    nbrs2 = neighbor2_fn.allocate(positions)
    self.assertAllClose(
      jnp.sort(nbrs.idx, axis=-1), jnp.sort(nbrs2.idx, axis=-1)
    )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_case={i}_mask_self={ms}_format={fmt.name}',
        'r_cut': r_cut,
        'disable_cell_list': dc,
        'capacity_multiplier': cm,
        'mask_self': ms,
        'format': fmt,
      }
      for i, (r_cut, dc, cm) in enumerate(
        [
          (0.12, True, 1.5),
          (0.25, False, 1.5),
          (0.31, False, 1.5),
          (0.31, False, 1.0),
        ]
      )
      for ms in [False, True]
      for fmt in [partition.Dense, partition.Sparse, partition.OrderedSparse]
    )
  )
  def test_issue191_2(
    self, r_cut, disable_cell_list, capacity_multiplier, mask_self, format
  ):
    box = onp.ones(3)
    # box = 1.0
    if format is partition.Dense:
      desired_shape = (20, 19) if mask_self else (20, 20)
      _positions = jnp.ones((20,)) * 0.5
    elif format is partition.Sparse:
      desired_shape = (2, 20 * 19) if mask_self else (2, 20**2)
      _positions = jnp.ones((20,)) * 0.5
    elif format is partition.OrderedSparse:
      desired_shape = (2, 20 * 19 // 2)
      _positions = jnp.ones((20,)) * 0.5
    positions = jnp.stack([_positions, _positions, _positions], axis=1)
    displacement, _ = space.periodic(box)

    neighbor_fn = partition.neighbor_list(
      displacement,
      box,
      r_cut,
      0.1 * r_cut,
      capacity_multiplier=capacity_multiplier,
      disable_cell_list=disable_cell_list,
      mask_self=mask_self,
      format=format,
    )

    nbrs = neighbor_fn.allocate(positions)

    self.assertFalse(nbrs.did_buffer_overflow)
    self.assertEqual(nbrs.idx.shape, desired_shape)

    new_nbrs = nbrs.update(positions + 0.1)
    self.assertFalse(new_nbrs.did_buffer_overflow)
    self.assertEqual(new_nbrs.idx.shape, desired_shape)

  def test_general_unit_cell(self):
    N = 128
    dim = 3
    rho = 0.05

    energy_form = lambda dr, **kwargs: jnp.where(dr < 2.5, 1.0, 0.0)

    def cell(key, a, b, c, alpha, beta, gamma):
      alpha = alpha * jnp.pi / 180
      beta = beta * jnp.pi / 180
      gamma = gamma * jnp.pi / 180
      xx = a
      yy = b * jnp.sin(gamma)
      xy = b * jnp.cos(gamma)
      xz = c * jnp.cos(beta)
      yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
      zz = jnp.sqrt(c**2 - xz**2 - yz**2)
      U = jnp.array([[xx, xy, xz], [0, yy, yz], [0, 0, zz]])
      return U

    key = random.PRNGKey(0)

    L = quantity.box_size_at_number_density(N, rho, dim)

    for _ in range(10):
      key, cell_key, pos_key, L_key, angle_key = random.split(key, 5)
      Lx, Ly, Lz = random.uniform(L_key, (3,), minval=L * 0.5, maxval=L * 3)
      alpha, beta, gamma = random.uniform(
        angle_key, (3,), minval=45.0, maxval=125.0
      )
      cl = cell(cell_key, Lx, Ly, Lz, alpha, beta, gamma)
      if jnp.any(jnp.isnan(cl)):
        continue
      displacement, shift = space.periodic_general(cl)
      metric = space.metric(displacement)

      E_exact = smap.pair(energy_form, metric, reduce_axis=(-1,))
      E = smap.pair_neighbor_list(energy_form, metric, reduce_axis=(-1,))
      neighbor_fn = partition.neighbor_list(
        displacement,
        cl,
        2.5,
        capacity_multiplier=1.0,
        fractional_coordinates=True,
      )

      R = random.uniform(key, (N, dim))
      nbrs = neighbor_fn.allocate(R)
      E_target = E_exact(R)
      self.assertTrue(jnp.any(jnp.abs(E_target) > 0.25))
      self.assertAllClose(E_target, E(R, neighbor=nbrs))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {'testcase_name': f'_factor={int(10 * f)}', 'factor': f}
      for f in [0.5, 2.0]
    )
  )
  def test_general_unit_cell_resize(self, factor):
    N = 128
    dim = 3
    rho = 0.05

    energy_form = lambda dr, **kwargs: jnp.where(dr < 2.5, 1.0, 0.0)

    def cell(key, a, b, c, alpha, beta, gamma):
      alpha = alpha * jnp.pi / 180
      beta = beta * jnp.pi / 180
      gamma = gamma * jnp.pi / 180
      xx = a
      yy = b * jnp.sin(gamma)
      xy = b * jnp.cos(gamma)
      xz = c * jnp.cos(beta)
      yz = (b * c * jnp.cos(alpha) - xy * xz) / yy
      zz = jnp.sqrt(c**2 - xz**2 - yz**2)
      U = jnp.array([[xx, xy, xz], [0, yy, yz], [0, 0, zz]])
      return U

    key = random.PRNGKey(0)

    L = quantity.box_size_at_number_density(N, rho, dim)

    for _ in range(10):
      key, cell_key, pos_key, L_key, angle_key = random.split(key, 5)
      Lx, Ly, Lz = random.uniform(L_key, (3,), minval=L * 0.5, maxval=L * 3)
      alpha, beta, gamma = random.uniform(
        angle_key, (3,), minval=45.0, maxval=125.0
      )
      cl = cell(cell_key, Lx, Ly, Lz, alpha, beta, gamma)
      if jnp.any(jnp.isnan(cl)):
        continue
      displacement, shift = space.periodic_general(cl)
      metric = space.metric(displacement)

      E_exact = smap.pair(energy_form, metric, reduce_axis=(-1,))
      E = smap.pair_neighbor_list(energy_form, metric, reduce_axis=(-1,))
      neighbor_fn = partition.neighbor_list(
        displacement,
        cl,
        2.5,
        capacity_multiplier=1.0,
        fractional_coordinates=True,
      )

      R = random.uniform(key, (N, dim))
      nbrs = neighbor_fn.allocate(R)
      nbrs = nbrs.update(R, box=cl * factor)
      E_target = E_exact(R, box=cl * factor)

      if factor > 1:
        self.assertTrue(jnp.any(jnp.abs(E_target) > 0.25))
        self.assertAllClose(E_target, E(R, neighbor=nbrs, box=cl * factor))
      elif nbrs.cell_list_fn is not None:
        self.assertTrue(jnp.any(nbrs.cell_size_too_small))


_STANDARD_NEIGHBOR_FORMATS = (
  partition.Dense, partition.Sparse, partition.OrderedSparse
)


def _standard_neighbor_pairs(nbrs):
  idx = onp.asarray(nbrs.idx)
  assert idx.dtype == onp.int32
  n = len(nbrs.reference_position)
  assert onp.all((idx >= 0) & (idx <= n))
  if nbrs.format is partition.Dense:
    i, slot = onp.nonzero(idx < n)
    edges = list(zip(i.tolist(), idx[i, slot].tolist()))
  else:
    j, i = idx[:, idx[0] < n]
    assert onp.all(i < n)
    edges = list(zip(i.tolist(), j.tolist()))
  assert len(edges) == len(set(edges))
  return set(edges)


def _standard_neighbor_reference(
  position, box, cutoff, fmt, mask_self=True, custom=False
):
  position = onp.asarray(position, dtype=onp.float64)
  delta = position[None, :, :] - position[:, None, :]
  if box is not None:
    # Independent image enumeration for the mildly skew cells used here.
    trials = onp.array(list(itertools.product((-1, 0, 1), repeat=3)))
    distances = onp.stack([
      onp.sum((delta + t @ box.T) ** 2, axis=-1) for t in trials
    ])
    squared = distances.min(axis=0)
  else:
    squared = onp.sum(delta ** 2, axis=-1)
  i, j = onp.nonzero(squared < cutoff ** 2)
  return {
    (int(a), int(b)) for a, b in zip(i, j)
    if (not mask_self or a != b)
    and (fmt is not partition.OrderedSparse or b < a)
    and (not custom or (a + b) % 2 == 0)
  }


def _assert_neighbors_unchanged(a, b):
  for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
    onp.testing.assert_array_equal(x, y)


class NeighborListReuseRegressionTest(test_util.JAXMDTestCase):
  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    geometry=['free', 'periodic', 'triclinic'],
    fractional=[False, True],
    disable_cells=[False, True],
  )
  def test_reuse_and_actual_rebuild(
    self, fmt, geometry, fractional, disable_cells
  ):
    box = onp.array([[20., .8, .3], [0., 21., .4], [0., 0., 22.]], onp.float32)
    if geometry != 'triclinic':
      box = onp.diag(onp.diag(box))
    f = onp.array([[.01, .1, .1], [.04, .12, .1], [.2, .1, .1],
                  [.23, .13, .11], [.7, .8, .9], [.72, .82, .9]], onp.float32)
    cart = f @ box.T
    if geometry == 'free':
      displacement, _ = space.free()
      # Fractional flag is still a supported cell-list setting for free space;
      # use Cartesian input without that flag in the nonperiodic cases.
      fractional = False
    elif geometry == 'periodic' and not fractional:
      displacement, _ = space.periodic(np.diag(np.asarray(box)))
    else:
      displacement, _ = space.periodic_general(
        np.asarray(box), fractional_coordinates=fractional
      )
    base = np.asarray(f if fractional else cart)
    factory = partition.neighbor_list(
      displacement, box if fractional else onp.diag(box), 2., 1.,
      capacity_multiplier=1.25, disable_cell_list=disable_cells,
      fractional_coordinates=fractional, format=fmt,
    )
    initial = factory.allocate(base, extra_capacity=5)
    assert not initial.did_buffer_overflow
    assert _standard_neighbor_pairs(initial) == _standard_neighbor_reference(cart, None if geometry == 'free' else box,
                                       3., fmt)
    update = jax.jit(factory.update)
    # Two different atoms move, each below the per-atom bound. Their squared
    # displacements must not be added together to trigger a global rebuild.
    delta = onp.zeros_like(cart)
    delta[0, 1] = .4
    delta[-1, 1] = -.4
    if fractional:
      delta = delta @ onp.linalg.inv(box.T)
    small = base + np.asarray(delta)
    reused = update(small, initial)
    _assert_neighbors_unchanged(initial, reused)
    delta[2] = (onp.array([.61, 0., 0.]) @ onp.linalg.inv(box.T)
                if fractional else onp.array([.61, 0., 0.]))
    moved = base + np.asarray(delta)
    rebuilt = update(moved, reused)
    onp.testing.assert_array_equal(rebuilt.reference_position, moved)
    assert not rebuilt.did_buffer_overflow
    cart_moved = onp.asarray(moved) @ box.T if fractional else moved
    assert _standard_neighbor_pairs(rebuilt) == _standard_neighbor_reference(cart_moved, None if geometry == 'free' else box,
                                       3., fmt)
    _assert_neighbors_unchanged(rebuilt, update(moved, rebuilt))

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    metric=[False, True],
    distance=[onp.nextafter(onp.float32(0.5), onp.float32(0)), onp.float32(0.5), onp.nextafter(onp.float32(0.5), onp.float32(1))],
  )
  def test_strict_threshold_and_reference(self, fmt, metric, distance):
    displacement, _ = space.free()
    fn = space.metric(displacement) if metric else displacement
    factory = partition.neighbor_list(fn, 20., 2., 1., format=fmt,
                                      disable_cell_list=True)
    base = np.array([[0., 0., 0.], [1., .1, 0.], [3., 0., 0.]], np.float32)
    initial = factory.allocate(base, extra_capacity=3)
    moved = base.at[0, 0].set(distance)
    new = jax.jit(factory.update)(moved, initial)
    if distance > .5:
      onp.testing.assert_array_equal(new.reference_position, moved)
      # A later displacement is measured from the actual rebuild, not allocation.
      later = moved.at[0, 0].set(.8)
      _assert_neighbors_unchanged(new, jax.jit(factory.update)(later, new))
    else:
      _assert_neighbors_unchanged(initial, new)
      # Cumulative small updates cannot reset the rebuild reference.
      later = moved.at[0, 0].set(.51)
      rebuilt = jax.jit(factory.update)(later, new)
      onp.testing.assert_array_equal(rebuilt.reference_position, later)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    fractional=[False, True],
  )
  def test_periodic_crossing(self, fmt, fractional):
    displacement, _ = space.periodic_general(20., fractional_coordinates=fractional)
    base = np.array([[19.9, 1., 1.], [.5, 1., 1.], [2., 1., 1.]])
    if fractional:
      base = base / 20
    factory = partition.neighbor_list(displacement, 20., 2., 1.,
                                      fractional_coordinates=fractional, format=fmt)
    initial = factory.allocate(base, extra_capacity=3)
    reused = base.at[0, 0].set(.1 / 20 if fractional else .1)
    _assert_neighbors_unchanged(initial, jax.jit(factory.update)(reused, initial))
    moved = reused.at[0, 0].set(.6 / 20 if fractional else .6)
    new = jax.jit(factory.update)(moved, initial)
    onp.testing.assert_array_equal(new.reference_position, moved)
    assert _standard_neighbor_pairs(new) == _standard_neighbor_reference(onp.asarray(moved) * (20 if fractional else 1),
                                   onp.eye(3)*20, 3., fmt)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    mask_self=[False, True],
    custom=[False, True],
  )
  def test_masks_and_zero_threshold(self, fmt, mask_self, custom):
    displacement, _ = space.free()
    def mask(idx):
      center = np.arange(len(idx))[:, None]
      return np.where((idx + center) % 2 == 0, idx, len(idx))
    factory = partition.neighbor_list(
      displacement, 20., 2., 0., disable_cell_list=True, format=fmt,
      mask_self=mask_self, custom_mask_function=mask if custom else None,
    )
    base = np.array([[0., 0., 0.], [1., 0., 0.], [1.5, .3, 0.]])
    initial = factory.allocate(base, extra_capacity=4)
    moved = base.at[0, 0].set(.001)
    new = jax.jit(factory.update)(moved, initial)
    onp.testing.assert_array_equal(new.reference_position, moved)
    assert _standard_neighbor_pairs(new) == _standard_neighbor_reference(moved, None, 2., fmt, mask_self, custom)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    disable_cells=[False, True],
  )
  def test_overflow_and_reallocation(self, fmt, disable_cells):
    displacement, _ = space.free()
    factory = partition.neighbor_list(displacement, 40., 2., 1., format=fmt,
                                      disable_cell_list=disable_cells)
    base = np.array([[5., 5., 5.], [15., 15., 15.], [25., 25., 25.], [35., 35., 35.]])
    initial = factory.allocate(base)
    moved = np.array([[5., 5., 5.], [5.2, 5., 5.], [5.4, 5., 5.], [5.6, 5., 5.]])
    broken = jax.jit(factory.update)(moved, initial)
    assert broken.did_buffer_overflow
    onp.testing.assert_array_equal(broken.reference_position, moved)
    # Overflow flags remain sticky even if the next update reuses the list.
    _assert_neighbors_unchanged(broken, jax.jit(factory.update)(moved, broken))
    recovered = factory.allocate(moved, extra_capacity=4)
    assert not recovered.did_buffer_overflow
    assert _standard_neighbor_pairs(recovered) == _standard_neighbor_reference(moved, None, 3., fmt)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
  )
  def test_changing_box_flags(self, fmt):
    displacement, _ = space.periodic_general(30.)
    base = np.array([[.1, .2, .3], [.12, .2, .3], [.6, .6, .6]])
    factory = partition.neighbor_list(displacement, 30., 2., 1.,
                                      fractional_coordinates=True, format=fmt)
    initial = factory.allocate(base, extra_capacity=4)
    update = jax.jit(factory.update)
    # Preserve the baseline changing-box contract, including error accumulation.
    grown = update(base, initial, box=np.float32(30.1))
    _assert_neighbors_unchanged(initial, grown)
    shrunk = update(base, grown, box=np.float32(10.))
    assert shrunk.cell_size_too_small
    onp.testing.assert_array_equal(shrunk.reference_position, base)
    malformed = np.array([[30., 0., 0.], [1., 30., 0.], [0., 0., 30.]])
    bad = update(base, initial, box=malformed)
    assert bad.malformed_box

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
  )
  def test_paired_lists_and_downstream_mask(self, fmt):
    displacement, _ = space.periodic(20.)
    factory = partition.paired_neighbor_list(displacement, 20., 1.5, 2.5,
                                             dr_threshold=1., format=fmt)
    base = np.array([[1., 1., 1.], [2., 1., 1.], [4., 1., 1.], [8., 8., 8.]])
    initial = factory.allocate(base, extra_capacity=4)
    new = jax.jit(factory.update)(base + .1, initial)
    _assert_neighbors_unchanged(initial, new)
    moved = base.at[2, 0].add(.6)
    rebuilt = jax.jit(factory.update)(moved, new)
    for cutoff, n in [(2.5, rebuilt.short), (3.5, rebuilt.long_range)]:
      onp.testing.assert_array_equal(n.reference_position, moved)
      expected = _standard_neighbor_reference(moved, onp.eye(3)*20, cutoff, fmt)
      assert _standard_neighbor_pairs(n) == expected
      assert int(partition.neighbor_list_mask(n).sum()) == len(expected)
      if fmt is not partition.Dense:
        graph = jax.jit(partition.to_jraph)(n)
        assert int(graph.n_edge[0]) == len(expected)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    other_moves=[False, True],
  )
  def test_nan_does_not_hide_another_atoms_threshold_crossing(
    self, fmt, other_moves
  ):
    displacement, _ = space.free()
    factory = partition.neighbor_list(displacement, 20., 2., 1., format=fmt,
                                      disable_cell_list=True)
    base = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    initial = factory.allocate(base, extra_capacity=4)
    moved = base.at[0, 0].set(np.nan)
    if other_moves:
      moved = moved.at[1, 0].add(.75)
    new = jax.jit(factory.update)(moved, initial)
    if other_moves:
      onp.testing.assert_array_equal(new.reference_position, moved)
    else:
      _assert_neighbors_unchanged(initial, new)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    dim=[2, 3],
  )
  def test_float64_threshold_and_reference(self, fmt, dim):
    # Explicit float64 callers retain the strict threshold in both dimensions,
    # even when the application-wide default uses float32.
    with jax.enable_x64():
      displacement, _ = space.free()
      factory = partition.neighbor_list(displacement, 20., 2., 1., format=fmt,
                                        disable_cell_list=True)
      base = np.zeros((3, dim), np.float64).at[1, 0].set(1.).at[2, 0].set(4.)
      initial = factory.allocate(base, extra_capacity=3)
      update = jax.jit(factory.update)
      _assert_neighbors_unchanged(initial, update(base.at[0, 0].set(.5), initial))
      moved = base.at[0, 0].set(onp.nextafter(onp.float64(.5), onp.float64(1.)))
      new = update(moved, initial)
      onp.testing.assert_array_equal(new.reference_position, moved)
      assert new.reference_position.dtype == np.float64


class NeighborListRebuildRegressionTest(test_util.JAXMDTestCase):
  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    threshold=[0.0, 1.0],
    kind=['scalar', 'vector', 'diagonal', 'upper', 'lower'],
  )
  def test_fractional_box_error_flags(self, fmt, threshold, kind):
    boxes = {
      'scalar': np.float32(30.),
      'vector': np.array([30., 31., 32.]),
      'diagonal': np.diag(np.array([30., 31., 32.])),
      'upper': np.array([[30., .4, .2], [0., 31., .3], [0., 0., 32.]]),
      'lower': np.array([[30., 0., 0.], [.4, 31., 0.], [0., 0., 32.]]),
    }
    box = boxes[kind]
    valid = kind != 'lower'
    assert bool(partition.is_box_valid(box)) == valid
    displacement, _ = space.periodic_general(box)
    factory = partition.neighbor_list(
      displacement, box, 2., threshold, fractional_coordinates=True, format=fmt)
    base = np.array([[.1, .2, .3], [.12, .2, .3], [.6, .6, .6]])
    allocated = factory.allocate(base, extra_capacity=4)
    assert bool(allocated.malformed_box) == (not valid)
    assert not allocated.did_buffer_overflow
    # The explicit allocation override and JIT update must use the same polarity.
    overridden = factory.allocate(base, extra_capacity=4, box=box)
    assert bool(overridden.malformed_box) == (not valid)
    clean = dataclasses.replace(
      allocated, error=partition.PartitionError(np.uint8(0)))
    updated = jax.jit(factory.update)(base + .00001, clean, box=box)
    assert bool(updated.malformed_box) == (not valid)
    assert not updated.did_buffer_overflow
    if valid:
      assert int(updated.error.code) == 0
    else:
      # Errors accumulate until reallocation, including after a valid box update.
      again = jax.jit(factory.update)(base, updated, box=np.float32(32.))
      assert again.malformed_box
      recovered = factory.allocate(base, extra_capacity=4, box=np.float32(32.))
      assert int(recovered.error.code) == 0

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    disable_cells=[False, True],
    fractional=[False, True],
  )
  def test_zero_threshold_recomputes_every_step(
    self, fmt, disable_cells, fractional
  ):
    box = np.float32(24.)
    displacement, _ = space.periodic_general(box, fractional_coordinates=fractional)
    factory = partition.neighbor_list(
      displacement, box, 2., 0., format=fmt,
      disable_cell_list=disable_cells, fractional_coordinates=fractional)
    cart = np.array([[1., 1., 1.], [2.999, 1., 1.], [23.8, 1., 1.],
                      [10., 10., 10.]])
    base = cart / box if fractional else cart
    allocated = factory.allocate(base, extra_capacity=4)
    # A deliberately stale index array proves the update computes connectivity
    # even at identical positions (a displacement predicate would skip it).
    poisoned = dataclasses.replace(allocated, idx=np.full_like(allocated.idx, 4))
    update = jax.jit(factory.update)
    restored = update(base, poisoned)
    assert _standard_neighbor_pairs(restored) == _standard_neighbor_pairs(allocated)
    old = restored
    seen = []
    for step in range(1, 7):
      # Each displacement is much smaller than a normal half-skin threshold;
      # it crosses a strict cutoff and a periodic boundary in alternating steps.
      moved = cart.at[1, 0].add(.003 * (step % 2))
      moved = moved.at[2, 0].set((23.8 + .1 * step) % 24.)
      r = moved / box if fractional else moved
      new = update(r, old)
      onp.testing.assert_array_equal(new.reference_position, r)
      assert int(new.error.code) == 0
      d = onp.asarray(moved, dtype=onp.float64)
      delta = d[:, None] - d[None, :]
      delta -= 24. * onp.round(delta / 24.)
      i, j = onp.nonzero(onp.sum(delta * delta, axis=-1) < 4.)
      expected = {(int(a), int(b)) for a, b in zip(i, j)
                  if a != b and (fmt is not partition.OrderedSparse or b < a)}
      assert _standard_neighbor_pairs(new) == expected
      seen.append(expected)
      old = new
    assert seen[0] != seen[1]

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    disable_cells=[False, True],
  )
  def test_rebuild_fills_last_capacity_slot_and_reports_overflow(
    self, fmt, disable_cells
  ):
    displacement, _ = space.periodic(24.)
    factory = partition.neighbor_list(
      displacement, 24., 2., 0., capacity_multiplier=1.,
      disable_cell_list=disable_cells, format=fmt)
    base = np.array([[1., 1., 1.], [2., 1., 1.], [3.25, 1., 1.],
                      [10., 10., 10.]])
    expected = {(0, 1), (1, 0), (1, 2), (2, 1)}
    if fmt is partition.OrderedSparse:
      expected = {(i, j) for i, j in expected if j < i}
    initial = factory.allocate(base)
    assert initial.max_occupancy == (2 if fmt is partition.Dense else len(expected))
    update = jax.jit(factory.update)
    rebuilt = update(base + .001, initial)
    assert int(rebuilt.error.code) == 0
    assert _standard_neighbor_pairs(rebuilt) == expected
    # The last valid slot cannot be overwritten by discarded candidates.
    if fmt is partition.Dense:
      assert onp.all(onp.asarray(rebuilt.idx)[1] < len(base))
    else:
      assert onp.all(onp.asarray(rebuilt.idx) < len(base))
    crowded = base.at[3].set(np.array([2.2, 1., 1.]))
    overflowed = update(crowded, rebuilt)
    assert overflowed.did_buffer_overflow
    assert update(base, overflowed).did_buffer_overflow
    recovered = factory.allocate(crowded, extra_capacity=4)
    assert int(recovered.error.code) == 0
    assert len(_standard_neighbor_pairs(recovered)) == (5 if fmt is partition.OrderedSparse else 10)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
  )
  def test_single_particle_zero_neighbor_capacity(self, fmt):
    displacement, _ = space.periodic(24.)
    factory = partition.neighbor_list(displacement, 24., 2., 0., format=fmt)
    base = np.array([[1., 1., 1.]])
    initial = factory.allocate(base)
    assert initial.max_occupancy == 0
    moved = base + .001
    rebuilt = jax.jit(factory.update)(moved, initial)
    assert int(rebuilt.error.code) == 0
    assert _standard_neighbor_pairs(rebuilt) == set()
    onp.testing.assert_array_equal(rebuilt.reference_position, moved)

  @parameterized.product(
    dim=[2, 3],
    extra=[0, 9],
  )
  def test_cell_overflow_sorted_occupancies_and_named_buffers(self, dim, extra):
    # Four cells have exactly three particles each. Hashes are interleaved in
    # particle order, so sorting and run boundaries both matter.
    n = 12
    base = onp.full((n, dim), .5, onp.float32)
    base[:, 0] = .25 + 2 * (onp.arange(n) % 4)
    tags = np.arange(n, dtype=np.int32) * 3 + 100
    factory = partition.cell_list(12., 2., buffer_size_multiplier=1.)
    initial = factory.allocate(np.asarray(base), extra_capacity=extra, tag=tags)
    assert initial.cell_capacity == 3 + extra
    assert initial.did_buffer_overflow.dtype == np.bool_
    assert not initial.did_buffer_overflow
    update = jax.jit(lambda r, c: c.update(r, tag=tags))

    def check_contents(cell, r):
      ids = onp.asarray(cell.id_buffer).reshape(-1)
      valid = ids < n
      assert set(ids[valid].tolist()) == set(range(n))
      assert onp.count_nonzero(valid) == n
      positions = onp.asarray(cell.position_buffer).reshape(-1, dim)
      onp.testing.assert_array_equal(positions[valid], r[ids[valid]])
      values = onp.asarray(cell.named_buffer['tag']).reshape(-1)
      onp.testing.assert_array_equal(values[valid], onp.asarray(tags)[ids[valid]])

    check_contents(initial, base)
    crowded = base.copy()
    if extra:
      crowded[:, 0] = .25  # N == capacity, exactly full and still valid.
    else:
      crowded[1, 0] = .25  # One run has capacity+1 particles.
    updated = update(np.asarray(crowded), initial)
    keys = [tuple(x) for x in (onp.trunc(crowded / 2).astype(int) % 6)]
    counts = [keys.count(key) for key in set(keys)]
    assert bool(updated.did_buffer_overflow) == (max(counts) > initial.cell_capacity)
    if not updated.did_buffer_overflow:
      check_contents(updated, crowded)
    returned = update(np.asarray(base), updated)
    assert bool(returned.did_buffer_overflow) == bool(updated.did_buffer_overflow)
    check_contents(returned, base)
    recovered = factory.allocate(np.asarray(base), tag=tags)
    assert not recovered.did_buffer_overflow
    check_contents(recovered, base)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    disable_cells=[False, True],
    scalar_metric=[False, True],
  )
  def test_zero_threshold_strict_physical_cutoff(
    self, fmt, disable_cells, scalar_metric
  ):
    displacement, _ = space.free()
    metric = space.metric(displacement) if scalar_metric else displacement
    factory = partition.neighbor_list(
      metric, 24., 2., 0., disable_cell_list=disable_cells, format=fmt)
    base = np.array([[0., 0., 0.], [2., 0., 0.], [6., 6., 6.]])
    neighbors = factory.allocate(base, extra_capacity=4)
    update = jax.jit(factory.update)
    # Free displacements isolate the strict cutoff from periodic modulo rounding.
    for distance in [onp.nextafter(onp.float32(2), onp.float32(0)), onp.float32(2),
                     onp.nextafter(onp.float32(2), onp.float32(3))]:
      r = base.at[1, 0].set(distance)
      neighbors = update(r, neighbors)
      expected = {(0, 1), (1, 0)} if distance < 2 else set()
      if fmt is partition.OrderedSparse:
        expected = {(i, j) for i, j in expected if j < i}
      assert _standard_neighbor_pairs(neighbors) == expected
      assert int(neighbors.error.code) == 0
      onp.testing.assert_array_equal(neighbors.reference_position, r)

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    mask_self=[False, True],
  )
  def test_zero_threshold_cell_custom_mask_and_empty_slots(
    self, fmt, mask_self
  ):
    displacement, _ = space.periodic(24.)
    def mask(idx):
      center = np.arange(len(idx))[:, None]
      return np.where((idx + center) % 2 == 0, idx, len(idx))
    factory = partition.neighbor_list(
      displacement, 24., 2., 0., format=fmt, mask_self=mask_self,
      custom_mask_function=mask)
    base = np.array([[23.8, 1., 1.], [.2, 1., 1.], [2.3, 1., 1.],
                      [1.1, 1.8, 1.3], [18., 18., 18.]])
    neighbors = factory.allocate(base, extra_capacity=2)
    assert neighbors.cell_list_fn is not None
    update = jax.jit(factory.update)
    for step in range(6):
      r = base.at[0, 0].set(np.mod(base[0, 0] + .3 * step, 24.))
      r = r.at[2, 0].add(.07 * step)
      poisoned = dataclasses.replace(neighbors, idx=np.full_like(neighbors.idx, 5))
      neighbors = update(r, poisoned)
      physical = onp.asarray(r, dtype=onp.float64)
      delta = physical[:, None] - physical[None, :]
      delta -= 24. * onp.round(delta / 24.)
      i, j = onp.nonzero(onp.sum(delta * delta, axis=-1) < 4.)
      expected = {(int(a), int(b)) for a, b in zip(i, j)
                  if (not mask_self or a != b) and (a + b) % 2 == 0
                  and (fmt is not partition.OrderedSparse or b < a)}
      assert _standard_neighbor_pairs(neighbors) == expected
      assert int(neighbors.error.code) == 0
      onp.testing.assert_array_equal(neighbors.reference_position, r)


class StandardCellListRebuildTest(test_util.JAXMDTestCase):
  @parameterized.product(
    dim=[2, 3],
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    threshold=[0.0, 1.0],
  )
  def test_cell_candidate_slots_match_public_grid(self, dim, fmt, threshold):
    # Interleaved IDs in occupied cells; most of this anisotropic grid is empty.
    # A slot-dependent mask detects changes to the legacy cyclic cell ordering.
    box = onp.array([96., 108.] if dim == 2 else [48., 54., 60.], onp.float32)
    rng = onp.random.default_rng(681)
    r = rng.uniform(.3, 8., (39, dim)).astype(onp.float32)
    r[::4] = box - rng.uniform(.1, 1.3, r[::4].shape)
    r = np.asarray(r)
    cutoff = 2.5 + threshold
    cl_fn = partition.cell_list(box, cutoff, 1.25)
    cell = cl_fn.allocate(r, extra_capacity=3)
    ids = onp.asarray(cell.id_buffer).reshape(-1, cell.cell_capacity)
    cells = onp.floor(box / cutoff).astype(int)
    offsets = onp.array(list(onp.ndindex(*([3]*dim)))) - 1
    hashes = onp.cumprod(onp.r_[1, cells[:-1]])
    shifted = (onp.asarray(cell.particle_cell_id)[:, None] + offsets) % cells
    candidate = ids[onp.sum(shifted*hashes, axis=-1)].reshape(len(r), -1)
    physical = onp.asarray(r)
    delta = physical[:, None] - physical[onp.minimum(candidate, len(r)-1)]
    delta -= onp.round(delta/box)*box
    valid = (onp.sum(delta*delta, axis=-1) < cutoff**2) & (candidate < len(r))
    valid &= candidate != onp.arange(len(r))[:, None]
    valid &= onp.arange(candidate.shape[1])[None] % 3 != 1
    expected = {(i, int(candidate[i, s])) for i, s in zip(*onp.nonzero(valid))
                if fmt is not partition.OrderedSparse or candidate[i, s] < i}

    def mask(idx):
      return np.where(np.arange(idx.shape[1])[None] % 3 != 1, idx, len(idx))
    displacement, _ = space.periodic(box)
    factory = partition.neighbor_list(
        displacement, box, 2.5, threshold, format=fmt,
        capacity_multiplier=1.25, custom_mask_function=mask)
    allocated = factory.allocate(r, extra_capacity=3)
    assert allocated.cell_list_capacity == cell.cell_capacity
    assert _standard_neighbor_pairs(allocated) == expected
    rebuilt = jax.jit(factory.update)(r, allocated)
    assert int(rebuilt.error.code) == 0
    assert _standard_neighbor_pairs(rebuilt) == expected

  @parameterized.product(
    fmt=_STANDARD_NEIGHBOR_FORMATS,
    threshold=[0.0, 1.0],
  )
  def test_cell_motion_occupancy_and_overflow_recovery(self, fmt, threshold):
    displacement, _ = space.periodic(48.)
    factory = partition.neighbor_list(displacement, 48., 2., threshold,
                                      format=fmt, capacity_multiplier=1.)
    r = np.asarray(onp.array(list(onp.ndindex(4, 3, 2)), onp.float32)*5 + .31)
    n = factory.allocate(r, extra_capacity=2)
    update = jax.jit(factory.update)
    # Move whole populations across cell boundaries, with periodic wrapping.
    for move in [3.2, 17.4, 43.6]:
      x = np.mod(r + move, 48.)
      n = update(x, n)
      delta = (
      onp.asarray(x, onp.float64)[:, None]
      - onp.asarray(x, onp.float64)[None]
    )
      delta -= onp.round(delta/48.)*48.
      i, j = onp.nonzero(onp.sum(delta*delta, axis=-1) < (2.+threshold)**2)
      expected = {(int(a), int(b)) for a, b in zip(i, j)
                  if a != b and (fmt is not partition.OrderedSparse or b < a)}
      assert _standard_neighbor_pairs(n) == expected
      assert int(n.error.code) == 0
    crowded = np.ones_like(r)*.4 + np.arange(len(r))[:, None]*.001
    n = update(crowded, n)
    assert n.did_buffer_overflow
    assert int(n.error.code) & int(
      partition.PartitionErrorCode.CELL_LIST_OVERFLOW
    )
    recovered = factory.allocate(crowded, extra_capacity=2)
    assert int(recovered.error.code) == 0
    count = len(r)*(len(r)-1)
    assert len(_standard_neighbor_pairs(recovered)) == (
      count//2 if fmt is partition.OrderedSparse else count
    )

  @parameterized.product(
    fmt=[partition.Sparse, partition.OrderedSparse],
    disable_cells=[False, True],
    mask_self=[False, True],
  )
  def test_sparse_compaction_empty_rows_and_endpoint_padding(
    self, fmt, disable_cells, mask_self
  ):
    r = np.array([[1., 1., 1.], [1.3, 1.2, 1.], [1.7, 1., 1.],
                   [2.1, 1.2, 1.], [15., 16., 17.], [1.5, 1.5, 1.4],
                   [34., 35., 36.], [1.8, 1.8, 1.3]])
    def mask(idx):
      center = np.arange(len(idx))[:, None]
      return np.where((center % 3 != 1) & ((idx + center) % 2 == 0),
                       idx, len(idx))
    displacement, _ = space.periodic(48.)
    factory = partition.neighbor_list(
        displacement, 48., 2., 0., capacity_multiplier=1.,
        format=fmt, disable_cell_list=disable_cells, mask_self=mask_self,
        custom_mask_function=mask)
    n = factory.allocate(r)
    n = jax.jit(factory.update)(r + .01, n)
    assert int(n.error.code) == 0
    x = onp.asarray(r, onp.float64)
    delta = x[:, None]-x[None]
    delta -= onp.round(delta/48.)*48.
    i, j = onp.nonzero(onp.sum(delta*delta, axis=-1) < 4.)
    expected = {(int(a), int(b)) for a, b in zip(i, j)
                if (not mask_self or a != b) and a % 3 != 1 and (a+b) % 2 == 0
                and (fmt is not partition.OrderedSparse or b < a)}
    assert _standard_neighbor_pairs(n) == expected
    idx = onp.asarray(n.idx)
    valid = idx[0] < len(r)
    assert onp.all(idx[:, ~valid] == len(r))
    assert onp.all(onp.diff(idx[1, valid]) >= 0)
    # A multiplier of one fills the last capacity slot, including after empty rows.
    assert n.max_occupancy == len(expected)
    assert onp.all(valid)

  @parameterized.product(
    threshold=[0.0, 1.0],
  )
  def test_dense_custom_mask_inserted_id_still_checks_cutoff(self, threshold):
    # Custom masks may insert IDs, not only remove them. They must retain the
    # general Dense cutoff consumer, even when input candidates were prefiltered.
    r = np.array([[1., 1., 1.], [12., 12., 12.], [1.5, 1., 1.]])
    displacement, _ = space.periodic(48.)
    def mask(idx):
      return idx.at[0, -1].set(1)
    factory = partition.neighbor_list(
        displacement, 48., 2., threshold, format=partition.Dense,
        custom_mask_function=mask)
    n = factory.allocate(r, extra_capacity=2)
    n = jax.jit(factory.update)(r, n)
    assert _standard_neighbor_pairs(n) == {(0, 2), (2, 0)}
    assert int(n.error.code) == 0

  @parameterized.product(
    threshold=[0.0, 1.0],
    periodic=[False, True],
  )
  def test_dense_cell_candidates_preserve_both_metric_directions(
    self, threshold, periodic
  ):
    displacement, _ = space.periodic(48.) if periodic else space.free()
    def directed(a, b, **kwargs):
      return space.distance(displacement(a, b, **kwargs)) + np.where(
        a[0] > b[0], .75, 0.
      )
    radius = 2. + threshold
    r = np.array([[1., 1., 1.], [1.+radius-.25, 1., 1.], [20., 20., 20.]])
    factory = partition.neighbor_list(directed, 48., 2., threshold,
                                      format=partition.Dense)
    n = factory.allocate(r)
    assert n.cell_list_fn is not None
    # Cell filtering and Dense consumption historically apply opposite directions.
    # Each directed pair fails one of them. Dropping either test adds an edge.
    assert _standard_neighbor_pairs(n) == set()
    n = jax.jit(factory.update)(r + .00001, n)
    assert _standard_neighbor_pairs(n) == set()
    assert int(n.error.code) == 0

  def test_dense_periodic_cutoff_keeps_reverse_roundoff_check(self):
    # Forward and reverse minimum-image arithmetic round differently in f32.
    near = onp.nextafter(onp.float32(6.1), onp.float32(0))
    r = np.array([[.1, 1., 1.], [near, 1., 1.], [15., 15., 15.]])
    displacement, _ = space.periodic(24.)
    factory = partition.neighbor_list(displacement, 24., 6., 0.,
                                      format=partition.Dense)
    n = factory.allocate(r)
    assert _standard_neighbor_pairs(n) == set()
    n = jax.jit(factory.update)(r, n)
    assert _standard_neighbor_pairs(n) == set()
    assert int(n.error.code) == 0


if __name__ == '__main__':
  absltest.main()
