Simulation Routines
===========================

.. default-role:: code
.. automodule:: jax_md.simulate

Deterministic Simulation Environments
--------------------------------------
.. autofunction:: nve
.. autofunction:: nvt_nose_hoover
.. autofunction:: npt_nose_hoover

Nose-Hoover chain implementation notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The public NHC API, default parameters, integration equations, and precision
semantics are unchanged by these implementation optimizations. Chains of length
up to eight use statically indexed scalar propagation; longer chains retain a
scan fallback to limit compiler expansion.

Suzuki–Yoshida stage unrolling is bounded at eight. For short chains with at
least sixteen stages per thermostat half-step, chain positions also use scalar
carry. The stage count is ``chain_steps * sy_steps`` per half-step. Particle
momenta are rescaled at every substep, including when scalar carry is used.

These choices trade more compilation work for potentially lower execution
overhead; the benefit depends on the workload. Float32 extended-energy drift is
a known pre-existing numerical limitation, not introduced by the optimized
recurrence. No numerical correction is included here.

Stochastic Simulation Environments
--------------------------------------
.. autofunction:: nvt_langevin
.. autofunction:: brownian
.. autofunction:: npt_csvr
.. autofunction:: hybrid_swap_mc

Helper Functions
-----------------
.. autofunction:: velocity_verlet
.. autofunction:: nose_hoover_chain
.. autofunction:: npt_box
.. autofunction:: sinhx_x

Testing Functions
------------------
.. autofunction:: nvt_nose_hoover_invariant
.. autofunction:: npt_nose_hoover_invariant

Data Types
-----------
.. autoclass:: NoseHooverChain
.. autoclass:: NVEState
.. autoclass:: NVTNoseHooverState
.. autoclass:: NPTNoseHooverState
.. autoclass:: NVTLangevinState
.. autoclass:: BrownianState
.. autoclass:: NPTCSVRState
