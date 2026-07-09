"""
learned_ansatz.py

Adaptive UCCSD ansatz builder for VQE.

Strategy (ADAPT-VQE: gradient-screened growth):

    Start with a singles-only UCC ansatz and optimize. When the
    optimization on the current circuit plateaus (energy stops improving
    for several consecutive evaluations), decide what to grow by
    screening -- not by trial-and-optimize.

    Screening means: for every excitation still outside the ansatz,
    compute a cheap energy gradient at the *current* optimized parameters
    with the candidate's own parameter fixed at 0 (a 2-point finite
    difference, no optimizer run). Whichever candidate(s) have the largest
    |gradient| are the ones actually worth paying for, so only those get
    added, and only then do we pay for a full optimization.

    If every remaining candidate's gradient magnitude is below
    `gradient_threshold`, none of them would move the energy in any
    meaningful direction from here -- that's the ADAPT-VQE convergence
    criterion (max gradient < threshold), and the manager stops. That is
    the "win" condition: a smaller-than-full-UCCSD circuit, arrived at
    because nothing left in the pool actually helps, not because a fixed
    ordering ran out.

    Whenever the ansatz *is* grown, previously-optimized parameters are
    carried over unchanged (warm start); only the newly added parameter(s)
    get fresh small-random or zero initial values.

This module owns:
    - the excitation pool (generated once; screened in arbitrary order --
      the pool ordering no longer matters, since every candidate is
      screened every round)
    - the "how much of the pool is currently active" state
    - plateau detection
    - gradient-based candidate screening and the grow/stop decision
    - parameter inheritance across growth steps

This module does NOT own:
    - the Hamiltonian / qubit mapper construction (inject a mapper)
    - the classical optimizer (SPSA, COBYLA, etc. -- lives in your VQE driver)
    - the Estimator itself -- but it DOES need a way to ask for an energy
      at an arbitrary circuit/parameter vector, purely for the cheap
      gradient screen (2 evals per candidate, no optimization). That's
      the `energy_evaluator` callable you pass in below.

Typical usage
-------------
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=JordanWignerMapper(),
        energy_evaluator=evaluate_energy,  # same fn you use in cost_fn below
        plateau_threshold=1e-4,     # "stopped improving" sensitivity
        plateau_patience=5,         # how many flat steps in a row = stuck
        gradient_threshold=1e-3,    # ADAPT-VQE stopping criterion
    )

    while not manager.is_done:
        circuit = manager.circuit
        x0 = manager.initial_point

        def cost_fn(x):
            energy = evaluate_energy(circuit, x)          # your estimator call
            changed = manager.observe(energy, params=x)    # feed every eval
            if changed:
                raise StopIteration   # ansatz just grew, or just stopped
            return energy

        try:
            your_optimizer.minimize(cost_fn, x0=x0)
        except StopIteration:
            pass  # loop repeats (or exits, if manager.is_done is now True)

    print(manager.summary())          # final circuit size + growth log

See the `if __name__ == "__main__":` block at the bottom for a runnable,
backend-agnostic structural test (no VQE needed) using a fake energy curve
that intentionally saturates partway through the pool, to confirm the
manager stops early with a smaller-than-full-UCCSD ansatz.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.mappers import QubitMapper

Excitation = Tuple[Tuple[int, ...], Tuple[int, ...]]


# --------------------------------------------------------------------------
# Excitation pool generation
# --------------------------------------------------------------------------
#
# We prefer Qiskit Nature's own fermionic excitation generator so the pool
# is guaranteed consistent with what UCC would build internally. That
# function lives under a semi-internal path that has moved between
# qiskit-nature versions, so we import it defensively and fall back to a
# small hand-rolled generator (standard restricted singles/doubles,
# block spin-orbital ordering: alpha = [0, num_spatial_orbitals),
# beta = [num_spatial_orbitals, 2*num_spatial_orbitals)) if the import fails.
# (This fallback essentially never triggers on a normal install -- it's
# just insurance against a future qiskit-nature version moving the path.)

def _get_excitation_generator():
    try:
        from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import (
            generate_fermionic_excitations,
        )
        return generate_fermionic_excitations
    except ImportError:
        warnings.warn(
            "Could not import qiskit_nature's internal fermionic excitation "
            "generator (API may have moved in your qiskit-nature version). "
            "Falling back to a manual singles/doubles generator.",
            stacklevel=2,
        )
        return _manual_generate_fermionic_excitations


def _manual_generate_fermionic_excitations(
    num_excitations: int,
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
) -> List[Excitation]:
    """Fallback generator for restricted singles (num_excitations=1) and
    doubles (num_excitations=2), block spin-orbital ordering, spin-preserving
    only. Mirrors the shape of qiskit-nature's own generator closely enough
    to be used interchangeably as an excitation pool."""
    from itertools import combinations, product

    num_alpha, num_beta = num_particles
    occ_a = list(range(num_alpha))
    virt_a = list(range(num_alpha, num_spatial_orbitals))
    occ_b = list(range(num_spatial_orbitals, num_spatial_orbitals + num_beta))
    virt_b = list(range(num_spatial_orbitals + num_beta, 2 * num_spatial_orbitals))

    excitations: List[Excitation] = []

    if num_excitations == 1:
        for o, v in product(occ_a, virt_a):
            excitations.append(((o,), (v,)))
        for o, v in product(occ_b, virt_b):
            excitations.append(((o,), (v,)))
        return excitations

    if num_excitations == 2:
        for (o1, o2), (v1, v2) in product(combinations(occ_a, 2), combinations(virt_a, 2)):
            excitations.append(((o1, o2), (v1, v2)))
        for (o1, o2), (v1, v2) in product(combinations(occ_b, 2), combinations(virt_b, 2)):
            excitations.append(((o1, o2), (v1, v2)))
        for oa, ob, va, vb in product(occ_a, occ_b, virt_a, virt_b):
            excitations.append(((oa, ob), (va, vb)))
        return excitations

    raise NotImplementedError("Manual fallback only supports singles (1) and doubles (2).")


def build_excitation_pool(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
) -> Tuple[List[Excitation], int]:
    """Build the full singles-then-doubles excitation pool.

    Returns
    -------
    pool : list of excitation tuples, singles first, then doubles.
    num_singles : how many of the leading entries in `pool` are singles.
    """
    generate = _get_excitation_generator()
    singles = list(generate(1, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles))
    doubles = list(generate(2, num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles))
    return singles + doubles, len(singles)


# --------------------------------------------------------------------------
# Bookkeeping types
# --------------------------------------------------------------------------

@dataclass
class GrowthEvent:
    """Record of one gradient-screening decision, useful for logging / paper
    write-ups (shows exactly what was screened-in, with what gradient, or
    that nothing cleared the threshold and the manager stopped)."""
    added_excitations: List[Excitation]
    gradient_magnitude: float
    num_active_before: int
    num_active_after: int
    energy_before_growth: float
    stopped: bool = False


class AdaptiveAnsatzManager:
    """Owns the growable UCC ansatz, the plateau-detection rule, the
    gradient-based grow/stop decision, and parameter warm-starting.

    Parameters
    ----------
    num_spatial_orbitals, num_particles, qubit_mapper :
        Same meaning as for qiskit-nature's UCC. Passed straight through.
    energy_evaluator : Callable[[circuit, params], float]
        Used ONLY for the cheap gradient screen: given a trial circuit and
        a parameter vector, return its energy. The same function you
        already call in your VQE cost function works here -- pass it in.
        Each plateau costs `2 * len(remaining_pool)` calls to this (a
        forward/backward finite difference per remaining candidate), which
        is what replaces paying for a full optimization on every candidate.
    plateau_threshold : float
        |E_i - E_{i-1}| below this counts as "not improving" between
        consecutive evaluations -- used to detect that the *current* stage
        has settled (converged for now).
    plateau_patience : int
        Number of consecutive non-improving steps required before treating
        the current stage as settled.
    gradient_threshold : float
        The ADAPT-VQE stopping criterion. Once the largest |gradient|
        among all remaining candidates falls below this, nothing left in
        the pool would move the energy in any meaningful direction, so the
        manager stops. This is what gives you a smaller-than-full-UCCSD
        circuit instead of always exhausting the whole pool.
    gradient_eps : float
        Step size for the finite-difference gradient estimate (evaluated
        at +eps and -eps around the candidate's parameter, which is
        otherwise held at 0, with all other parameters fixed at their
        current optimized values).
    growth_batch_size : int
        How many of the highest-gradient candidates (that clear
        `gradient_threshold`) to add per growth step.
    new_param_init : {"zeros", "small_random"}
        How to initialize newly added parameters.
    new_param_std : float
        Std-dev for "small_random" initialization.
    seed : Optional[int]
        RNG seed for reproducibility of new-parameter initialization.
    start_with_singles_only : bool
        If True (default), stage 0 is all singles. If False, stage 0 is
        just `growth_batch_size` excitations from the pool.
    """

    def __init__(
        self,
        num_spatial_orbitals: int,
        num_particles: Tuple[int, int],
        qubit_mapper: QubitMapper,
        energy_evaluator: Callable[[Any, np.ndarray], float],
        plateau_threshold: float = 1e-4,
        plateau_patience: int = 5,
        gradient_threshold: float = 1e-3,
        gradient_eps: float = 1e-2,
        growth_batch_size: int = 1,
        new_param_init: str = "small_random",
        new_param_std: float = 0.01,
        seed: Optional[int] = None,
        start_with_singles_only: bool = True,
    ):
        if new_param_init not in ("zeros", "small_random"):
            raise ValueError("new_param_init must be 'zeros' or 'small_random'")

        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.qubit_mapper = qubit_mapper
        self.energy_evaluator = energy_evaluator

        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.gradient_threshold = gradient_threshold
        self.gradient_eps = gradient_eps
        self.growth_batch_size = growth_batch_size
        self.new_param_init = new_param_init
        self.new_param_std = new_param_std
        self._rng = np.random.default_rng(seed)

        self.pool, self._num_singles = build_excitation_pool(num_spatial_orbitals, num_particles)
        if len(self.pool) == 0:
            raise RuntimeError("Excitation pool is empty -- check num_particles/orbitals.")

        initial_count = self._num_singles if start_with_singles_only else min(
            growth_batch_size, len(self.pool)
        )
        initial_count = max(1, min(initial_count, len(self.pool)))

        # `active_indices` are the pool positions currently included in the
        # ansatz. Unlike the old fixed-order scheme, there's no pointer
        # into the pool any more -- every remaining index is re-screened
        # by gradient at every plateau, so membership in `active_indices`
        # is the only state that matters.
        self.active_indices: List[int] = list(range(initial_count))

        self.parameters: np.ndarray = (
            np.zeros(len(self.active_indices))
            if new_param_init == "zeros"
            else self._rng.normal(0.0, new_param_std, len(self.active_indices))
        )

        # Every energy ever observed, across all stages -- for plotting.
        self.full_energy_history: List[float] = []
        # Energies observed since the last growth decision -- for plateau
        # detection of the *current* stage only.
        self._stage_history: List[float] = []

        self._done: bool = False

        self.growth_log: List[GrowthEvent] = []
        self.stage: int = 0

        self.ansatz = None
        self._rebuild_ansatz()

    # ----------------------------------------------------------------
    # Circuit construction
    # ----------------------------------------------------------------

    @property
    def num_active(self) -> int:
        return len(self.active_indices)

    def _build_ucc_for_indices(self, indices: List[int]) -> UCC:
        """Build a UCC circuit for an arbitrary set of pool indices. Shared
        by `_rebuild_ansatz` (the real, committed ansatz) and the gradient
        screen (throwaway trial circuits, one per candidate, never stored
        on `self`)."""
        active = [self.pool[i] for i in indices]

        from qiskit_nature.second_q.circuit.library import HartreeFock

        hf_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, self.qubit_mapper)

        def excitation_fn(num_spatial_orbitals, num_particles, _active=active):
            return _active

        return UCC(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=self.qubit_mapper,
            excitations=excitation_fn,
            initial_state=hf_state,
        )

    def _rebuild_ansatz(self) -> None:
        self.ansatz = self._build_ucc_for_indices(self.active_indices)

        if self.ansatz.num_parameters != self.num_active:
            raise RuntimeError(
                f"Expected {self.num_active} ansatz parameters (one per active "
                f"excitation) but got {self.ansatz.num_parameters}. The "
                "parameter-inheritance assumption in this manager no longer "
                "holds for your qiskit-nature version -- inspect UCC's "
                "excitation_list / parameter ordering before trusting warm starts."
            )

    @property
    def circuit(self):
        """Current parameterized UCC circuit (Qiskit QuantumCircuit)."""
        return self.ansatz

    @property
    def initial_point(self) -> np.ndarray:
        """Warm-started parameter vector to hand to the optimizer for the
        current stage."""
        return self.parameters.copy()

    @property
    def is_fully_grown(self) -> bool:
        """True once every excitation in the pool is active -- there's
        nothing left to screen."""
        return self.num_active >= len(self.pool)

    @property
    def is_done(self) -> bool:
        """True once the manager has decided to stop -- either because no
        remaining candidate's gradient cleared `gradient_threshold`, or
        because the full pool is active and even that final stage
        plateaued."""
        return self._done

    # ----------------------------------------------------------------
    # Plateau detection
    # ----------------------------------------------------------------

    def _has_plateaued(self, history: List[float]) -> bool:
        if len(history) < self.plateau_patience + 1:
            return False
        recent = history[-(self.plateau_patience + 1):]
        deltas = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
        return all(d < self.plateau_threshold for d in deltas)

    def _settled_energy(self, history: List[float]) -> float:
        """The energy this stage has settled at, once plateaued: the mean
        of the last `plateau_patience` values (smooths out noise better
        than taking a single point)."""
        window = history[-self.plateau_patience:]
        return float(np.mean(window))

    # ----------------------------------------------------------------
    # The main entry point: call this after every energy evaluation
    # ----------------------------------------------------------------

    def observe(self, energy: float, params: Optional[np.ndarray] = None) -> bool:
        """Feed one energy evaluation (ideally every optimizer iteration,
        e.g. from a callback) to the manager.

        Returns
        -------
        changed : bool
            True if the ansatz just grew, OR just stopped (no remaining
            candidate cleared `gradient_threshold`). Either way, the
            circuit under your optimizer is no longer the same object it
            was a moment ago -- stop the current optimizer run. Check
            `manager.is_done` next: if True, you're finished; if False,
            start a fresh optimizer run on `manager.circuit` /
            `manager.initial_point`.
        """
        self.full_energy_history.append(energy)
        self._stage_history.append(energy)
        if params is not None:
            self.parameters = np.asarray(params, dtype=float)

        if self._done:
            return False

        if not self._has_plateaued(self._stage_history):
            return False

        settled_energy = self._settled_energy(self._stage_history)
        return self._grow_or_stop(settled_energy)

    def finalize_stage(self, final_energy: float, final_params: np.ndarray) -> bool:
        """Convenience for optimizers that don't expose a per-iteration
        callback. Call once, after your optimizer converges on the current
        stage, with its final energy/parameters. This manufactures a
        settled window from that single point so the grow/judge logic can
        still fire. Less precise than feeding every iteration via
        `observe`, but works as a fallback.

        Same return-value meaning as `observe`.
        """
        changed = False
        for _ in range(self.plateau_patience + 1):
            changed = self.observe(final_energy, params=final_params)
        return changed

    # ----------------------------------------------------------------
    # Internal: gradient screening + grow/stop decision
    # ----------------------------------------------------------------

    def _candidate_gradient(self, candidate_index: int) -> float:
        """Cheap 2-eval finite-difference estimate of dE/dtheta for one
        candidate excitation, at the current optimized parameters, with
        the candidate's own parameter fixed at 0. No optimizer run."""
        trial_indices = self.active_indices + [candidate_index]
        trial_circuit = self._build_ucc_for_indices(trial_indices)

        base = self.parameters
        plus = np.concatenate([base, [self.gradient_eps]])
        minus = np.concatenate([base, [-self.gradient_eps]])

        e_plus = self.energy_evaluator(trial_circuit, plus)
        e_minus = self.energy_evaluator(trial_circuit, minus)
        return (e_plus - e_minus) / (2.0 * self.gradient_eps)

    def _screen_remaining_candidates(self) -> List[Tuple[int, float]]:
        """Returns [(pool_index, |gradient|), ...] for every excitation not
        yet active, sorted by |gradient| descending."""
        active_set = set(self.active_indices)
        remaining = [i for i in range(len(self.pool)) if i not in active_set]
        scored = [(i, abs(self._candidate_gradient(i))) for i in remaining]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _grow_or_stop(self, settled_energy: float) -> bool:
        """Called once a stage has plateaued. Screens every remaining pool
        candidate by gradient (cheap), and either commits the
        highest-gradient one(s) and starts a fresh optimization, or -- if
        nothing clears `gradient_threshold` -- stops here. Returns True in
        both cases (the circuit/state changed one way or the other)."""
        if self.is_fully_grown:
            self._done = True
            return False

        scored = self._screen_remaining_candidates()
        winners = [(i, g) for i, g in scored if g >= self.gradient_threshold][: self.growth_batch_size]

        if not winners:
            best_grad = scored[0][1] if scored else 0.0
            self.growth_log.append(
                GrowthEvent(
                    added_excitations=[],
                    gradient_magnitude=best_grad,
                    num_active_before=self.num_active,
                    num_active_after=self.num_active,
                    energy_before_growth=settled_energy,
                    stopped=True,
                )
            )
            self._done = True
            return True

        winner_indices = [i for i, _ in winners]
        added = [self.pool[i] for i in winner_indices]
        pre_num_active = self.num_active

        self.active_indices = self.active_indices + winner_indices
        self._rebuild_ansatz()

        new_params = (
            np.zeros(len(winner_indices))
            if self.new_param_init == "zeros"
            else self._rng.normal(0.0, self.new_param_std, len(winner_indices))
        )
        self.parameters = np.concatenate([self.parameters, new_params])
        self._stage_history = []
        self.stage += 1

        self.growth_log.append(
            GrowthEvent(
                added_excitations=added,
                gradient_magnitude=winners[0][1],
                num_active_before=pre_num_active,
                num_active_after=self.num_active,
                energy_before_growth=settled_energy,
                stopped=False,
            )
        )
        return True

    # ----------------------------------------------------------------
    # Reporting helpers
    # ----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Excitation pool: {len(self.pool)} total "
            f"({self._num_singles} singles, {len(self.pool) - self._num_singles} doubles)",
            f"Final ansatz size: {self.num_active} / {len(self.pool)} excitations "
            f"({'full UCCSD' if self.is_fully_grown else 'smaller than full UCCSD'})",
            f"Total energy evaluations recorded: {len(self.full_energy_history)}",
            "Growth steps (gradient-screened):",
        ]
        for ev in self.growth_log:
            if ev.stopped:
                lines.append(
                    f"  stopped at {ev.num_active_before} params: best remaining "
                    f"|gradient|={ev.gradient_magnitude:.2e} < threshold  [STOPPED]"
                )
            else:
                names = ", ".join(str(exc) for exc in ev.added_excitations)
                lines.append(
                    f"  {ev.num_active_before} -> {ev.num_active_after} params: "
                    f"added {names} (|gradient|={ev.gradient_magnitude:.2e}), "
                    f"settled E before growth = {ev.energy_before_growth:.8f}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Structural self-test (no VQE/backend needed) -- run:
#   python learned_ansatz.py
#
# Uses a fake energy model that genuinely saturates partway through the
# pool (extra excitations beyond a certain point contribute ~0), so we can
# confirm the manager stops EARLY with a smaller-than-full-UCCSD ansatz,
# instead of always exhausting the whole pool.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    mapper = JordanWignerMapper()

    # True energy as a function of how many excitations are active: drops
    # fast at first, then genuinely saturates (extra excitations stop
    # mattering) around 6 doubles beyond the singles, well short of the
    # full pool. Defined before the manager since the fake evaluator below
    # (used for gradient screening) needs it.
    def true_energy(num_active: int, num_singles: int) -> float:
        # doubles added so far, but capped at 6: beyond that point extra
        # excitations contribute nothing at all (true saturation), so the
        # manager MUST detect this via a near-zero gradient and stop
        # instead of using them.
        k = min(max(num_active - num_singles, 0), 6)
        return -1.0 - 3.0 * (1 - np.exp(-0.35 * k))

    def fake_energy_evaluator(circuit, params: np.ndarray) -> float:
        """Stands in for a real Estimator call during the gradient screen.
        `params` is [existing optimized params..., candidate_theta]. Builds
        a toy energy surface whose central-difference gradient in
        candidate_theta exactly equals `true_energy(k) - true_energy(k+1)`
        -- i.e. it reproduces, via finite differences, how much adding one
        more excitation would actually lower the true energy curve above,
        including saturating to ~0 once the curve flattens."""
        k = len(params) - 1  # active excitations before this candidate
        e_low = true_energy(k, manager._num_singles)
        e_high = true_energy(k + 1, manager._num_singles)
        slope = e_low - e_high
        theta_new = params[-1]
        return e_low - slope * np.sin(theta_new)

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=mapper,
        energy_evaluator=fake_energy_evaluator,
        plateau_threshold=1e-4,
        plateau_patience=5,
        gradient_threshold=1e-3,
        growth_batch_size=1,
        new_param_init="small_random",
        seed=42,
    )

    print(manager.summary())
    print(f"(starting active excitations: {manager.num_active} out of {len(manager.pool)})")
    print()

    step = 0
    max_steps = 5000
    while not manager.is_done and step < max_steps:
        step += 1
        stage_age = len(manager._stage_history)
        target = true_energy(manager.num_active, manager._num_singles)
        # simulate the optimizer converging toward `target` within the
        # stage, plus a little numerical noise
        prev_energy = manager.full_energy_history[-1] if manager.full_energy_history else -1.0
        fake_energy = target + (prev_energy - target) * np.exp(-stage_age / 4.0)
        fake_energy += manager._rng.normal(0, 1e-6)
        fake_params = manager.parameters + 1e-4

        prev_log_len = len(manager.growth_log)
        changed = manager.observe(fake_energy, params=fake_params)

        if changed and len(manager.growth_log) > prev_log_len:
            last = manager.growth_log[-1]
            if last.stopped:
                print(
                    f"step {step:4d}: stopped at {last.num_active_before} params "
                    f"-- best remaining |gradient|={last.gradient_magnitude:.2e} "
                    f"< threshold  [STOPPED]"
                )
            else:
                print(
                    f"step {step:4d}: plateaued at {last.num_active_before} params -- "
                    f"gradient screen picked {last.added_excitations}, "
                    f"|gradient|={last.gradient_magnitude:.2e}  [GREW to "
                    f"{last.num_active_after}]"
                )

    print()
    print("Final state:")
    print(manager.summary())

    assert not manager.is_fully_grown, (
        "Expected the manager to stop EARLY (smaller than full UCCSD) given "
        "the saturating fake energy model, but it used the whole pool."
    )
    assert manager.is_done
    print(
        f"\nStructural self-test PASSED: stopped at {manager.num_active}/{len(manager.pool)} "
        "excitations -- smaller than the full UCCSD circuit, as intended."
    )