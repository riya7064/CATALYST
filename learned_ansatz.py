"""
learned_ansatz.py

Adaptive UCCSD ansatz builder for VQE.

Strategy (Option 2 + ADAPT-style trial/rollback growth):

    Start with a singles-only UCC ansatz. When the optimization on the
    current circuit plateaus (energy stops improving for several
    consecutive evaluations), *try* growing: append the next excitation(s)
    from a pre-generated pool and optimize again.

    This is treated as a TRIAL, not a guaranteed improvement. Once the new,
    bigger circuit also plateaus, we compare its settled energy to the
    settled energy right before we grew:

        - If it improved meaningfully  -> keep the growth, it was worth it.
          Immediately try growing again (test the next excitation).
        - If it barely improved at all -> that excitation didn't help.
          Roll back to the smaller circuit (and its already-optimized
          parameters) and STOP. That smaller circuit is your final answer.

    This is the actual "win" condition: finding the ground state with fewer
    excitations than the full UCCSD circuit, instead of always building the
    whole thing. It mirrors the spirit of ADAPT-VQE (grow only what
    measurably helps), just with a fixed excitation ordering instead of
    ADAPT-VQE's gradient-based operator selection.

    Whenever the ansatz *is* grown, previously-optimized parameters are
    carried over unchanged (warm start); only the newly added parameter(s)
    get fresh small-random or zero initial values.

This module owns:
    - the excitation pool (generated once, in a fixed order: singles first,
      then doubles)
    - the "how much of the pool is currently active" state
    - plateau detection
    - the grow/judge/rollback decision logic
    - parameter inheritance across growth steps

This module does NOT own:
    - the Hamiltonian / qubit mapper construction (inject a mapper)
    - the classical optimizer (SPSA, COBYLA, etc. -- lives in your VQE driver)
    - the Estimator / energy evaluation

Typical usage
-------------
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=JordanWignerMapper(),
        plateau_threshold=1e-4,     # "stopped improving" sensitivity
        plateau_patience=5,         # how many flat steps in a row = stuck
        growth_benefit_threshold=1e-3,  # how big a drop counts as "worth it"
    )

    while not manager.is_done:
        circuit = manager.circuit
        x0 = manager.initial_point

        def cost_fn(x):
            energy = evaluate_energy(circuit, x)          # your estimator call
            changed = manager.observe(energy, params=x)    # feed every eval
            if changed:
                raise StopIteration   # ansatz just grew or rolled back+stopped
            return energy

        try:
            your_optimizer.minimize(cost_fn, x0=x0)
        except StopIteration:
            pass  # loop repeats (or exits, if manager.is_done is now True)

    print(manager.summary())          # final circuit size + growth/rollback log

See the `if __name__ == "__main__":` block at the bottom for a runnable,
backend-agnostic structural test (no VQE needed) using a fake energy curve
that intentionally saturates partway through the pool, to confirm the
manager stops early with a smaller-than-full-UCCSD ansatz.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
class _TrialState:
    """Snapshot taken right before a trial growth, so we can either keep
    going from here or roll all the way back to it."""
    pre_num_active: int
    pre_parameters: np.ndarray
    pre_settled_energy: float
    added_excitations: List[Excitation]
    # How many consecutive "not worth it" verdicts this same trial has
    # already received. Used by the confirm-before-rollback retry: a single
    # under-converged stage can no longer kill growth permanently -- the
    # negative verdict must repeat `rollback_confirmation_patience` times
    # in a row before the manager actually rolls back and stops.
    retries: int = 0


@dataclass
class GrowthEvent:
    """Record of one grow-and-judge decision, useful for logging / paper
    write-ups (shows exactly what was tried, and whether it was kept)."""
    added_excitations: List[Excitation]
    num_active_before: int
    num_active_after: int
    energy_before: float
    energy_after: float
    accepted: bool


class AnsatzGrowthSignal(Exception):
    """Raised by `GrowthObservingCostFunction` the instant `manager.observe()`
    reports that the ansatz just grew or just rolled back+stopped.

    Whatever optimizer is currently running (SPSA, COBYLA, ...) is, at that
    moment, iterating on a circuit/parameter-vector that no longer matches
    reality -- either the circuit just got bigger (new params appended) or
    the parameters were just reverted to a smaller circuit's optimum. Either
    way the in-flight `.minimize()` call must be unwound immediately rather
    than continuing to evaluate a stale ansatz. This exception is how that
    unwind happens: it propagates up out of the optimizer's `minimize()`
    call so the outer growth loop can react (start a fresh stage, or stop).
    """

    def __init__(self, is_done: bool):
        self.is_done = is_done
        super().__init__(f"Ansatz changed mid-optimization (manager.is_done={is_done})")


class GrowthObservingCostFunction:
    """Wraps a VQE cost function so that *every single evaluation* -- not
    just the final one at the end of a stage -- is fed to
    `AdaptiveAnsatzManager.observe()`.

    This replaces the old pattern of calling `manager.finalize_stage()`
    once after the optimizer finishes: that fallback only ever sees the
    stage's last (possibly unconverged) energy, so plateau/growth-benefit
    decisions were being made on a single, potentially noisy point. Here,
    the manager sees the true per-iteration trace and makes its plateau
    and growth-benefit calls the way it was designed to -- on real
    convergence behavior, not an artificially manufactured settled window.

    Duck-types the same interface `AdaptiveVQEOptimizer` / qiskit optimizers
    expect from a cost function (`__call__`, `.phase`, `.history`), so it
    can be dropped in wherever a plain `VQECostFunction` was used.
    """

    def __init__(self, cost_fn, manager: "AdaptiveAnsatzManager") -> None:
        self.cost_fn = cost_fn
        self.manager = manager

    @property
    def history(self):
        return self.cost_fn.history

    @property
    def phase(self) -> str:
        return self.cost_fn.phase

    @phase.setter
    def phase(self, value: str) -> None:
        self.cost_fn.phase = value

    def __call__(self, params: np.ndarray) -> float:
        energy = self.cost_fn(params)
        changed = self.manager.observe(energy, params=params)
        if changed:
            raise AnsatzGrowthSignal(is_done=self.manager.is_done)
        return energy


class AdaptiveAnsatzManager:
    """Owns the growable UCC ansatz, the plateau-detection rule, the
    grow/judge/rollback decision, and parameter warm-starting.

    Parameters
    ----------
    num_spatial_orbitals, num_particles, qubit_mapper :
        Same meaning as for qiskit-nature's UCC. Passed straight through.
    plateau_threshold : float
        |E_i - E_{i-1}| below this counts as "not improving" between
        consecutive evaluations -- used to detect that the *current* stage
        has settled (converged for now), not whether growing was worth it.
    plateau_patience : int
        Number of consecutive non-improving steps required before treating
        the current stage as settled.
    growth_benefit_threshold : float
        After growing, the new stage's settled energy is compared to the
        settled energy right before growth. If it dropped by at least this
        much, the growth is kept. If not, it's rolled back and the manager
        stops (this is the "smaller than full UCCSD" win condition). This
        is deliberately a separate, usually larger, tolerance from
        `plateau_threshold` -- plateau_threshold answers "has this stage
        stopped moving", growth_benefit_threshold answers "was growing
        actually worth the extra parameter".
    growth_batch_size : int
        How many new excitations to add from the pool per growth trial.
    rollback_confirmation_patience : int
        A "not worth it" verdict must repeat this many consecutive times
        before the manager actually rolls back and stops. On a verdict
        that isn't yet confirmed, the manager keeps the current (grown)
        circuit and simply gives it another settling window instead of
        discarding it -- so one under-converged stage can't permanently
        end growth on its own.
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
        plateau_threshold: float = 1e-4,
        plateau_patience: int = 5,
        growth_benefit_threshold: float = 1e-3,
        growth_batch_size: int = 1,
        rollback_confirmation_patience: int = 2,
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

        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.growth_benefit_threshold = growth_benefit_threshold
        self.growth_batch_size = growth_batch_size
        self.rollback_confirmation_patience = max(1, rollback_confirmation_patience)
        self.new_param_init = new_param_init
        self.new_param_std = new_param_std
        self._rng = np.random.default_rng(seed)

        self.pool, self._num_singles = build_excitation_pool(num_spatial_orbitals, num_particles)
        if len(self.pool) == 0:
            raise RuntimeError("Excitation pool is empty -- check num_particles/orbitals.")

        self.num_active = self._num_singles if start_with_singles_only else min(
            growth_batch_size, len(self.pool)
        )
        self.num_active = max(1, min(self.num_active, len(self.pool)))

        self.parameters: np.ndarray = (
            np.zeros(self.num_active)
            if new_param_init == "zeros"
            else self._rng.normal(0.0, new_param_std, self.num_active)
        )

        # Every energy ever observed, across all stages -- for plotting.
        self.full_energy_history: List[float] = []
        # Energies observed since the last growth decision -- for plateau
        # detection of the *current* stage only.
        self._stage_history: List[float] = []

        self._trial: Optional[_TrialState] = None
        self._done: bool = False

        self.growth_log: List[GrowthEvent] = []
        self.stage: int = 0

        self.ansatz = None
        self._rebuild_ansatz()

    # ----------------------------------------------------------------
    # Circuit construction
    # ----------------------------------------------------------------

    def _rebuild_ansatz(self) -> None:
        active = self.pool[: self.num_active]

        from qiskit_nature.second_q.circuit.library import HartreeFock

        hf_state = HartreeFock(self.num_spatial_orbitals, self.num_particles, self.qubit_mapper)

        def excitation_fn(num_spatial_orbitals, num_particles, _active=active):
            return _active

        self.ansatz = UCC(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=self.qubit_mapper,
            excitations=excitation_fn,
            initial_state=hf_state,
        )

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
        return self.num_active >= len(self.pool)

    @property
    def is_done(self) -> bool:
        """True once the manager has decided to stop -- either because a
        growth trial didn't help and was rolled back, or because the full
        pool was used and even that final stage plateaued."""
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
            True if the ansatz just grew, OR just rolled back and finished.
            Either way, the circuit under your optimizer is no longer the
            same object it was a moment ago (different size, or reverted
            parameters) -- stop the current optimizer run. Check
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

        if self._trial is not None:
            return self._judge_trial(settled_energy)

        # First plateau ever with no trial pending: try to grow.
        if self.is_fully_grown:
            self._done = True
            return False

        self._start_trial(settled_energy)
        return True

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
    # Internal: trial growth / judgement / rollback
    # ----------------------------------------------------------------

    def _start_trial(self, pre_settled_energy: float) -> None:
        add_n = min(self.growth_batch_size, len(self.pool) - self.num_active)
        added = self.pool[self.num_active: self.num_active + add_n]

        self._trial = _TrialState(
            pre_num_active=self.num_active,
            pre_parameters=self.parameters.copy(),
            pre_settled_energy=pre_settled_energy,
            added_excitations=list(added),
        )

        self.num_active += add_n
        self._rebuild_ansatz()

        new_params = (
            np.zeros(add_n)
            if self.new_param_init == "zeros"
            else self._rng.normal(0.0, self.new_param_std, add_n)
        )
        self.parameters = np.concatenate([self._trial.pre_parameters, new_params])
        self._stage_history = []

    def _judge_trial(self, settled_energy: float) -> bool:
        trial = self._trial
        improvement = trial.pre_settled_energy - settled_energy  # positive = got lower (better)

        if improvement >= self.growth_benefit_threshold:
            # Worth it: keep the growth, log acceptance, try growing further.
            # Any pending rollback confirmation is cleared -- a real
            # improvement resets the "how many times in a row has this
            # looked unhelpful" counter.
            self.growth_log.append(
                GrowthEvent(
                    added_excitations=trial.added_excitations,
                    num_active_before=trial.pre_num_active,
                    num_active_after=self.num_active,
                    energy_before=trial.pre_settled_energy,
                    energy_after=settled_energy,
                    accepted=True,
                )
            )
            self.stage += 1
            self._trial = None

            if self.is_fully_grown:
                self._done = True
                return False

            self._start_trial(settled_energy)
            return True

        # Looked unhelpful this time. Don't act on a single unlucky,
        # possibly under-converged verdict -- require it to repeat
        # `rollback_confirmation_patience` times in a row first.
        trial.retries += 1
        if trial.retries < self.rollback_confirmation_patience:
            # Give the same (already-grown) circuit another settling
            # window before condemning it: keep the current ansatz size
            # and parameters exactly as they are, just re-arm plateau
            # detection so the optimizer keeps refining instead of being
            # judged again on a trace that hasn't had time to converge.
            self._stage_history = []
            return False

        # Confirmed `rollback_confirmation_patience` times in a row: roll
        # back to the smaller circuit and stop for good.
        self.growth_log.append(
            GrowthEvent(
                added_excitations=trial.added_excitations,
                num_active_before=trial.pre_num_active,
                num_active_after=self.num_active,
                energy_before=trial.pre_settled_energy,
                energy_after=settled_energy,
                accepted=False,
            )
        )
        self.num_active = trial.pre_num_active
        self.parameters = trial.pre_parameters.copy()
        self._rebuild_ansatz()
        self._trial = None
        self._done = True
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
            "Growth trials:",
        ]
        for ev in self.growth_log:
            verdict = "KEPT" if ev.accepted else "ROLLED BACK (stopped here)"
            lines.append(
                f"  {ev.num_active_before} -> {ev.num_active_after} params: "
                f"E {ev.energy_before:.8f} -> {ev.energy_after:.8f} "
                f"(Δ={ev.energy_before - ev.energy_after:.2e})  [{verdict}]"
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
    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=mapper,
        plateau_threshold=1e-4,
        plateau_patience=5,
        growth_benefit_threshold=1e-3,
        growth_batch_size=1,
        new_param_init="small_random",
        seed=42,
    )

    print(manager.summary())
    print(f"(starting active excitations: {manager.num_active} out of {len(manager.pool)})")
    print()

    # True energy as a function of how many excitations are active: drops
    # fast at first, then genuinely saturates (extra excitations stop
    # mattering) around num_active ~= 14, well short of the full pool (26).
    def true_energy(num_active: int) -> float:
        # doubles added so far, but capped at 6: beyond that point extra
        # excitations contribute nothing at all (true saturation), so the
        # manager MUST detect this and roll back instead of using them.
        k = min(max(num_active - manager._num_singles, 0), 6)
        return -1.0 - 3.0 * (1 - np.exp(-0.35 * k))

    step = 0
    max_steps = 5000
    while not manager.is_done and step < max_steps:
        step += 1
        stage_age = len(manager._stage_history)
        target = true_energy(manager.num_active)
        # simulate the optimizer converging toward `target` within the
        # stage, plus a little numerical noise
        prev_energy = manager.full_energy_history[-1] if manager.full_energy_history else -1.0
        fake_energy = target + (prev_energy - target) * np.exp(-stage_age / 4.0)
        fake_energy += manager._rng.normal(0, 1e-6)
        fake_params = manager.parameters + 1e-4

        prev_active = manager.num_active
        prev_log_len = len(manager.growth_log)
        changed = manager.observe(fake_energy, params=fake_params)

        if changed and len(manager.growth_log) > prev_log_len:
            # a trial was judged this step (grown-and-kept, or rolled back)
            last = manager.growth_log[-1]
            tag = "KEPT" if last.accepted else "ROLLED BACK / STOPPED"
            print(
                f"step {step:4d}: trial {last.num_active_before}->{last.num_active_after} "
                f"params, E {last.energy_before:.6f} -> {last.energy_after:.6f}  [{tag}]"
            )
        elif changed:
            # a new trial just started (plateau detected, growing to test it)
            print(f"step {step:4d}: plateaued at {prev_active} params -- starting growth trial")

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