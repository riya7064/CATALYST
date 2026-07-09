"""
learned_ansatz.py

Adaptive UCCSD ansatz builder for VQE.

Strategy (ADAPT-VQE: gradient-screened growth):

    Start with a singles-only UCC ansatz and optimize. When the
    optimization on the current circuit plateaus (the *best-so-far*
    energy stops improving for several consecutive evaluations, after a
    minimum number of evaluations for this stage has elapsed), decide
    what to grow by screening -- not by trial-and-optimize.

    Screening means: for every excitation still outside the ansatz,
    compute a cheap energy gradient at the *current* optimized parameters
    with the candidate's own parameter fixed at 0 (a 2-point finite
    difference, no optimizer run). Whichever candidate(s) have the largest
    |gradient| are the ones actually worth paying for, so only those get
    added, and only then do we pay for a full optimization. All of these
    2 * len(remaining_candidates) screening evaluations for one round are
    batched into a single call to `energy_evaluator`.

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
    - plateau detection (best-so-far, with a minimum-evals floor)
    - gradient-based candidate screening and the grow/stop decision
    - parameter inheritance across growth steps

This module does NOT own:
    - the Hamiltonian / qubit mapper construction (inject a mapper)
    - the classical optimizer (SPSA, COBYLA, etc. -- lives in your VQE driver)
    - the Estimator itself -- but it DOES need a way to ask for energies
      at arbitrary circuit/parameter vectors, purely for the cheap
      gradient screen. That's the `energy_evaluator` callable you pass in
      below.

energy_evaluator interface
---------------------------
    energy_evaluator(pairs: List[Tuple[circuit, params]]) -> List[float]

    Batched: one call per screening round, covering every (circuit,
    params) pair that round needs (2 per remaining candidate), instead of
    one call per candidate. This matches the batched-Estimator-job design
    used elsewhere in the pipeline (see vqe.py's `energy_evaluator`, which
    packs every pair into a single `estimator.run(pubs)` job) -- so the
    same function you already pass into `run_learned_adaptive_ansatz` in
    vqe.py can be passed straight into this manager unchanged.

Typical usage
-------------
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    def energy_evaluator(pairs):
        # pairs: List[(circuit, params)] -> List[float], ONE Estimator job
        pubs = [(circuit, qubit_op, params) for circuit, params in pairs]
        results = estimator.run(pubs).result()
        return [float(r.data.evs) for r in results]

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=JordanWignerMapper(),
        energy_evaluator=energy_evaluator,
        plateau_threshold=1e-4,     # best-so-far improvement below this = flat
        plateau_patience=5,         # window (in evals) checked for flatness
        min_evals_per_stage=20,     # hard floor before a stage can plateau
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
manager stops early with a smaller-than-full-UCCSD ansatz -- and does so
only once the stage has genuinely converged, not on a false-plateau blip.
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
    energy_evaluator : Callable[[List[Tuple[circuit, params]]], List[float]]
        BATCHED. Called once per screening round with every (circuit,
        params) pair that round needs -- 2 per remaining candidate (a
        forward/backward finite-difference point) -- and must return one
        float per pair, in the same order. This is what replaces paying
        for a full optimization on every candidate, and it's meant to be
        backed by a single Estimator job (see the module docstring).
    plateau_threshold : float
        How much the *best-so-far* energy must improve over the trailing
        `plateau_patience` evaluations to still count as "improving".
        Below this, the current stage is considered settled.
    plateau_patience : int
        Width (in evaluations) of the trailing window checked for
        best-so-far improvement.
    min_evals_per_stage : int
        Hard floor: a stage cannot be declared plateaued until at least
        this many evaluations have been observed for it, regardless of
        how flat the trace looks. This exists because warm-started stages
        only have one truly "live" (near-zero) parameter, and many
        gradient-free optimizers (COBYLA in particular) spend their first
        several evaluations building an initial simplex/trust region
        before making real progress -- that early stretch can look flat
        without the stage having converged. Tune this to comfortably
        exceed your optimizer's typical "getting started" length.
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
        current optimized values). If your energy_evaluator has any
        sampling/shot noise in it, this needs to be large enough that the
        finite difference isn't dominated by that noise -- a noisy
        evaluator producing spuriously small gradients looks identical to
        genuine ADAPT-VQE convergence, so err on the larger side (0.05-0.1)
        unless you're on an exact statevector evaluator.
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
        energy_evaluator: Callable[[List[Tuple[Any, np.ndarray]]], List[float]],
        plateau_threshold: float = 1e-4,
        plateau_patience: int = 5,
        min_evals_per_stage: int = 20,
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
        if min_evals_per_stage < 2 * plateau_patience:
            warnings.warn(
                f"min_evals_per_stage={min_evals_per_stage} is smaller than "
                f"2 * plateau_patience={2 * plateau_patience}. The best-so-far "
                "plateau check needs at least 2*plateau_patience points to "
                "compare a 'before' window against a 'recent' window, so the "
                "effective floor will be max(min_evals_per_stage, "
                "2*plateau_patience) regardless of what you set here.",
                stacklevel=2,
            )

        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles
        self.qubit_mapper = qubit_mapper
        self.energy_evaluator = energy_evaluator

        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.min_evals_per_stage = min_evals_per_stage
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
    # Plateau detection (best-so-far, with a minimum-evals floor)
    # ----------------------------------------------------------------

    def _has_plateaued(self, history: List[float]) -> bool:
        """A stage is settled once the *best* energy seen hasn't
        meaningfully improved over the trailing `plateau_patience`
        evaluations -- checked only after `min_evals_per_stage` evals have
        happened at all.

        This is immune to a handful of near-identical evaluations early in
        an optimizer run (e.g. COBYLA building its initial simplex, or a
        warm-started stage's one live parameter starting at ~0): those
        make the *raw* trace look flat without the run having actually
        converged, but they don't fool a best-so-far comparison, because
        best-so-far only moves when something has genuinely improved.
        """
        effective_floor = max(self.min_evals_per_stage, 2 * self.plateau_patience)
        if len(history) < effective_floor:
            return False

        best = np.minimum.accumulate(np.asarray(history, dtype=float))
        # Best value up through the window vs. best value within the
        # trailing window: if the trailing window hasn't beaten what came
        # before it by more than plateau_threshold, nothing's improving.
        best_before = float(np.min(best[: -self.plateau_patience]))
        best_recent = float(np.min(best[-self.plateau_patience:]))
        return (best_before - best_recent) < self.plateau_threshold

    def _settled_energy(self, history: List[float]) -> float:
        """The energy this stage has settled at, once plateaued: the best
        (lowest) energy seen this stage -- not a mean, since a mean can be
        pulled upward by early, still-converging evaluations that a
        best-so-far view correctly discards."""
        return float(np.min(np.asarray(history, dtype=float)))

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
        floor = max(self.min_evals_per_stage, 2 * self.plateau_patience)
        for _ in range(floor + 1):
            changed = self.observe(final_energy, params=final_params)
        return changed

    # ----------------------------------------------------------------
    # Internal: gradient screening + grow/stop decision
    # ----------------------------------------------------------------
    #
    # Both finite-difference points for every remaining candidate are
    # packed into ONE call to `self.energy_evaluator` (batched), matching
    # the `energy_evaluator(pairs: List[(circuit, params)]) -> List[float]`
    # interface used by the rest of the pipeline (one Estimator job per
    # screening round instead of one job per candidate per point).

    def _screen_remaining_candidates(self) -> List[Tuple[int, float]]:
        """Returns [(pool_index, |gradient|), ...] for every excitation not
        yet active, sorted by |gradient| descending. Costs exactly one
        batched call to `energy_evaluator`, covering 2 * len(remaining)
        (circuit, params) pairs."""
        active_set = set(self.active_indices)
        remaining = [i for i in range(len(self.pool)) if i not in active_set]
        if not remaining:
            return []

        base = self.parameters
        pairs: List[Tuple[Any, np.ndarray]] = []
        for candidate_index in remaining:
            trial_indices = self.active_indices + [candidate_index]
            trial_circuit = self._build_ucc_for_indices(trial_indices)
            plus = np.concatenate([base, [self.gradient_eps]])
            minus = np.concatenate([base, [-self.gradient_eps]])
            pairs.append((trial_circuit, plus))
            pairs.append((trial_circuit, minus))

        energies = self.energy_evaluator(pairs)
        if len(energies) != len(pairs):
            raise RuntimeError(
                f"energy_evaluator returned {len(energies)} energies for "
                f"{len(pairs)} (circuit, params) pairs -- it must return "
                "exactly one float per pair, in the same order."
            )

        scored: List[Tuple[int, float]] = []
        for i, candidate_index in enumerate(remaining):
            e_plus = energies[2 * i]
            e_minus = energies[2 * i + 1]
            gradient = (e_plus - e_minus) / (2.0 * self.gradient_eps)
            scored.append((candidate_index, abs(gradient)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _grow_or_stop(self, settled_energy: float) -> bool:
        """Called once a stage has plateaued. Screens every remaining pool
        candidate by gradient (cheap, one batched call), and either commits
        the highest-gradient one(s) and starts a fresh optimization, or --
        if nothing clears `gradient_threshold` -- stops here. Returns True
        in both cases (the circuit/state changed one way or the other)."""
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
# instead of always exhausting the whole pool -- AND that it takes each
# stage far enough to actually converge before deciding that (this is the
# regression test for the false-plateau bug: the old raw-consecutive-delta
# check would fire almost immediately on a warm-started, slowly-moving
# parameter and stop with a tiny, under-optimized circuit).
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

    def fake_energy_evaluator(pairs) -> List[float]:
        """Stands in for a real batched Estimator call during the gradient
        screen. `pairs` is a list of (circuit, params) where each `params`
        is [existing optimized params..., candidate_theta]. Builds a toy
        energy surface whose central-difference gradient in candidate_theta
        exactly equals `true_energy(k) - true_energy(k+1)` -- i.e. it
        reproduces, via finite differences, how much adding one more
        excitation would actually lower the true energy curve above,
        including saturating to ~0 once the curve flattens. Returns one
        energy per pair, batched, matching the real pipeline's interface."""
        out = []
        for _circuit, params in pairs:
            k = len(params) - 1  # active excitations before this candidate
            e_low = true_energy(k, manager._num_singles)
            e_high = true_energy(k + 1, manager._num_singles)
            slope = e_low - e_high
            theta_new = params[-1]
            out.append(e_low - slope * np.sin(theta_new))
        return out

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        qubit_mapper=mapper,
        energy_evaluator=fake_energy_evaluator,
        plateau_threshold=1e-4,
        plateau_patience=5,
        min_evals_per_stage=20,
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
        # stage, plus a little numerical noise. Deliberately slow (tau=8,
        # not 4) so a false-plateau bug would trigger well before real
        # convergence -- this is what makes the test actually discriminate
        # between the old buggy detector and the fixed one.
        prev_energy = manager.full_energy_history[-1] if manager.full_energy_history else -1.0
        fake_energy = target + (prev_energy - target) * np.exp(-stage_age / 8.0)
        fake_energy += manager._rng.normal(0, 1e-6)
        fake_params = manager.parameters + 1e-4

        prev_log_len = len(manager.growth_log)
        changed = manager.observe(fake_energy, params=fake_params)

        if changed and len(manager.growth_log) > prev_log_len:
            last = manager.growth_log[-1]
            gap_to_target = abs(last.energy_before_growth - target)
            if last.stopped:
                print(
                    f"step {step:4d}: stopped at {last.num_active_before} params "
                    f"-- best remaining |gradient|={last.gradient_magnitude:.2e} "
                    f"< threshold  [STOPPED]"
                )
            else:
                print(
                    f"step {step:4d}: plateaued at {last.num_active_before} params "
                    f"(after {stage_age} evals, |E_settled - E_target|={gap_to_target:.2e}) -- "
                    f"gradient screen picked {last.added_excitations}, "
                    f"|gradient|={last.gradient_magnitude:.2e}  [GREW to "
                    f"{last.num_active_after}]"
                )
                assert gap_to_target < 1e-2, (
                    "Stage was declared plateaued while still far from its true "
                    "target energy -- the plateau detector is firing too early."
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
        "excitations -- smaller than the full UCCSD circuit, and each stage was "
        "genuinely converged (not falsely plateaued) before growing."
    )