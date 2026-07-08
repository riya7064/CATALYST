"""
optimizer.py
------------
Phase 2B optimizer suite + Phase 4C Adaptive Optimizer Controller for the
VQE ground-state energy pipeline.

WHAT CHANGED FROM THE ORIGINAL DRAFT
=====================================
The overall class design (OptimizationHistory / OptimizerStrategy /
OptimizerFactory / ConvergenceCriterion / VQECostFunction /
AdaptiveVQEOptimizer) is preserved -- it was already clean, single-
responsibility, and the convergence rule ("has the *best-so-far* energy
stopped improving?") is a genuinely convergence-based signal rather than
a fixed iteration count, which is exactly what 4C asks for.

The one real bug was *how* SPSA was monitored. The original code ran
SPSA in fixed-size chunks:

    for chunk in range(0, total_budget, chunk_size):
        spsa = SPSA(maxiter=chunk)      # <-- a BRAND NEW optimizer object
        result = spsa.minimize(...)
        if criterion.is_converged(history):
            break

Every chunk boundary threw away and rebuilt the SPSA optimizer. SPSA is
not stateless: its perturbation size and learning-rate schedule are
calibrated relative to `maxiter` the first time `minimize()` runs (and,
if `resamplings`/momentum options are used, its internal state carries
across iterations too). Restarting it every `chunk_size` iterations
means the annealing schedule resets every chunk -- so the energy trace
you're monitoring for a plateau is partly an artifact of the restart,
not a clean signal of whether SPSA itself has converged. It also makes
"switch after a plateau, not a fixed count" only approximately true: you
can only ever switch on a `chunk_size` boundary, so you've reintroduced
a fixed-iteration granularity through the back door.

The fix: qiskit_algorithms' `SPSA` accepts a `termination_checker`
callback that fires after *every single iteration* with the running
(nfev, params, value, step_norm, accepted). That is the correct hook
for "monitor convergence metrics and dynamically decide, iteration by
iteration, whether to stay or switch." SPSA now runs exactly once, for
the full budget, and hands control back the instant the shared
`ConvergenceCriterion` says the trace has plateaued -- no restarts, no
chunk-boundary granularity, and the plateau signal is now trustworthy
because nothing about SPSA's internal schedule was disturbed while it
was being observed. COBYLA (with L-BFGS-B as a genuinely different,
gradient-based fallback if COBYLA raises) then gets whatever budget
remains.

    OptimizationHistory   -- records every cost-function evaluation
    OptimizerStrategy      -- (ABC) builds a concrete qiskit optimizer
        SPSAStrategy
        COBYLAStrategy
        LBFGSBStrategy      <- fallback / third optimizer
        SLSQPStrategy       <- extra strategy for the 2B comparison
        GradientDescentStrategy
    OptimizerFactory        -- registry / factory for the strategies above
    ConvergenceCriterion    -- decides "has this phase flattened out?"
    VQECostFunction         -- wraps Estimator + ansatz + Hamiltonian,
                               feeds a shared OptimizationHistory
    AdaptiveVQEOptimizer    -- 4C controller: SPSA (event-driven switch)
                               -> COBYLA -> L-BFGS-B fallback
    SingleOptimizerRunner   -- 2B baseline: one optimizer, start to finish
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
from qiskit_algorithms.optimizers import (
    SPSA,
    COBYLA,
    SLSQP,
    L_BFGS_B,
    GradientDescent,
)

# Raised by learned_ansatz.GrowthObservingCostFunction when the growth
# manager grows or rolls back mid-optimization. This is a *signal*, not a
# real optimizer failure, so it must never be swallowed by the generic
# `except Exception` fallback below -- it needs to propagate all the way
# out to the growth loop in vqe.py.
from learned_ansatz import AnsatzGrowthSignal

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. OptimizationHistory -- single source of truth for everything a run
#    needs to report (energies, params, eval count) and everything the
#    convergence criterion needs to reason about.
# ===========================================================================
class OptimizationHistory:
    """Records every cost-function evaluation across one or more optimizer
    phases (e.g. SPSA then COBYLA). A single instance is shared by every
    phase, so the energy trace is continuous across the SPSA -> COBYLA
    hand-off -- which is what lets the convergence criterion "see" the
    whole trajectory instead of restarting its view at each phase.
    """

    def __init__(self) -> None:
        self.energies: List[float] = []
        self.params: List[np.ndarray] = []
        self.phase_labels: List[str] = []
        self.n_evals: int = 0

    def record(self, energy: float, params: np.ndarray, phase: str = "") -> None:
        self.energies.append(float(energy))
        self.params.append(np.array(params, copy=True))
        self.phase_labels.append(phase)
        self.n_evals += 1

    # -- convenience analytics used by ConvergenceCriterion / reporting -- #
    def recent_window(self, window: int) -> np.ndarray:
        if not self.energies:
            return np.array([])
        return np.asarray(self.energies[-window:])

    def mean_abs_delta(self, window: int) -> float:
        """Mean absolute successive difference over the last `window`
        energies -- a raw "how jittery is the trace right now" signal.
        Kept for reporting/diagnostics; the switch decision itself uses
        `improvement_over`, which is robust to SPSA's per-eval noise."""
        recent = self.recent_window(window)
        if recent.size < 2:
            return float("inf")
        return float(np.mean(np.abs(np.diff(recent))))

    def stability_std(self, fraction: float = 0.1) -> float:
        """Std-dev of energy over the trailing `fraction` of all evals."""
        if not self.energies:
            return float("nan")
        tail = max(1, int(len(self.energies) * fraction))
        return float(np.std(self.energies[-tail:]))

    def best_so_far(self) -> np.ndarray:
        """Running best (lowest) energy at each evaluation -- the
        monotonically-non-increasing curve used to detect a plateau."""
        if not self.energies:
            return np.array([])
        return np.minimum.accumulate(np.asarray(self.energies))

    def improvement_over(self, patience: int) -> float:
        """How much the *best-so-far* energy improved over the last
        `patience` evaluations. Positive means it got lower (better);
        ~0 or negative means the run has plateaued -- no improvement.
        Using best-so-far rather than the raw trace is what makes this
        safe to call on a noisy, single-shot-per-eval SPSA trajectory:
        a single noisy eval can't un-plateau it."""
        best = self.best_so_far()
        if best.size <= patience:
            return float("inf")  # not enough history yet -> "still improving"
        return float(best[-patience - 1] - best[-1])

    def __len__(self) -> int:
        return len(self.energies)


# ===========================================================================
# 2. OptimizerStrategy -- one class per optimizer, each responsible only
#    for building a correctly-configured qiskit optimizer instance.
# ===========================================================================
class OptimizerStrategy(ABC):
    """Common interface for every optimizer choice in the suite."""

    def __init__(self, maxiter: int = 100, **kwargs: Any) -> None:
        self.maxiter = maxiter
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build(self):
        """Return a ready-to-use qiskit_algorithms optimizer instance."""
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(maxiter={self.maxiter})"


class SPSAStrategy(OptimizerStrategy):
    """Gradient-free, stochastic. Good first-phase explorer for noisy /
    hardware-like cost landscapes; cheap at 2 evals/iteration. Accepts a
    `termination_checker` kwarg (passed straight through to qiskit's
    SPSA) so the controller can stop it mid-run without rebuilding it."""

    @property
    def name(self) -> str:
        return "spsa"

    def build(self) -> SPSA:
        return SPSA(maxiter=self.maxiter, **self.kwargs)


class COBYLAStrategy(OptimizerStrategy):
    """Gradient-free local optimizer. Used as the refinement phase once
    SPSA's energy trace has flattened out."""

    @property
    def name(self) -> str:
        return "cobyla"

    def build(self) -> COBYLA:
        return COBYLA(maxiter=self.maxiter, **self.kwargs)


class LBFGSBStrategy(OptimizerStrategy):
    """Quasi-Newton, gradient-based optimizer (finite-difference gradients
    if none supplied). This is the chosen third / fallback optimizer: it
    is used automatically if the COBYLA refinement phase fails, since a
    gradient-based method is a genuinely different strategy from the two
    gradient-free ones above and is more likely to actually recover."""

    @property
    def name(self) -> str:
        return "lbfgsb"

    def build(self) -> L_BFGS_B:
        return L_BFGS_B(maxiter=self.maxiter, **self.kwargs)


class SLSQPStrategy(OptimizerStrategy):
    """Gradient-based strategy required for the 2B comparison table."""

    @property
    def name(self) -> str:
        return "slsqp"

    def build(self) -> SLSQP:
        return SLSQP(maxiter=self.maxiter, **self.kwargs)


class GradientDescentStrategy(OptimizerStrategy):
    """Plain gradient descent, required for the 2B comparison table."""

    def __init__(self, maxiter: int = 100, learning_rate: float = 0.01, **kwargs: Any) -> None:
        super().__init__(maxiter=maxiter, **kwargs)
        self.learning_rate = learning_rate

    @property
    def name(self) -> str:
        return "gradient_descent"

    def build(self) -> GradientDescent:
        return GradientDescent(maxiter=self.maxiter, learning_rate=self.learning_rate, **self.kwargs)


# ===========================================================================
# 3. OptimizerFactory -- registry / factory, replaces a dict-of-functions.
# ===========================================================================
class OptimizerFactory:
    _registry: Dict[str, Type[OptimizerStrategy]] = {
        "spsa": SPSAStrategy,
        "cobyla": COBYLAStrategy,
        "lbfgsb": LBFGSBStrategy,
        "l_bfgs_b": LBFGSBStrategy,
        "slsqp": SLSQPStrategy,
        "gradient_descent": GradientDescentStrategy,
        "gd": GradientDescentStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_cls: Type[OptimizerStrategy]) -> None:
        """Allows adding new optimizer choices without touching this file."""
        cls._registry[name.strip().lower()] = strategy_cls

    @classmethod
    def create(cls, name: str, maxiter: int = 100, **kwargs: Any) -> OptimizerStrategy:
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in cls._registry:
            raise ValueError(f"Unknown optimizer '{name}'. Options: {sorted(cls._registry)}")
        return cls._registry[key](maxiter=maxiter, **kwargs)

    @classmethod
    def build(cls, name: str, maxiter: int = 100, **kwargs: Any):
        """Shortcut: create the strategy and immediately return the
        underlying qiskit optimizer instance."""
        return cls.create(name, maxiter=maxiter, **kwargs).build()


# ===========================================================================
# 4. ConvergenceCriterion -- decides, from the shared history, whether the
#    SPSA phase should hand off to COBYLA. The rule is a "no improvement"
#    / patience check on the *best energy achieved so far* -- not how
#    noisy the raw trace looks. As long as the best-so-far energy keeps
#    dropping, SPSA keeps running; only once it stalls does the switch
#    happen. Every threshold is a constructor parameter (nothing hardcoded
#    into the decision itself), and it is checked once per iteration via
#    the termination_checker hook, not once per fixed-size chunk.
# ===========================================================================
@dataclass
class ConvergenceCriterion:
    patience: int = 5        # how many recent evals to look back over
    min_delta: float = 1e-6  # improvement smaller than this (Hartree) counts as "no improvement"
    min_evals: int = 10      # don't even consider switching before this many evals exist

    def is_converged(self, history: OptimizationHistory) -> bool:
        if len(history) < max(self.patience, self.min_evals):
            return False

        improvement = history.improvement_over(self.patience)
        no_improvement = improvement <= self.min_delta

        if no_improvement:
            logger.info(
                "No-improvement criterion met: best-energy improved by only %.3e "
                "over the last %d evals (min_delta=%.3e) -> switching optimizer",
                improvement, self.patience, self.min_delta,
            )
        return no_improvement


# ===========================================================================
# 5. VQECostFunction -- wraps Estimator + ansatz + Hamiltonian, records
#    into a shared OptimizationHistory so phases stay continuous.
# ===========================================================================
class VQECostFunction:
    """Callable cost function `energy = cost(params)` for a VQE problem.
    Every call is logged into `self.history`, tagged with whichever phase
    label is currently active (see `phase` setter), so a single history
    object can be reused across the SPSA -> COBYLA -> fallback pipeline.
    """

    def __init__(self, estimator, ansatz, hamiltonian, history: Optional[OptimizationHistory] = None) -> None:
        self.estimator = estimator
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.history = history if history is not None else OptimizationHistory()
        self._phase = "unspecified"

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        self._phase = value

    def __call__(self, params: np.ndarray) -> float:
        job = self.estimator.run([(self.ansatz, self.hamiltonian, params)])
        energy = float(job.result()[0].data.evs)
        self.history.record(energy, params, phase=self._phase)
        return energy


# ===========================================================================
# 6. AdaptiveVQEOptimizer -- the 4C controller. Implements the
#    SPSA -> COBYLA switch (with L-BFGS-B fallback), driven by an
#    event-based termination_checker rather than fixed-size chunks.
# ===========================================================================
@dataclass
class PhaseRecord:
    optimizer: str
    iterations: int
    reason: str


@dataclass
class OptimizationReport:
    final_energy: float
    optimal_params: np.ndarray
    num_iterations: int
    num_function_evals: int
    runtime_sec: float
    stability_std: float
    energy_history: List[float]
    phases: List[PhaseRecord] = field(default_factory=list)
    switched_at_eval: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["phases"] = [p.__dict__ for p in self.phases]
        return d


class AdaptiveVQEOptimizer:
    """Runs SPSA once, for up to `total_budget` iterations, monitoring the
    shared `criterion` after *every single SPSA iteration* via qiskit's
    `termination_checker` hook. SPSA keeps running as long as the best
    energy found so far keeps improving; the instant it plateaus (per
    `criterion.patience` / `criterion.min_delta`), SPSA halts itself and
    control hands off to COBYLA for whatever budget remains. If COBYLA
    raises, it automatically falls back to L-BFGS-B.

    Why event-driven instead of chunked: SPSA calibrates its step-size /
    perturbation schedule once, when `minimize()` starts. Restarting it
    every N iterations to "check in" throws that calibration away each
    time, so the very trace you're using to decide "has it plateaued?"
    would be contaminated by restart artifacts. Running it once and
    listening via `termination_checker` avoids that entirely -- nothing
    about the criterion here is a hardcoded iteration count.
    """

    def __init__(
        self,
        criterion: Optional[ConvergenceCriterion] = None,
        fallback_name: str = "lbfgsb",
        spsa_kwargs: Optional[Dict[str, Any]] = None,
        cobyla_kwargs: Optional[Dict[str, Any]] = None,
        fallback_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.criterion = criterion or ConvergenceCriterion()
        self.fallback_name = fallback_name
        self.spsa_kwargs = spsa_kwargs or {}
        self.cobyla_kwargs = cobyla_kwargs or {}
        self.fallback_kwargs = fallback_kwargs or {}

    def optimize(
        self,
        cost_fn: VQECostFunction,
        initial_point: np.ndarray,
        total_budget: int = 200,
    ) -> OptimizationReport:
        t0 = time.time()
        x_current = np.array(initial_point, copy=True)
        phases: List[PhaseRecord] = []

        # ---- Phase 1: SPSA, monitored iteration-by-iteration ---- #
        cost_fn.phase = "spsa"
        switch_eval: Dict[str, Optional[int]] = {"value": None}

        def termination_checker(nfev, params, value, step_norm, accepted) -> bool:
            # Called by SPSA after every iteration. Returning True stops
            # SPSA immediately -- this is the "dynamically decide whether
            # to stay or switch" hook 4C asks for.
            if self.criterion.is_converged(cost_fn.history):
                switch_eval["value"] = cost_fn.history.n_evals
                return True
            return False

        spsa = OptimizerFactory.build(
            "spsa",
            maxiter=total_budget,
            termination_checker=termination_checker,
            **self.spsa_kwargs,
        )
        result = spsa.minimize(fun=cost_fn, x0=x_current)
        x_current = result.x

        spsa_iters_used = int(getattr(result, "nit", total_budget) or total_budget)
        converged_early = switch_eval["value"] is not None
        switch_point = switch_eval["value"] if converged_early else cost_fn.history.n_evals

        reason = (
            "converged -> switching to COBYLA"
            if converged_early
            else "exhausted budget without switching"
        )
        phases.append(PhaseRecord("spsa", spsa_iters_used, reason))

        remaining = max(total_budget - spsa_iters_used, 0) if converged_early else 0

        # ---- Phase 2: COBYLA refinement, with L-BFGS-B fallback ---- #
        if remaining > 0:
            cost_fn.phase = "cobyla"
            try:
                cobyla = OptimizerFactory.build("cobyla", maxiter=remaining, **self.cobyla_kwargs)
                result = cobyla.minimize(fun=cost_fn, x0=x_current)
                x_current = result.x
                cobyla_iters = int(getattr(result, "nit", remaining) or remaining)
                phases.append(PhaseRecord("cobyla", cobyla_iters, "refinement phase"))
            except AnsatzGrowthSignal:
                # Not a COBYLA failure -- the growth manager just changed
                # the ansatz mid-run. Let it propagate to the growth loop.
                raise
            except Exception as exc:
                logger.warning("COBYLA phase failed (%s); falling back to %s", exc, self.fallback_name)
                cost_fn.phase = self.fallback_name
                fallback = OptimizerFactory.build(self.fallback_name, maxiter=remaining, **self.fallback_kwargs)
                result = fallback.minimize(fun=cost_fn, x0=x_current)
                x_current = result.x
                fb_iters = int(getattr(result, "nit", remaining) or remaining)
                phases.append(PhaseRecord(self.fallback_name, fb_iters, f"fallback after COBYLA error: {exc}"))

        runtime = time.time() - t0
        history = cost_fn.history

        return OptimizationReport(
            final_energy=float(result.fun),
            optimal_params=result.x,
            num_iterations=sum(p.iterations for p in phases),
            num_function_evals=history.n_evals,
            runtime_sec=runtime,
            stability_std=history.stability_std(),
            energy_history=list(history.energies),
            phases=phases,
            switched_at_eval=switch_point,
        )


# ===========================================================================
# 7. Single-optimizer runner -- kept for straight, non-adaptive comparisons
#    (2B: SPSA vs COBYLA vs SLSQP/L-BFGS-B vs Gradient Descent).
# ===========================================================================
class SingleOptimizerRunner:
    """Runs one optimizer strategy start-to-finish; used as the 2B baseline
    against which the AdaptiveVQEOptimizer (4C) is compared."""

    def __init__(self, strategy_name: str, maxiter: int = 100, **kwargs: Any) -> None:
        self.strategy_name = strategy_name
        self.maxiter = maxiter
        self.kwargs = kwargs

    def optimize(self, cost_fn: VQECostFunction, initial_point: np.ndarray) -> OptimizationReport:
        cost_fn.phase = self.strategy_name
        optimizer = OptimizerFactory.build(self.strategy_name, maxiter=self.maxiter, **self.kwargs)

        t0 = time.time()
        result = optimizer.minimize(fun=cost_fn, x0=initial_point)
        runtime = time.time() - t0

        history = cost_fn.history
        return OptimizationReport(
            final_energy=float(result.fun),
            optimal_params=result.x,
            num_iterations=int(result.nit) if getattr(result, "nit", None) is not None else history.n_evals,
            num_function_evals=int(result.nfev) if getattr(result, "nfev", None) is not None else history.n_evals,
            runtime_sec=runtime,
            stability_std=history.stability_std(),
            energy_history=list(history.energies),
            phases=[PhaseRecord(self.strategy_name, history.n_evals, "single-optimizer run")],
        )


# ===========================================================================
# 8. Demo / Phase 2B comparison + Phase 4C adaptive run
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from qiskit.primitives import StatevectorEstimator
    from hamiltonian import get_qubit_hamiltonian
    from ansatz import get_ansatz

    qubit_op, problem, mapper = get_qubit_hamiltonian("H2", mapping="jordan_wigner")
    ansatz, hf_state = get_ansatz("uccsd", problem=problem, mapper=mapper)

    estimator = StatevectorEstimator()
    x0 = np.zeros(ansatz.num_parameters)
    nre = problem.nuclear_repulsion_energy

    print("=== 2B baseline: single optimizers ===")
    for opt_name in ["cobyla", "slsqp", "spsa", "gradient_descent"]:
        cost_fn = VQECostFunction(estimator, ansatz, qubit_op)
        runner = SingleOptimizerRunner(opt_name, maxiter=50)
        report = runner.optimize(cost_fn, x0)
        total = report.final_energy + nre
        print(f"{opt_name:16s} total_energy={total:.6f}  "
              f"iters={report.num_iterations}  nfev={report.num_function_evals}  "
              f"stability_std={report.stability_std:.2e}  runtime={report.runtime_sec:.3f}s")

    print("\n=== 4C adaptive controller: SPSA -> COBYLA (fallback: L-BFGS-B) ===")
    cost_fn = VQECostFunction(estimator, ansatz, qubit_op)
    adaptive = AdaptiveVQEOptimizer(
        criterion=ConvergenceCriterion(patience=5, min_delta=1e-6, min_evals=10),
        fallback_name="lbfgsb",
    )
    report = adaptive.optimize(cost_fn, x0, total_budget=100)
    total = report.final_energy + nre
    print(f"adaptive         total_energy={total:.6f}  "
          f"nfev={report.num_function_evals}  runtime={report.runtime_sec:.3f}s  "
          f"switched_at_eval={report.switched_at_eval}")
    for phase in report.phases:
        print(f"  phase={phase.optimizer:8s} iters={phase.iterations:4d}  reason={phase.reason}")