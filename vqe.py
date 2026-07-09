"""vqe.py
--------
Orchestration layer for the Adapt-VQE pipeline.

This file owns the cross-module wiring only. Model building, ansatz
construction, optimization, and noise utilities live in their respective
modules; this file stitches them together into the Phase 1-6 workflows
used by the notebook.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint, MP2InitialPoint
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

from hamiltonian import (
    build_driver_for_molecule,
    get_electronic_structure_problem,
    get_mapper,
    get_reference_energy,
)
from active_space_reduction import auto_select_active_space, report_problem_size
from ansatz import ansatz_stats, build_hartree_fock_state, get_ansatz
from learned_ansatz import AdaptiveAnsatzManager
from optimizer import (
    AdaptiveVQEOptimizer,
    ConvergenceCriterion,
    OptimizationHistory,
    OptimizationInterrupted,
    SingleOptimizerRunner,
    VQECostFunction,
)
from denoiser import build_noise_model, run_noise_comparison


# --------------------------------------------------------------------------- #
# Shared setup helper: molecule name -> (qubit_op, problem, mapper, nuc_rep)
# --------------------------------------------------------------------------- #
def prepare_problem(
    molecule_name: str,
    mapping: str = "jordan_wigner",
    use_active_space: bool = False,
    n_occupied: int = 2,
    n_virtual: int = 2,
):
    """Build the qubit Hamiltonian for a molecule, optionally reducing the
    active space first (recommended for H2O / NH3)."""
    driver = build_driver_for_molecule(molecule_name)
    problem = get_electronic_structure_problem(driver)

    if use_active_space:
        report_problem_size(problem, f"{molecule_name} full space")
        problem = auto_select_active_space(problem, n_occupied=n_occupied, n_virtual=n_virtual)
        report_problem_size(problem, f"{molecule_name} active space")

    mapper = get_mapper(mapping)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    return qubit_op, problem, mapper


# --------------------------------------------------------------------------- #
# Phase 2C: initial point strategies
# --------------------------------------------------------------------------- #
def get_initial_point(
    strategy: str,
    ansatz,
    problem=None,
    seed: int = 42,
    scale: float = 0.1,
) -> np.ndarray:
    """
    strategy : "random" | "hf" | "mp2"

    "hf"   -> all-zero amplitudes (Hartree-Fock reference, only valid/meaningful
              for UCC-type ansatze where params=0 means "just HF").
    "mp2"  -> MP2-informed amplitudes seeded into the UCC ansatz
              (falls back to zeros if the ansatz/problem doesn't support it,
              e.g. HEA / TwoLocal have no chemical meaning for MP2 amplitudes).
    "random" -> small random perturbation around zero.
    """
    strategy = strategy.strip().lower()
    n = ansatz.num_parameters

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-scale, scale, n)

    if strategy == "hf":
        try:
            initial_point = HFInitialPoint()
            initial_point.ansatz = ansatz
            initial_point.problem = problem
            return initial_point.to_numpy_array()
        except Exception:
            return np.zeros(n)

    if strategy == "mp2":
        try:
            initial_point = MP2InitialPoint()
            initial_point.ansatz = ansatz
            initial_point.problem = problem
            return initial_point.to_numpy_array()
        except Exception:
            # Ansatz has no UCC excitation structure (e.g. HEA/TwoLocal) ->
            # MP2 amplitudes don't map onto it; fall back to zeros.
            return np.zeros(n)

    raise ValueError(f"Unknown initialization strategy '{strategy}'. Use random/hf/mp2.")


# --------------------------------------------------------------------------- #
# Phase 1: baseline VQE
# --------------------------------------------------------------------------- #
def run_baseline_vqe(
    molecule_name: str,
    ansatz_name: str = "uccsd",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    init_strategy: str = "hf",
    maxiter: int = 200,
    reps: int = 1,
    use_active_space: bool = False,
    n_occupied: int = 2,
    n_virtual: int = 2,
) -> dict:
    """
    Full Phase 1 baseline run:
        build Hamiltonian -> build ansatz -> pick initial point ->
        optimize -> compare to classical reference energy.
    """
    qubit_op, problem, mapper = prepare_problem(
        molecule_name, mapping, use_active_space, n_occupied, n_virtual
    )
    nuclear_repulsion = problem.nuclear_repulsion_energy

    ansatz, _hf_state = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)
    x0 = get_initial_point(init_strategy, ansatz, problem=problem)

    estimator = StatevectorEstimator()
    cost_fn = VQECostFunction(estimator, ansatz, qubit_op)
    runner = SingleOptimizerRunner(optimizer_name, maxiter=maxiter)
    report = runner.optimize(cost_fn, x0)

    computed_total_energy = report.final_energy + nuclear_repulsion
    reference_total_energy = get_reference_energy(problem)

    return {
        "molecule": molecule_name,
        "ansatz": ansatz_name,
        "optimizer": optimizer_name,
        "mapping": mapping,
        "init_strategy": init_strategy,
        "computed_energy": computed_total_energy,
        "reference_energy": reference_total_energy,
        "absolute_error": abs(computed_total_energy - reference_total_energy),
        "convergence_curve": [e + nuclear_repulsion for e in report.energy_history],
        "num_iterations": report.num_iterations,
        "num_function_evals": report.num_function_evals,
        "runtime_sec": report.runtime_sec,
        "optimal_params": report.optimal_params,
        "stability_std": report.stability_std,
        "ansatz_stats": ansatz_stats(ansatz),
    }


# --------------------------------------------------------------------------- #
# Phase 2A: ansatz comparison
# --------------------------------------------------------------------------- #
def compare_ansatze(
    molecule_name: str,
    ansatz_names: Sequence[str] = ("hea", "twolocal", "uccsd"),
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 2,
) -> dict:
    """Run the same molecule/optimizer with each ansatz type and collect
    accuracy / circuit depth / parameter count / runtime / convergence speed."""
    results = {}
    for name in ansatz_names:
        results[name] = run_baseline_vqe(
            molecule_name,
            ansatz_name=name,
            optimizer_name=optimizer_name,
            mapping=mapping,
            init_strategy="hf" if name == "uccsd" else "random",
            maxiter=maxiter,
            reps=reps,
        )
    return results


# --------------------------------------------------------------------------- #
# Phase 2B: optimizer comparison
# --------------------------------------------------------------------------- #
def compare_optimizers(
    molecule_name: str,
    optimizer_names: Sequence[str] = ("spsa", "cobyla", "slsqp", "gradient_descent"),
    ansatz_name: str = "uccsd",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 1,
) -> dict:
    """Run the same molecule/ansatz with each optimizer and collect
    iterations / function evals / runtime / stability / final energy."""
    results = {}
    for name in optimizer_names:
        results[name] = run_baseline_vqe(
            molecule_name,
            ansatz_name=ansatz_name,
            optimizer_name=name,
            mapping=mapping,
            init_strategy="hf",
            maxiter=maxiter,
            reps=reps,
        )
    return results


# --------------------------------------------------------------------------- #
# Phase 2C: initialization comparison
# --------------------------------------------------------------------------- #
def compare_initializations(
    molecule_name: str,
    init_strategies: Sequence[str] = ("random", "hf", "mp2"),
    ansatz_name: str = "uccsd",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 1,
) -> dict:
    """
    Run the same molecule/ansatz/optimizer with each initialization strategy
    and collect initial energy / convergence speed / final accuracy.
    """
    qubit_op, problem, mapper = prepare_problem(molecule_name, mapping)
    nuclear_repulsion = problem.nuclear_repulsion_energy
    reference_energy = get_reference_energy(problem)
    results = {}

    for strategy in init_strategies:
        ansatz, _ = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)
        x0 = get_initial_point(strategy, ansatz, problem=problem)

        estimator = StatevectorEstimator()
        initial_energy = float(estimator.run([(ansatz, qubit_op, x0)]).result()[0].data.evs) + nuclear_repulsion

        cost_fn = VQECostFunction(estimator, ansatz, qubit_op)
        runner = SingleOptimizerRunner(optimizer_name, maxiter=maxiter)
        report = runner.optimize(cost_fn, x0)

        final_energy = report.final_energy + nuclear_repulsion

        results[strategy] = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "reference_energy": reference_energy,
            "absolute_error": abs(final_energy - reference_energy),
            "num_iterations": report.num_iterations,
            "num_function_evals": report.num_function_evals,
            "convergence_curve": [e + nuclear_repulsion for e in report.energy_history],
        }

    return results


# --------------------------------------------------------------------------- #
# Phase 2D: noise study
# --------------------------------------------------------------------------- #
def run_noise_study(
    molecule_name: str,
    ansatz_name: str = "twolocal",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 100,
    reps: int = 1,
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.02,
    zne_scale_factors: Sequence[int] = (1, 3, 5),
) -> dict:
    """
    Optimize on the ideal simulator first (cheap), then evaluate the
    converged parameters under: ideal / noisy / noisy+ZNE. This isolates
    the *evaluation*-time noise degradation from optimizer noise-robustness
    (which is instead covered in compare_optimizers with SPSA vs gradient-based).
    """
    baseline = run_baseline_vqe(
        molecule_name,
        ansatz_name=ansatz_name,
        optimizer_name=optimizer_name,
        mapping=mapping,
        init_strategy="hf" if ansatz_name == "uccsd" else "random",
        maxiter=maxiter,
        reps=reps,
    )

    qubit_op, problem, mapper = prepare_problem(molecule_name, mapping)
    ansatz, _ = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)

    noise_model = build_noise_model(single_qubit_error, two_qubit_error, readout_error)
    electronic_reference = baseline["reference_energy"] - problem.nuclear_repulsion_energy

    noise_report = run_noise_comparison(
        ansatz,
        qubit_op,
        baseline["optimal_params"],
        reference_energy=electronic_reference,
        noise_model=noise_model,
        scale_factors=zne_scale_factors,
    )

    nuc = problem.nuclear_repulsion_energy
    return {
        "molecule": molecule_name,
        "ansatz": ansatz_name,
        "ideal_energy": noise_report["ideal_energy"] + nuc,
        "noisy_energy": noise_report["noisy_energy"] + nuc,
        "zne_energy": noise_report["zne_energy"] + nuc,
        "ideal_error": noise_report["ideal_error"],
        "noisy_error": noise_report["noisy_error"],
        "zne_error": noise_report["zne_error"],
        "zne_details": noise_report["zne_details"],
        "reference_energy": baseline["reference_energy"],
    }


def adaptive_select_active_space(problem, max_qubits: int = 8):
    """Pick an active-space window that preserves occupied orbitals when possible."""
    full_orbitals = problem.num_spatial_orbitals
    num_alpha = problem.num_particles[0]

    if 2 * num_alpha <= max_qubits:
        n_occ = num_alpha
        max_orbitals = max_qubits // 2
        n_virt = max(1, min(full_orbitals - num_alpha, max_orbitals - n_occ))
    else:
        n_occ = max(1, max_qubits // 4)
        n_virt = max(1, max_qubits // 2 - n_occ)

    reduced = auto_select_active_space(problem, n_occupied=n_occ, n_virtual=n_virt)
    return reduced, (n_occ, n_virt)


class ObservingCostFunction(VQECostFunction):
    """Fix #1: feeds every single energy evaluation to the growth manager
    in real time, instead of only reporting once at stage-end (the old
    `finalize_stage` fallback). This means plateau/growth decisions are
    judged against the *true* optimization trace, not a single,
    possibly-under-converged end point.

    The instant the manager signals a change (ansatz grew, or a trial was
    rolled back), this raises `OptimizationInterrupted` so the enclosing
    `AdaptiveVQEOptimizer.optimize()` call unwinds cleanly (see optimizer.py)
    instead of continuing to tune a circuit that's already stale.
    """

    def __init__(self, estimator, ansatz, hamiltonian, manager, history=None):
        super().__init__(estimator, ansatz, hamiltonian, history=history)
        self.manager = manager

    def __call__(self, params: np.ndarray) -> float:
        energy = super().__call__(params)
        if self.manager.observe(energy, params=np.asarray(params, dtype=float)):
            raise OptimizationInterrupted()
        return energy


def run_learned_adaptive_ansatz(
    problem,
    mapper,
    qubit_op,
    total_budget_per_stage: int = 80,
    budget_growth_per_param: int = 8,
    plateau_threshold: float = 1e-5,
    plateau_patience: int = 4,
    gradient_threshold: float = 1e-4,
    max_stages: int = 25,
    spsa_min_delta: float = 1e-4,
):
    """Run the learned ansatz growth loop and return the final circuit plus logs.

    total_budget_per_stage : base per-stage evaluation budget, granted to
        the very first (smallest) stage.
    budget_growth_per_param : Fix #2 -- extra evals granted *per currently-
        active parameter*, added on top of the base budget. Bigger ansatzes
        get proportionally more room to actually converge before the next
        growth decision, instead of every stage -- big or small --
        competing for the same fixed budget.
    gradient_threshold : forwarded to AdaptiveAnsatzManager -- the
        ADAPT-VQE stopping criterion. Once every remaining excitation's
        screened |gradient| falls below this, growth stops.
    spsa_min_delta : how much the best-so-far energy has to improve over
        `patience` evals before SPSA counts as "still improving". This was
        previously left at 1e-7 -- far tighter than SPSA's own eval-to-eval
        noise -- so SPSA almost never detected a plateau and burned its
        *entire* stage_budget on every single growth step (and stage_budget
        itself grows with num_active, so later stages paid the most). This
        is the single biggest lever on total runtime: loosening it lets
        COBYLA take over much sooner once SPSA has done its job.
    """
    # Used both for the real per-stage optimization (via ObservingCostFunction
    # below) and, here, as the manager's `energy_evaluator` for its cheap
    # gradient screen (2 evals per remaining candidate, no optimization) --
    # batched into a single Estimator call per screening round instead of
    # one call per candidate.
    estimator = StatevectorEstimator()

    def energy_evaluator(pairs):
        """pairs: List[(circuit, params)]. Runs every (circuit, params) pair
        in ONE Estimator job instead of one job per candidate -- this is
        what the screening step calls 2*len(remaining_candidates) worth of
        evaluations through, now as a single batched call."""
        pubs = [(circuit, qubit_op, params) for circuit, params in pairs]
        results = estimator.run(pubs).result()
        return [float(r.data.evs) for r in results]

    manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        energy_evaluator=energy_evaluator,
        plateau_threshold=plateau_threshold,
        plateau_patience=plateau_patience,
        gradient_threshold=gradient_threshold,
        growth_batch_size=1,
        new_param_init="small_random",
        seed=42,
    )

    shared_history = OptimizationHistory()
    stage_log = []
    t0 = time.time()
    step_guard = 0

    while not manager.is_done and step_guard < max_stages:
        step_guard += 1
        x0 = manager.initial_point
        stage_budget = total_budget_per_stage + budget_growth_per_param * manager.num_active

        cost_fn = ObservingCostFunction(estimator, manager.circuit, qubit_op, manager, history=shared_history)
        cost_fn.phase = f"stage{manager.stage}(n={manager.num_active})"

        optimizer = AdaptiveVQEOptimizer(
            criterion=ConvergenceCriterion(patience=5, min_delta=spsa_min_delta, min_evals=10)
        )
        stage_report = optimizer.optimize(cost_fn, x0, total_budget=stage_budget)
        # NOTE: no manager.finalize_stage() call anymore -- growth decisions
        # were already made live, inside cost_fn, via manager.observe().
        print(f"[stage {step_guard}] n_active={manager.num_active} budget={stage_budget} "
            f"total_evals_so_far={shared_history.n_evals} best_so_far={min(shared_history.energies):.6f}",
            flush=True)

        stage_log.append(
            {
                "step": step_guard,
                "num_active_params": manager.num_active,
                "stage_budget": stage_budget,
                "energy": stage_report.final_energy,
                "phases": [(phase.optimizer, phase.iterations, phase.reason) for phase in stage_report.phases],
                "switched_at_eval": stage_report.switched_at_eval,
                "interrupted": stage_report.interrupted,
                "cumulative_eval_at_stage_end": shared_history.n_evals,
            }
        )

    # The last history entry may belong to a rejected/rolled-back trial
    # rather than the ansatz we're actually keeping -- a rollback reverts
    # parameters without re-measuring their energy. Do one confirming
    # evaluation on the final retained ansatz/parameters so downstream
    # reporting (and ZNE) measures the real final answer, not a stale one.
    final_energy = float(
        estimator.run([(manager.circuit, qubit_op, manager.parameters)]).result()[0].data.evs
    )
    shared_history.record(final_energy, manager.parameters, phase="final_confirmation")

    runtime = time.time() - t0
    return {
        "manager": manager,
        "final_circuit": manager.circuit,
        "final_params": manager.parameters,
        "final_energy": final_energy,
        "history": shared_history,
        "stage_log": stage_log,
        "runtime_sec": runtime,
    }


def adaptive_zne(
    circuit,
    qubit_op,
    params,
    reference_electronic_energy: float,
    noise_model=None,
    noisy_error_threshold: float = 5e-3,
    shots: int = 2048,
):
    """Apply ZNE only when the noisy error is large enough to justify it."""
    if noise_model is None:
        noise_model = build_noise_model()

    from denoiser import run_ideal, run_noisy, zne_extrapolate

    ideal_energy = run_ideal(circuit, qubit_op, params)
    noisy_energy = run_noisy(circuit, qubit_op, params, noise_model, shots=shots)
    noisy_err = abs(noisy_energy - reference_electronic_energy)

    if noisy_err > noisy_error_threshold:
        zne_energy, details = zne_extrapolate(
            circuit,
            qubit_op,
            params,
            noise_model,
            scale_factors=(1, 3),
            shots=shots,
        )
        applied = True
    else:
        zne_energy, details = noisy_energy, None
        applied = False

    zne_err = abs(zne_energy - reference_electronic_energy)
    return {
        "ideal_energy": ideal_energy,
        "noisy_energy": noisy_energy,
        "noisy_error": noisy_err,
        "zne_applied": applied,
        "zne_energy": zne_energy,
        "zne_error": zne_err,
        "zne_details": details,
    }


def run_adaptive_pipeline(
    molecule_name: str,
    max_qubits: int = 8,
    mapping: str = "jordan_wigner",
    stage_budget: int = 50,
    budget_growth_per_param: int = 8,
    gradient_threshold: float = 1e-4,
    spsa_min_delta: float = 1e-4,
):
    """Run the adaptive 4A-4D pipeline for a selected molecule."""
    driver = build_driver_for_molecule(molecule_name)
    problem_full = get_electronic_structure_problem(driver)
    report_problem_size(problem_full, f"{molecule_name} full space")

    reduced_problem, (n_occ, n_virt) = adaptive_select_active_space(problem_full, max_qubits=max_qubits)
    report_problem_size(reduced_problem, f"{molecule_name} adaptive active space (occ={n_occ}, virt={n_virt})")

    mapper = get_mapper(mapping)
    qubit_op = mapper.map(reduced_problem.hamiltonian.second_q_op())
    nuclear_repulsion = reduced_problem.nuclear_repulsion_energy

    grow_result = run_learned_adaptive_ansatz(
        reduced_problem,
        mapper,
        qubit_op,
        total_budget_per_stage=stage_budget,
        budget_growth_per_param=budget_growth_per_param,
        gradient_threshold=gradient_threshold,
        spsa_min_delta=spsa_min_delta,
    )

    final_energy_electronic = grow_result["final_energy"]
    final_total_energy = final_energy_electronic + nuclear_repulsion

    zne_result = adaptive_zne(
        grow_result["final_circuit"],
        qubit_op,
        grow_result["final_params"],
        reference_electronic_energy=final_energy_electronic,
    )

    full_ref = get_reference_energy(problem_full)
    reduced_ref = get_reference_energy(reduced_problem)
    final_depth = ansatz_stats(grow_result["final_circuit"])["transpiled_depth"]
    

    return {
        "molecule": molecule_name,
        "n_occupied": n_occ,
        "n_virtual": n_virt,
        "qubits": qubit_op.num_qubits,
        "final_total_energy": final_total_energy,
        "full_space_reference": full_ref,
        "reduced_space_reference": reduced_ref,
        "absolute_error_vs_full": abs(final_total_energy - full_ref),
        "absolute_error_vs_reduced": abs(final_total_energy - reduced_ref),
        "num_function_evals": grow_result["history"].n_evals,
        "runtime_sec": grow_result["runtime_sec"],
        "final_circuit": grow_result["final_circuit"],
        "final_num_params": grow_result["final_circuit"].num_parameters,
        "final_circuit_depth": final_depth,
        "stage_log": grow_result["stage_log"],
        "full_energy_history": list(grow_result["history"].energies),
        "zne_result": zne_result,
        "manager_summary": grow_result["manager"].summary(),
        "nuclear_repulsion": nuclear_repulsion,
    }


def run_standard_pipeline(
    molecule_name: str,
    n_occupied: int,
    n_virtual: int,
    maxiter: int,
    reps: int = 1,
    mapping: str = "jordan_wigner",
):
    """Run the fixed baseline pipeline on the same active space as the adaptive run."""
    standard = run_baseline_vqe(
        molecule_name,
        ansatz_name="uccsd",
        optimizer_name="cobyla",
        mapping=mapping,
        init_strategy="hf",
        maxiter=maxiter,
        reps=reps,
        use_active_space=True,
        n_occupied=n_occupied,
        n_virtual=n_virtual,
    )

    qubit_op_std, problem_std, mapper_std = prepare_problem(
        molecule_name,
        mapping=mapping,
        use_active_space=True,
        n_occupied=n_occupied,
        n_virtual=n_virtual,
    )
    ansatz_std, _ = get_ansatz("uccsd", problem=problem_std, mapper=mapper_std, reps=reps)
    nuc_std = problem_std.nuclear_repulsion_energy
    standard_electronic_ref = standard["reference_energy"] - nuc_std
    standard_noise = run_noise_comparison(
        ansatz_std,
        qubit_op_std,
        standard["optimal_params"],
        reference_energy=standard_electronic_ref,
        scale_factors=(1, 3),
        shots=2048,
    )

    return {
        "standard": standard,
        "noise": standard_noise,
        "nuclear_repulsion": nuc_std,
        "qubit_op": qubit_op_std,
        "problem": problem_std,
    }


# --------------------------------------------------------------------------- #
# Demo / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Phase 1: baseline VQE (H2, UCCSD, COBYLA, HF init) ===")
    result = run_baseline_vqe("H2", ansatz_name="uccsd", optimizer_name="cobyla")
    print(
        f"computed={result['computed_energy']:.6f}  reference={result['reference_energy']:.6f}  "
        f"error={result['absolute_error']:.2e}  nfev={result['num_function_evals']}"
    )

    print("\n=== Phase 4: adaptive pipeline smoke test (H2) ===")
    adaptive = run_adaptive_pipeline("H2")
    print(
        f"qubits={adaptive['qubits']}  energy={adaptive['final_total_energy']:.6f}  "
        f"error={adaptive['absolute_error_vs_full']:.2e}  evals={adaptive['num_function_evals']}"
    )