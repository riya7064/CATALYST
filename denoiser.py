"""
denoiser.py
-----------
Phase 2D: Noise study.
    - get_ideal_estimator()      : noiseless statevector simulation
    - build_noise_model()        : depolarizing + readout noise model
    - get_noisy_estimator()      : Aer estimator wired to that noise model
    - fold_circuit()             : unitary/global folding used to scale noise
    - zne_extrapolate()          : Zero-Noise Extrapolation on top of the
                                   noisy estimator
    - run_noise_comparison()     : convenience wrapper running
                                   ideal vs noisy vs noisy+ZNE for one
                                   set of ansatz parameters
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]


def _prepare_parameter_values(ansatz: QuantumCircuit, params) -> np.ndarray:
    """Normalize parameter values for Qiskit estimators.

    Qiskit expects either an empty sequence for unparameterized circuits or a
    1D array whose length matches the number of circuit parameters.
    """
    if params is None:
        return np.array([], dtype=float)

    values = np.asarray(params, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)

    if ansatz.num_parameters == 0:
        return np.array([], dtype=float)

    if values.size != ansatz.num_parameters:
        raise ValueError(
            f"Expected {ansatz.num_parameters} parameter value(s), received {values.size}."
        )

    return values


# --------------------------------------------------------------------------- #
# 1. Ideal (noiseless) simulator
# --------------------------------------------------------------------------- #
def get_ideal_estimator() -> StatevectorEstimator:
    """Exact statevector estimator -- the noise-free baseline."""
    return StatevectorEstimator()


def run_ideal(ansatz: QuantumCircuit, hamiltonian, params: np.ndarray) -> float:
    """Evaluate <H> on the ideal simulator for the given parameters."""
    estimator = get_ideal_estimator()
    parameter_values = _prepare_parameter_values(ansatz, params)
    job = estimator.run([(ansatz, hamiltonian, parameter_values)])
    return float(job.result()[0].data.evs)


# --------------------------------------------------------------------------- #
# 2. Noise model
# --------------------------------------------------------------------------- #
def build_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.02,
) -> NoiseModel:
    """
    Simple, tunable depolarizing + readout noise model.
    Meant to represent a "typical NISQ device" for the noise study, not any
    specific real backend.
    """
    noise_model = NoiseModel()

    err1 = depolarizing_error(single_qubit_error, 1)
    err2 = depolarizing_error(two_qubit_error, 2)
    noise_model.add_all_qubit_quantum_error(err1, ["rz", "sx", "x", "id"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx"])

    ro_error = ReadoutError(
        [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    )
    noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model


# --------------------------------------------------------------------------- #
# 3. Noisy simulator
# --------------------------------------------------------------------------- #
def get_noisy_estimator(noise_model: NoiseModel, shots: int = 4096) -> AerEstimator:
    """Aer Estimator (shot-based) configured with the given noise model."""
    return AerEstimator(
        options={"backend_options": {"noise_model": noise_model}, "run_options": {"shots": shots}}
    )


def run_noisy(
    ansatz: QuantumCircuit,
    hamiltonian,
    params: np.ndarray,
    noise_model: NoiseModel,
    shots: int = 4096,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
) -> float:
    """Evaluate <H> on the noisy simulator for the given parameters."""
    ansatz_t = transpile(ansatz, basis_gates=list(basis_gates), optimization_level=1)
    estimator = get_noisy_estimator(noise_model, shots=shots)
    parameter_values = _prepare_parameter_values(ansatz_t, params)
    job = estimator.run([(ansatz_t, hamiltonian, parameter_values)])
    return float(job.result()[0].data.evs)


# --------------------------------------------------------------------------- #
# 4. Zero-Noise Extrapolation (ZNE) via global unitary folding
# --------------------------------------------------------------------------- #
def fold_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Global folding: U -> U (U^dagger U)^n, giving an odd noise scale
    factor of (2n + 1). scale_factor must be an odd integer >= 1.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor must be an odd integer >= 1 (1, 3, 5, ...)")

    n_folds = (scale_factor - 1) // 2
    folded = circuit.copy_empty_like()
    folded.compose(circuit, inplace=True)
    for _ in range(n_folds):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


def zne_extrapolate(
    ansatz: QuantumCircuit,
    hamiltonian,
    params: np.ndarray,
    noise_model: NoiseModel,
    scale_factors: Sequence[int] = (1, 3, 5),
    shots: int = 4096,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
    extrapolator: str = "linear",
) -> Tuple[float, dict]:
    """
    Zero-Noise Extrapolation: run the (transpiled) circuit at several
    folded noise scale factors, then extrapolate the energy back to
    zero noise.

    Returns
    -------
    (zne_energy, details) where details contains the raw scale factors,
    measured energies, and the fit coefficients used.
    """
    ansatz_t = transpile(ansatz, basis_gates=list(basis_gates), optimization_level=1)
    estimator = get_noisy_estimator(noise_model, shots=shots)

    energies = []
    for scale in scale_factors:
        folded_circuit = fold_circuit(ansatz_t, scale)
        parameter_values = _prepare_parameter_values(folded_circuit, params)
        job = estimator.run([(folded_circuit, hamiltonian, parameter_values)])
        energies.append(float(job.result()[0].data.evs))

    scale_factors = np.array(scale_factors, dtype=float)
    energies_arr = np.array(energies, dtype=float)

    if extrapolator == "linear":
        coeffs = np.polyfit(scale_factors, energies_arr, deg=1)
        zne_energy = float(np.polyval(coeffs, 0.0))
    elif extrapolator == "quadratic":
        coeffs = np.polyfit(scale_factors, energies_arr, deg=2)
        zne_energy = float(np.polyval(coeffs, 0.0))
    elif extrapolator == "richardson":
        # Richardson extrapolation for arbitrary (not necessarily evenly
        # spaced) scale factors.
        n = len(scale_factors)
        zne_energy = 0.0
        for i, x_i in enumerate(scale_factors):
            weight = 1.0
            for j, x_j in enumerate(scale_factors):
                if i != j:
                    weight *= (0.0 - x_j) / (x_i - x_j)
            zne_energy += weight * energies_arr[i]
        coeffs = None
    else:
        raise ValueError("extrapolator must be 'linear', 'quadratic', or 'richardson'")

    details = {
        "scale_factors": scale_factors.tolist(),
        "energies": energies_arr.tolist(),
        "fit_coeffs": coeffs.tolist() if coeffs is not None else None,
        "extrapolator": extrapolator,
    }
    return zne_energy, details


# --------------------------------------------------------------------------- #
# 5. One-call comparison used by Phase 2D
# --------------------------------------------------------------------------- #
def run_noise_comparison(
    ansatz: QuantumCircuit,
    hamiltonian,
    params: np.ndarray,
    reference_energy: float,
    noise_model: NoiseModel = None,
    scale_factors: Sequence[int] = (1, 3, 5),
    shots: int = 4096,
) -> dict:
    """
    Run ideal / noisy / noisy+ZNE for a fixed set of ansatz parameters and
    report the errors relative to `reference_energy` (electronic + nuclear).
    """
    if noise_model is None:
        noise_model = build_noise_model()

    ideal_energy = run_ideal(ansatz, hamiltonian, params)
    noisy_energy = run_noisy(ansatz, hamiltonian, params, noise_model, shots=shots)
    zne_energy, zne_details = zne_extrapolate(
        ansatz, hamiltonian, params, noise_model, scale_factors=scale_factors, shots=shots
    )

    return {
        "ideal_energy": ideal_energy,
        "noisy_energy": noisy_energy,
        "zne_energy": zne_energy,
        "ideal_error": abs(ideal_energy - reference_energy),
        "noisy_error": abs(noisy_energy - reference_energy),
        "zne_error": abs(zne_energy - reference_energy),
        "zne_details": zne_details,
    }


if __name__ == "__main__":
    try:
        from hamiltonian import get_qubit_hamiltonian, get_reference_energy
        from ansatz import get_ansatz

        qubit_op, problem, mapper = get_qubit_hamiltonian("H2", mapping="jordan_wigner")
        ansatz, _ = get_ansatz("twolocal", problem=problem, mapper=mapper, reps=1)

        rng = np.random.default_rng(0)
        params = rng.uniform(-0.1, 0.1, ansatz.num_parameters)

        ref = get_reference_energy(problem) - problem.nuclear_repulsion_energy  # electronic part
        report = run_noise_comparison(ansatz, qubit_op, params, reference_energy=ref)
    except Exception as exc:
        print(
            f"Falling back to a synthetic circuit example because the chemistry stack is unavailable: {exc}"
        )

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        circuit = QuantumCircuit(1)
        circuit.ry(0.2, 0)
        observable = SparsePauliOp(["Z"])
        params = np.array([0.2])
        report = run_noise_comparison(circuit, observable, params, reference_energy=0.0)

    for k in ["ideal_energy", "noisy_energy", "zne_energy", "ideal_error", "noisy_error", "zne_error"]:
        print(f"{k}: {report[k]:.6f}")