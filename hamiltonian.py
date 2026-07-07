"""Molecular geometries plus OpenFermion fermionic and qubit Hamiltonian helpers."""

from __future__ import annotations

from typing import Sequence

DEFAULT_BASIS = "sto3g"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1

MOLECULES = {
	"h2": {
		"geometry": [
			("H", (0.0, 0.0, -0.3675)),
			("H", (0.0, 0.0, 0.3675)),
		],
		"basis": "sto3g",
		"charge": 0,
		"multiplicity": 1,
		"description": "h2",
	},
	"lih": {
		"geometry": [
			("Li", (0.0, 0.0, 0.0)),
			("H", (0.0, 0.0, 1.547)),
		],
		"basis": "sto3g",
		"charge": 0,
		"multiplicity": 1,
		"description": "lih",
	},
	"nh3": {
		"geometry": [
			("N", (0.0, 0.0, 0.0)),
			("H", (0.000000, 0.937700, -0.381600)),
			("H", (0.812100, -0.468850, -0.381600)),
			("H", (-0.812100, -0.468850, -0.381600)),
		],
		"basis": "sto3g",
		"charge": 0,
		"multiplicity": 1,
		"description": "nh3",
	},
}

H2_GEOMETRY = MOLECULES["h2"]["geometry"]
GEOMETRY_UNITS = "angstrom"


def get_molecule_settings(name: str = "h2") -> dict:
	"""Return the preset geometry and chemistry settings for a molecule."""
	key = name.strip().lower()
	if key not in MOLECULES:
		raise ValueError(f"Unknown molecule '{name}'. Choose one of: {', '.join(sorted(MOLECULES))}")
	return MOLECULES[key]


def choose_molecule_from_terminal() -> str:
	"""Prompt for a molecule name in the terminal and return the selected key."""
	print("Choose a molecule:")
	for index, molecule_name in enumerate(sorted(MOLECULES), start=1):
		print(f"  {index}. {molecule_name.upper()}")
	choice = input("Enter a molecule name or number: ").strip().lower()
	if choice.isdigit():
		choices = sorted(MOLECULES)
		index = int(choice) - 1
		if index < 0 or index >= len(choices):
			raise ValueError("Invalid molecule selection")
		return choices[index]
	if choice not in MOLECULES:
		raise ValueError(f"Unknown molecule '{choice}'. Choose one of: {', '.join(sorted(MOLECULES))}")
	return choice


def geometry_to_atom_string(geometry: Sequence[tuple[str, Sequence[float]]] = H2_GEOMETRY) -> str:
	"""Convert a geometry list into the atom string expected by PySCFDriver."""
	return "; ".join(
		f"{atom} {coordinates[0]} {coordinates[1]} {coordinates[2]}" for atom, coordinates in geometry
	)


def get_molecule_geometry(name: str = "h2") -> Sequence[tuple[str, Sequence[float]]]:
	"""Return the preset geometry for a molecule."""
	return get_molecule_settings(name)["geometry"]


def build_driver(
	geometry: Sequence[tuple[str, Sequence[float]]] = H2_GEOMETRY,
	basis: str = DEFAULT_BASIS,
	charge: int = DEFAULT_CHARGE,
	multiplicity: int = DEFAULT_MULTIPLICITY,
):
	"""Build a PySCFDriver for the requested molecule."""
	try:
		from qiskit_nature.second_q.drivers import PySCFDriver
	except ImportError as exc:
		raise ImportError(
			"qiskit-nature is required to build the PySCF driver"
		) from exc

	atom = geometry_to_atom_string(geometry)
	spin = multiplicity - 1
	return PySCFDriver(atom=atom, basis=basis, charge=charge, spin=spin)


def build_driver_for_molecule(name: str = "h2"):
	"""Build a PySCFDriver from one of the preset molecules."""
	settings = get_molecule_settings(name)
	return build_driver(
		geometry=settings["geometry"],
		basis=settings["basis"],
		charge=settings["charge"],
		multiplicity=settings["multiplicity"],
	)


def get_electronic_structure_problem(driver):
	"""Run the driver and return the electronic-structure problem."""
	return driver.run()


def get_second_q_hamiltonian(problem):
	"""Extract the fermionic second-quantized Hamiltonian from a Qiskit Nature problem."""
	if hasattr(problem, "hamiltonian") and hasattr(problem.hamiltonian, "second_q_op"):
		return problem.hamiltonian.second_q_op()
	if hasattr(problem, "second_q_ops"):
		second_q_ops = problem.second_q_ops()
		if isinstance(second_q_ops, tuple) and second_q_ops:
			return second_q_ops[0]
	raise TypeError("Unsupported electronic-structure problem type")


def get_openfermion_molecular_data(
	geometry: Sequence[tuple[str, Sequence[float]]] = H2_GEOMETRY,
	basis: str = DEFAULT_BASIS,
	charge: int = DEFAULT_CHARGE,
	multiplicity: int = DEFAULT_MULTIPLICITY,
	description: str = "h2",
):
	"""Build an OpenFermion MolecularData object for the requested geometry."""
	try:
		from openfermion.chem import MolecularData
	except ImportError as exc:
		raise ImportError("openfermion is required to build the molecular data") from exc

	molecule = MolecularData(
		geometry=list(geometry),
		basis=basis,
		multiplicity=multiplicity,
		charge=charge,
		description=description,
	)

	try:
		from openfermionpyscf import run_pyscf
	except ImportError:
		return molecule

	return run_pyscf(molecule)


def get_openfermion_molecular_data_for_molecule(name: str = "h2"):
	"""Build OpenFermion MolecularData for one of the preset molecules."""
	settings = get_molecule_settings(name)
	return get_openfermion_molecular_data(
		geometry=settings["geometry"],
		basis=settings["basis"],
		charge=settings["charge"],
		multiplicity=settings["multiplicity"],
		description=settings["description"],
	)


def get_openfermion_fermionic_hamiltonian(
	geometry: Sequence[tuple[str, Sequence[float]]] = H2_GEOMETRY,
	basis: str = DEFAULT_BASIS,
	charge: int = DEFAULT_CHARGE,
	multiplicity: int = DEFAULT_MULTIPLICITY,
	description: str = "h2",
):
	"""Build the OpenFermion fermionic Hamiltonian for the requested molecule."""
	molecule = get_openfermion_molecular_data(
		geometry=geometry,
		basis=basis,
		charge=charge,
		multiplicity=multiplicity,
		description=description,
	)

	if not hasattr(molecule, "get_molecular_hamiltonian"):
		raise ImportError(
			"openfermionpyscf is required to populate MolecularData before extracting the Hamiltonian"
		)

	try:
		from openfermion.transforms import get_fermion_operator
	except ImportError as exc:
		raise ImportError("openfermion is required to convert the molecular Hamiltonian") from exc

	return get_fermion_operator(molecule.get_molecular_hamiltonian())


def get_openfermion_fermionic_hamiltonian_for_molecule(name: str = "h2"):
	"""Build the OpenFermion fermionic Hamiltonian for one of the preset molecules."""
	settings = get_molecule_settings(name)
	return get_openfermion_fermionic_hamiltonian(
		geometry=settings["geometry"],
		basis=settings["basis"],
		charge=settings["charge"],
		multiplicity=settings["multiplicity"],
		description=settings["description"],
	)


def get_active_space_fermionic_hamiltonian_for_molecule(name: str = "h2"):
	"""Build an active-space reduced fermionic Hamiltonian for LiH and NH3 only."""
	key = name.strip().lower()
	fermionic_hamiltonian = get_openfermion_fermionic_hamiltonian_for_molecule(key)

	if key == "lih":
		occupied_orbitals = [0]
		unoccupied_orbitals = [4, 5]
	elif key == "nh3":
		occupied_orbitals = [0, 1, 2]
		unoccupied_orbitals = [14, 15]
	else:
		return fermionic_hamiltonian

	from openfermion.transforms import freeze_orbitals

	return freeze_orbitals(
		fermionic_hamiltonian,
		occupied=occupied_orbitals,
		unoccupied=unoccupied_orbitals,
		prune=True,
	)


def get_qubit_hamiltonian(
	geometry: Sequence[tuple[str, Sequence[float]]] = H2_GEOMETRY,
	basis: str = DEFAULT_BASIS,
	charge: int = DEFAULT_CHARGE,
	multiplicity: int = DEFAULT_MULTIPLICITY,
	description: str = "h2",
):
	"""Build the Jordan-Wigner qubit Hamiltonian for the requested molecule."""
	fermionic_hamiltonian = get_openfermion_fermionic_hamiltonian(
		geometry=geometry,
		basis=basis,
		charge=charge,
		multiplicity=multiplicity,
		description=description,
	)

	from openfermion.transforms import jordan_wigner

	return jordan_wigner(fermionic_hamiltonian)


def get_qubit_hamiltonian_for_molecule(name: str = "h2"):
	"""Build the Jordan-Wigner qubit Hamiltonian for one of the preset molecules."""
	settings = get_molecule_settings(name)
	return get_qubit_hamiltonian(
		geometry=settings["geometry"],
		basis=settings["basis"],
		charge=settings["charge"],
		multiplicity=settings["multiplicity"],
		description=settings["description"],
	)


def get_active_space_qubit_hamiltonian_for_molecule(name: str = "h2"):
	"""Build the Jordan-Wigner qubit Hamiltonian after active-space reduction for LiH and NH3."""
	fermionic_hamiltonian = get_active_space_fermionic_hamiltonian_for_molecule(name)

	from openfermion.transforms import jordan_wigner

	return jordan_wigner(fermionic_hamiltonian)


def print_pauli_terms(operator) -> None:
	"""Print the terms of an OpenFermion qubit operator."""
	for term, coeff in operator.terms.items():
		print(f"{coeff} {term}")


def main() -> None:
	"""Prompt for a molecule and print its Jordan-Wigner Pauli terms."""
	molecule_name = choose_molecule_from_terminal()
	if molecule_name in {"lih", "nh3"}:
		qubit_hamiltonian = get_active_space_qubit_hamiltonian_for_molecule(molecule_name)
		print(f"Selected molecule: {molecule_name.upper()} (active space)")
	else:
		qubit_hamiltonian = get_qubit_hamiltonian_for_molecule(molecule_name)
		print(f"Selected molecule: {molecule_name.upper()}")
	print_pauli_terms(qubit_hamiltonian)


if __name__ == "__main__":
	main()
