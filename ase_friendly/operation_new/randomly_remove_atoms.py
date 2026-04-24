import numpy as np
from ase import Atoms
from ase.io import read, write
import sys

def randomly_remove_atoms(atoms, removal_counts, seed=None):
    """
    Identifies and randomly removes a specified number of atoms for given elements.

    This function ensures reproducibility by using a NumPy random seed, which is
    reset upon every call, preventing interference from previous random operations.

    Parameters:
        atoms (ase.Atoms): The input ASE Atoms object.
        removal_counts (dict): A dictionary where keys are element symbols (str)
                               and values are the number of atoms (int) to remove.
                               Example: {'Li': 6, 'O': 2}
        seed (int, optional): A seed for the random number generator to ensure
                              that the same atoms are removed on every run.

    Returns:
        ase.Atoms: A new Atoms object with the specified atoms removed.
    """
    # Create a copy to avoid modifying the original object
    modified_atoms = atoms.copy()

    # Reset the seed each time the function is called for consistent behavior
    if seed is not None:
        np.random.seed(seed)

    indices_to_delete = []
    current_composition = modified_atoms.get_chemical_symbols()

    for element, count_to_remove in removal_counts.items():
        # Find the indices of all atoms of the current element type
        element_indices = [i for i, symbol in enumerate(current_composition) if symbol == element]
        
        if count_to_remove > len(element_indices):
            raise ValueError(
                f"Cannot remove {count_to_remove} atoms of '{element}', "
                f"because only {len(element_indices)} exist in the structure."
            )
        if count_to_remove < 0:
            raise ValueError(f"Number of atoms to remove must be non-negative.")

        chosen_indices = np.random.choice(
            element_indices, 
            size=count_to_remove, 
            replace=False
        )
        indices_to_delete.extend(chosen_indices)

    del modified_atoms[indices_to_delete]
    return modified_atoms

if __name__ == "__main__":
    file_input = input('file_input: ').strip()
    file_format = input('file_format: ').strip()
    remove_str = input("'Please enter elements and counts to remove (e.g. 'Li 11 Fe 1'): ").strip()
    seed_str = input("Please enter random seed (or 'None' for no seed control):").strip()
    ase_atoms = read(file_input, format=file_format)
    parts = remove_str.split()
    atoms_to_remove = {parts[i]: int(parts[i+1]) for i in range(0, len(parts), 2)}
    #atoms_to_remove = {'Li': 2, 'O': 0}
    seed = None if str(seed_str)=='None' else int(seed_str)
    ase_atoms = randomly_remove_atoms(ase_atoms, atoms_to_remove, seed=seed)
    ase_atoms.write('zz_out.vasp', format='vasp')
    print('saved to zz_out.vasp')