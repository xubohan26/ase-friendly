import numpy as np
import sys
from ase import Atoms
from ase.io import read, write

def randomly_replace_atoms(atoms, replacement_list, seed=None):
    """
    Randomly replaces specified atoms with a new element.
    replacement_list: list of [old_element, new_element, count]
    """
    modified_atoms = atoms.copy()
    if seed is not None:
        np.random.seed(seed)

    old_el, new_el, count = replacement_list
    count = int(count)

    # Find indices of the element to be replaced
    indices = [i for i, s in enumerate(modified_atoms.get_chemical_symbols()) if s == old_el]

    if count > len(indices):
        raise ValueError(f"Cannot replace {count} {old_el} atoms; only {len(indices)} found.")

    # Select random indices and update symbols
    chosen_indices = np.random.choice(indices, size=count, replace=False)
    chosen_indices = sorted(chosen_indices)
    print(f"chosen_indices {chosen_indices}")
    # symbols = np.array(modified_atoms.get_chemical_symbols())
    # symbols[chosen_indices] = new_el
    # modified_atoms.set_chemical_symbols(symbols)

    # ordering the atoms of new and old elements, so the new element are right behind the old element
    all_old_indices = sorted([i for i, s in enumerate(modified_atoms.get_chemical_symbols()) if s == old_el])
    # 2. These are the positions where the new atoms should end up
    target_indices = all_old_indices[-count:]
    print(f"target_indices {target_indices}")

    chosen_set = set(chosen_indices)
    target_set = set(target_indices)

    chosen_no_overlap = list(chosen_set - target_set)
    target_no_overlap = list(target_set - chosen_set)
    print(f"chosen_no_overlap {chosen_no_overlap}")
    print(f"target_no_overlap {target_no_overlap}")

    # 3. Get current data to modify
    positions = modified_atoms.get_positions()
    symbols = np.array(modified_atoms.get_chemical_symbols())

    # 4. Swap positions and symbols
    for chosen, target in zip(chosen_no_overlap, target_no_overlap):
        positions[[chosen, target]] = positions[[target, chosen]]
    for target in target_indices:
        symbols[target] = new_el

    modified_atoms.set_positions(positions)
    modified_atoms.set_chemical_symbols(symbols)
    return modified_atoms

if __name__ == "__main__":
    # Minimalistic input handling
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()
    rep_parts = input("Enter replacement (e.g., 'Si P 1'):").strip()
    seed_str = input("Random seed (or 'None'): ").strip()
    ase_atoms = read(file_in, format=file_fmt)
    parts=rep_parts.split()
    replace_config = [parts[0], parts[1], int(parts[2])]
    # replace_config = ['Si', 'P', 1]
    seed = None if seed_str.lower() == 'none' else int(seed_str)
    new_atoms = randomly_replace_atoms(ase_atoms, replace_config, seed=seed)
    new_atoms.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')