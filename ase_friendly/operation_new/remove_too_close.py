import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

def remove_too_close(atoms, cutoff=0.2, mic=True):
    """
    Removes one atom out of every pair sitting closer than a cutoff distance.

    Pairs are always resolved in favour of the smaller atom index, so the atom
    appearing first in the file is the one that survives.

    Parameters:
        atoms (ase.Atoms): The input ASE Atoms object.
        cutoff (float): Distance in Angstrom below which two atoms are duplicates.
        mic (bool): Use the minimum image convention, i.e. also compare an atom
                    with the periodic images of its neighbours.

    Returns:
        ase.Atoms: A new Atoms object with the too-close atoms removed.
    """
    modified_atoms = atoms.copy()

    # neighbor_list only looks at periodic images along directions where pbc is True
    probe_atoms = modified_atoms.copy()
    probe_atoms.set_pbc(mic)
    idx_i, idx_j = neighbor_list('ij', probe_atoms, cutoff)

    # Walk the pairs once; drop the later atom unless one of the two is already gone.
    # This guarantees no surviving pair is still closer than the cutoff.
    keep_mask = np.ones(len(modified_atoms), dtype=bool)
    for i, j in zip(idx_i, idx_j):
        if i < j and keep_mask[i] and keep_mask[j]:
            keep_mask[j] = False

    del modified_atoms[~keep_mask]

    return modified_atoms

if __name__ == "__main__":
    file_input = input('file_input: ').strip()
    file_format = input('file_format: ').strip()
    cutoff_str = input('Distance cutoff in Angstrom (e.g. 0.2): ').strip()
    ase_atoms = read(file_input, format=file_format)
    ase_atoms = remove_too_close(ase_atoms, cutoff=float(cutoff_str))
    ase_atoms.write('zz_out.vasp', format='vasp')
    print('saved to zz_out.vasp')
