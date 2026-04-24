"""
src/molcryst/bond/neighbor.py

Contains a function that wraps ASE's NeighborList but
returns Pythonic list-of-lists for neighbors and offsets.
"""

from ase.neighborlist import NeighborList
import numpy as np 
#from molcryst.ase_wrapper import covalent_radii


# Set of metallic atomic numbers (based on periodic table)
metallic_Z = set(
    list(range(19, 32)) +  # 19 K to 31 Ga
    list(range(37, 51)) +  # 37 Rb to 50 Sn
    list(range(55, 85)) +  # 55 Cs to 84 Po (some are semi-metals, adjust as needed)
    [3, 11, 12, 13]         # Li, Na, Mg, Al
)
#metallic_Z.update( {14, 32, 33, 51, 52, 85} ) # Si, Ge, As, Sb, Te, At (metalloids)
#metallic_Z.update( {34} ) #Se


def is_metal(Z):
    """Return True if atomic number Z corresponds to a metal."""
    return Z in metallic_Z

def is_metal_vectorized(Z):
    """Return a boolean array of same shape as Z, True where Z is metal.
    usage: metal_mask = is_metal_vectorized(atoms.numbers) """
    Z = np.asarray(Z)
    return np.isin(Z, list(metallic_Z))

def cutoffs_global_multiply( cutoffs, multiplier ):
    # cutoff is supposed to be in ase format with from ase.data import covalent_radii, or from molcryst.ase_wrapper import covalent_radii
    return [r * multiplier for r in cutoffs]

def cutoffs_global_add( cutoffs, dr ):
    # cutoff is supposed to be in ase format with from ase.data import covalent_radii, or from molcryst.ase_wrapper import covalent_radii
    return [r + dr for r in cutoffs]

def cutoffs_element_dict_add( cutoffs, element_dict, atoms ):
    # cutoff is supposed to be in ase format with from ase.data import covalent_radii, or from molcryst.ase_wrapper import covalent_radii
    return [r + element_dict.get(a.symbol, 0.0) for r, a in zip(cutoffs, atoms)]

def cutoffs_element_dict_set( cutoffs, element_dict, atoms ):
    # cutoff is supposed to be in ase format with from ase.data import covalent_radii, or from molcryst.ase_wrapper import covalent_radii
    return [element_dict.get(a.symbol, r) for r, a in zip(cutoffs, atoms)]


def neighbor_list_more_pythonic(atoms, cutoffs, bothways=True, self_interaction=False,):
    """
    I still need to add metalic bond tuggle by from molcryst.bond.neighbor import is_metal_vectorized

    Build a neighbor list using ASE's NeighborList, returning 
    Pythonic structures for neighbors and offsets.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object (pbc/cell if needed).
    cutoffs : list or array of float
        One cutoff per atom (ASE style).
    bothways : bool
        If True, symmetrical neighbors in both directions.
    self_interaction : bool
        If True, includes an atom as its own neighbor (rarely needed).

    Returns
    -------
    neighbors : list of lists of int
        neighbors[i] is a list of neighbor atom indices for atom i.
    offsets : list of lists of (ix, iy, iz)
        offsets[i] is a list of periodic offset tuples for each neighbor of atom i.
    """
    nl = NeighborList(cutoffs, bothways=bothways, self_interaction=self_interaction, skin=0.0,)
    nl.update(atoms)

    neighbors = []
    offsets = []
    for i in range(len(atoms)):
        neigh_indices, neigh_offsets = nl.get_neighbors(i)
        neighbors.append(list(neigh_indices))
        offsets.append([tuple(off) for off in neigh_offsets])
    return neighbors, offsets

