import numpy as np
from ase import Atoms


import os, sys
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import networkx as nx
from collections import deque

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


def connected_components_from_neighbors(neighbors):
    """
    Given a list-of-lists 'neighbors', build an undirected graph
    and return its connected components (each a set of atom indices).

    Parameters
    ----------
    neighbors : list of list of int
        neighbors[i] is the neighbor indices for atom i.

    Returns
    -------
    components : list of sets
        Each set corresponds to a connected component (molecule/cluster).
    """
    G = nx.Graph()
    num_atoms = len(neighbors)
    G.add_nodes_from(range(num_atoms))

    for i in range(num_atoms):
        for j in neighbors[i]:
            G.add_edge(i, j)

    return list(nx.connected_components(G))


#import numpy as np
#from collections import deque

def unwrap_component(positions, cell, component, neighbor_list, offsets, central_atom=None):
    if central_atom is None:
        central_atom = min(component)

    new_positions = {central_atom: positions[central_atom]}
    visited = set([central_atom])
    queue = deque([central_atom])

    while queue:
        i = queue.popleft()
        pos_i = new_positions[i]

        for k, j in enumerate(neighbor_list[i]):
            if j not in component:
                continue

            offset = np.array(offsets[i][k])
            disp = positions[j] + offset @ cell - positions[i]

            if j not in visited:
                new_positions[j] = pos_i + disp
                visited.add(j)
                queue.append(j)

    return new_positions

#from molcryst.bond.neighbor import neighbor_list_more_pythonic
#import warnings

import warnings
import numpy as np
from ase import Atoms
# your existing imports for neighbor_list_more_pythonic, connected_components_from_neighbors, unwrap_component

def update_atoms_positions_unwrap_all_molecules(
        atoms: Atoms,
        cutoffs,
        bothways: bool = True,
        self_interaction: bool = False,
        unwrap_near: tuple[float, float, float] | None = (0.25, 0.25, 0.25),
        unwrap_near_cartesian: bool = False
    ) -> None:
    """
    Update an ASE Atoms object's positions by unwrapping all connected components (molecules)
    according to periodic boundary conditions.

    Parameters
    ----------
    atoms : ase.Atoms
        The system containing one or more molecules/clusters.
    cutoffs : list[float]
        Per-atom cutoff values (typically half the desired bond length).
    bothways : bool, optional
        Whether to symmetrize the neighbor list (default True).
    self_interaction : bool, optional
        Whether to include an atom as its own neighbor (default False).
    unwrap_near : tuple of 3 floats, optional
        Coordinates of a point in space. Each molecule will anchor (unwrap)
        around its atom closest to this point. If None, the lowest‐index
        atom in each component is used as the anchor.
    unwrap_near_cartesian : bool, optional
        If True, interpret `unwrap_near` as Cartesian coordinates.
        If False (default), interpret `unwrap_near` as fractional coordinates.

    Returns
    -------
    None
        The function updates `atoms.positions` in place.
    """
    # 1. build neighbor list & offsets
    neighbor_list, offsets = neighbor_list_more_pythonic(
        atoms, cutoffs, bothways, self_interaction
    )

    # 2. find connected components (each is a molecule)
    components = connected_components_from_neighbors(neighbor_list)

    # 3. grab current positions + cell
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # 4. if requested, compute the Cartesian anchor point
    anchor_cart = None
    if unwrap_near is not None:
        if unwrap_near_cartesian:
            # interpret unwrap_near directly as Cartesian coordinates
            anchor_cart = np.array(unwrap_near)
        else:
            # compute the Cartesian anchor from fractional unwrap_near
            anchor_cart = np.dot(unwrap_near, cell)
            #print(cell)
    #print(unwrap_near)
    #print(unwrap_near_cartesian)
    #print(anchor_cart)

    # 5. unwrap each molecule
    for comp in components:
        # select per‐molecule central atom for anchoring the unwrap
        if anchor_cart is not None:
            comp_list = list(comp)
            subpos = positions[comp_list]
            # find index of the atom closest to the Cartesian anchor point
            local_idx = np.argmin(np.linalg.norm(subpos - anchor_cart, axis=1))
            comp_central = comp_list[local_idx]
        else:
            # fallback to the lowest-index atom if no anchor point is given
            comp_central = min(comp)

        # do the actual unwrap for this component
        unwrapped = unwrap_component(
            positions, cell, comp, neighbor_list, offsets,
            central_atom=comp_central
        )
        for idx, new_pos in unwrapped.items():
            positions[idx] = new_pos

    # 6. write back and warn
    atoms.set_positions(positions)
    warnings.warn(
        "Atom positions have been unwrapped. "
        "Recompute your neighbor list before any further PBC‐based analysis.",
        UserWarning
    )



def unwrap_by_bond_connectivity(atoms, global_multiply=1.0, global_add=0.1, unwrap_near=[0.25,0.25,0.25], unwrap_near_cartesian: bool = False):
    print(atoms)
    cutoffs = [covalent_radii[a.number] for a in atoms]
    # optional: modifying cutoffs

    # the two below are customizable
    element_add_dict = {
        #'H': -0.000,
        #'O': 0.001,
    }
    element_set_dict = {
        #'Co': 0.0,
        #'K':0.0
    }
    cutoffs=cutoffs_global_multiply(cutoffs, global_multiply)
    cutoffs=cutoffs_global_add(cutoffs, global_add)
    cutoffs=cutoffs_element_dict_add( cutoffs, element_add_dict, atoms )
    cutoffs=cutoffs_element_dict_set( cutoffs, element_set_dict, atoms )

    update_atoms_positions_unwrap_all_molecules(atoms, cutoffs, bothways=True, self_interaction=False, unwrap_near=unwrap_near, unwrap_near_cartesian=unwrap_near_cartesian)

    return atoms




if __name__ == "__main__":
    # 1. File Inputs
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()
    
    # 2. Parameter Inputs
    g_mult_str = input('bond cutoff global_multiply (e.g. 1.0): ').strip()
    g_add_str = input('bond cutoff global_add (e.g. 0.1): ').strip()
    u_near_str = input('unwrap_near fractional coordinate (e.g. "0.25 0.25 0.25"): ').strip()

    # 3. Format Inputs
    global_multiply = float(g_mult_str)
    global_add = float(g_add_str)
    unwrap_near = [float(x) for x in u_near_str.split()]
    

    # 4. Process and Save
    ase_atoms = read(file_in, format=file_fmt)
    
    new_atoms = unwrap_by_bond_connectivity(
        ase_atoms, 
        global_multiply=global_multiply, 
        global_add=global_add, 
        unwrap_near=unwrap_near, 
        unwrap_near_cartesian=False
    )
    
    new_atoms.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')
