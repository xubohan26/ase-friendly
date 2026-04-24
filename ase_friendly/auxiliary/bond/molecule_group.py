"""
src/molcryst/bond/molecule_group.py

Contains functionality for identifying connected components (molecules)
in a neighbor graph.
"""

import networkx as nx
import numpy as np
from collections import deque
#from molcryst.bond.neighbor import neighbor_list_more_pythonic
from .neighbor import neighbor_list_more_pythonic
import warnings
from ase.geometry.dimensionality import analyze_dimensionality
from collections import Counter
import re

#import networkx as nx

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


def zzOld_update_atoms_positions_unwrap_all_molecules(atoms, cutoffs, bothways=True, self_interaction=False, central_atom=None):
    """
    Update an ASE Atoms object's positions by unwrapping all connected components (molecules)
    according to periodic boundary conditions.

    This function computes the neighbor list (using the provided per-atom cutoff values),
    determines connected components, and then unwraps each component so that all atoms are
    continuously connected relative to a selected central atom (default is the smallest index).
    The original atoms object is modified in place.

    WARNING:
      Since atom positions are changed by unwrapping molecules, one should build the neighbor
      list again with:
          ci.calc_neighbors_and_offsets(bothways=True, self_interaction=False)
      Some previously defined objects like ci.calc_coord_numbers() might still work; however, it is
      safer to recalculate the neighbor list immediately after executing this code, so that the neighbor
      list offsets are updated.

    Parameters
    ----------
    atoms : ase.Atoms
        The system containing one or more molecules/clusters.
    cutoffs : list or array of float
        Per-atom cutoff values (remember: typically half the desired bond length).
    bothways : bool, optional
        Whether to symmetrize the neighbor list (default True).
    self_interaction : bool, optional
        Whether to include an atom as its own neighbor (default False).
    central_atom : int, optional
        Optional override for the central atom to use when unwrapping (default: min(index) per component).

    Returns
    -------
    None
        The function updates atoms.positions in place.
    """
    # Build neighbor list and offsets once.
    neighbor_list, offsets = neighbor_list_more_pythonic(atoms, cutoffs, bothways, self_interaction)

    # Get connected components.
    components = connected_components_from_neighbors(neighbor_list)
    
    # Get current positions and cell.
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # Unwrap each component and update atoms.positions accordingly.
    for comp in components:
        # If a specific central_atom is provided, use it for all; otherwise, use the smallest index in the component.
        comp_central = central_atom if central_atom is not None else min(comp)
        unwrapped = unwrap_component(positions, cell, comp, neighbor_list, offsets, central_atom=comp_central)
        for i, new_pos in unwrapped.items():
            positions[i] = new_pos

    # Update atoms positions.
    atoms.set_positions(positions)

    # Issue the warning.
    warnings.warn("Since atom positions are changed by unwrapping molecules, one should build the neighbor list again "
                  "with ci.calc_neighbors_and_offsets(bothways=True, self_interaction=False). Some previously defined "
                  "objects like ci.calc_coord_numbers() might still work; however, it is safer to recalculate the neighbor "
                  "list right after execution of this code, so that the neighbor list's offsets are updated.",
                  UserWarning)

#from ase.geometry.dimensionality import analyze_dimensionality
import networkx as nx
from ase.geometry.dimensionality import analyze_dimensionality

import networkx as nx
from ase.geometry.dimensionality import analyze_dimensionality
import warnings

def connected_components_and_dimensions_from_neighbors(atoms, neighbors):
    """
    For a given ASE Atoms object and a neighbor list (list-of-lists of ints),
    find its connected components (clusters) and analyze the connectivity
    dimensionality of each one. The result dictionary will have integer keys
    corresponding to the dimension (0, 1, 2, or 3) and values as lists of
    ASE Atoms objects representing each connected component.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure.
    neighbors : list of list of int
        neighbors[i] is a list of indices of atoms that are neighbors of atom i.
    
    Returns
    -------
    components_dim : dict
        A dictionary mapping an integer connectivity dimension (0, 1, 2, or 3) 
        to a list of ASE Atoms objects representing each connected component.
    """
    # Build connected components using networkx.
    def connected_components_from_neighbors(neighbors):
        G = nx.Graph()
        num_atoms = len(neighbors)
        G.add_nodes_from(range(num_atoms))
        for i in range(num_atoms):
            for j in neighbors[i]:
                G.add_edge(i, j)
        return list(nx.connected_components(G))
    
    comps = connected_components_from_neighbors(neighbors)
    components_dim = {}
    
    for comp in comps:
        comp_atoms = atoms[list(comp)]
        try:
            kintervals = analyze_dimensionality(comp_atoms)
        except Exception as e:
            warnings.warn(f"Error analyzing dimensionality for component {comp}: {e}. Defaulting to 0D.")
            kintervals = None
        
        if kintervals is None:
            dim_int = 0
        else:
            # Choose the KInterval with the highest score.
            max_k = max(kintervals, key=lambda k: k.score)
            try:
                # Assume kintervals' dimtype is a string like '0D', '1D', etc.
                dim_int = int(max_k.dimtype[0])
            except Exception as e:
                warnings.warn(f"Error converting dimtype '{max_k.dimtype}' to integer: {e}. Defaulting to 0.")
                dim_int = 0
        
        components_dim.setdefault(dim_int, []).append(comp_atoms)
    
    return components_dim


#from collections import Counter
#import re

# Helper function to parse a chemical formula into a Counter of elements.
def parse_formula(formula_str):
    """
    Parse a formula string such as 'H3C' or 'H2CH' into a Counter.
    This ensures that the order of atoms does not affect the match.
    For example, 'H3C' and 'H2CH' both yield Counter({'H': 3, 'C': 1}).
    """
    pattern = r'([A-Z][a-z]*)(\d*)'
    counts = Counter()
    for elem, count in re.findall(pattern, formula_str):
        counts[elem] += int(count) if count else 1
    return counts

def target_molecule_formula_detection(components_atoms, target_dict):
    """
    Scan through a dictionary containing lists of Atoms objects (each representing a connected group)
    and return a dictionary whose keys are the target molecule formulas (as defined in target_dict)
    and whose values are lists of Atoms objects satisfying the target formula.
    
    Parameters:
        components_atoms (dict): Dictionary where each value is a list of ASE Atoms objects.
        target_dict (dict): Dictionary mapping target molecule formulas (str) to their element count (Counter).
    
    Returns:
        dict: Keys are the target formulas (str) and the values are lists of Atoms objects
              that exactly match the target's elemental composition.
    """
    found = {}
    # Loop over each component (ignoring the component id, as requested)
    for atoms_list in components_atoms.values():
        for group in atoms_list:
            # Try to get a chemical formula as a string
            try:
                formula = group.get_chemical_formula()
            except AttributeError:
                formula = str(group.symbols)
            
            group_counter = parse_formula(formula)
            # Compare against each target by checking if the elemental counts match exactly.
            for target, target_counter in target_dict.items():
                if group_counter == target_counter:
                    found.setdefault(target, []).append(group)
    return found
