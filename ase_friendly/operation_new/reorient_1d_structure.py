import numpy as np
from ase import Atoms
from ase.io import read, write

def reorient_1d_structure(
    atoms,
    ini_axialcell_d,
    fin_cart_d,
    fin_cart_d_plus_1,
    fin_cart_d_plus_2,
    lattice_constant_d_plus_1,
    lattice_constant_d_plus_2,
):
    """
    Reorients a 1D structure by performing a full basis transformation of the
    simulation cell and atomic coordinates.

    This function defines an initial basis from the input cell vectors (applying
    Gram-Schmidt for orthogonality) and a target basis from the desired final
    Cartesian axes. It then computes a transformation matrix to map the atomic
    coordinates from the initial to the target orientation. The final cell is
    constructed to be diagonal, with dimensions set by the original material's
    axis length and the new specified lattice constants.

    Parameters:
        atoms (ase.Atoms): The input ASE Atoms object.
        ini_axialcell_d (str): The initial cell vector to be reoriented ('a', 'b', or 'c').
        fin_cart_d (str): The final Cartesian direction for the primary axis ('x', 'y', or 'z').
        fin_cart_d_plus_1 (str): The final Cartesian direction for the second axis.
        fin_cart_d_plus_2 (str): The final Cartesian direction for the third axis.
        lattice_constant_d_plus_1 (float): The new lattice constant for the second axis.
        lattice_constant_d_plus_2 (float): The new lattice constant for the third axis.

    Returns:
        ase.Atoms: A new Atoms object with the reoriented structure and modified cell.
    """
    # Create a copy to avoid modifying the original object
    modified_atoms = atoms.copy()
    initial_cell = modified_atoms.get_cell()

    # --- 1. Map string inputs to indices ---
    dir_map = {'a': 0, 'b': 1, 'c': 2, 'x': 0, 'y': 1, 'z': 2}
    
    # Validate inputs
    unique_final_dirs = {fin_cart_d.lower(), fin_cart_d_plus_1.lower(), fin_cart_d_plus_2.lower()}
    if len(unique_final_dirs) != 3 or not all(d in dir_map for d in unique_final_dirs):
        raise ValueError("Final Cartesian directions must be a unique set of 'x', 'y', and 'z'.")

    # --- 2. Define the initial ordered basis vectors ---
    idx_map = [dir_map[ini_axialcell_d.lower()]]
    idx_map.extend([i for i in [0, 1, 2] if i not in idx_map])

    v_orig_1 = initial_cell[idx_map[0]]
    v_orig_2 = initial_cell[idx_map[1]]
    v_orig_3 = initial_cell[idx_map[2]]
    
    original_length_v1 = np.linalg.norm(v_orig_1)

    # --- 3. Create an orthonormal initial basis (U) using Gram-Schmidt ---
    u1 = v_orig_1 / np.linalg.norm(v_orig_1)
    
    u2 = v_orig_2 - np.dot(v_orig_2, u1) * u1
    u2 /= np.linalg.norm(u2)
    
    u3 = v_orig_3 - np.dot(v_orig_3, u1) * u1 - np.dot(v_orig_3, u2) * u2
    u3 /= np.linalg.norm(u3)
    
    U = np.array([u1, u2, u3])

    # --- 4. Define the target orthonormal basis (V) ---
    cart_basis = np.eye(3)
    v1 = cart_basis[dir_map[fin_cart_d.lower()]]
    v2 = cart_basis[dir_map[fin_cart_d_plus_1.lower()]]
    v3 = cart_basis[dir_map[fin_cart_d_plus_2.lower()]]
    
    V = np.array([v1, v2, v3])

    # --- 5. Calculate and apply the transformation matrix ---
    # The transformation matrix M that maps coordinates from basis U to basis V
    # is M = U_inverse * V. Since U is orthonormal, U_inverse = U_transpose.
    # For row vectors (ASE positions), the operation is pos_new = pos * M.
    # The matrix M is U.T @ V.
    transformation_matrix = U.T @ V
    modified_atoms.positions @= transformation_matrix

    # --- 6. Construct the new diagonal cell ---
    new_cell = np.zeros((3, 3))
    
    # Create the scaled vectors for the new cell
    scaled_v1 = original_length_v1 * v1
    scaled_v2 = lattice_constant_d_plus_1 * v2
    scaled_v3 = lattice_constant_d_plus_2 * v3
    
    # Map the final Cartesian directions to the newly scaled vectors
    vec_map = {
        fin_cart_d.lower(): scaled_v1,
        fin_cart_d_plus_1.lower(): scaled_v2,
        fin_cart_d_plus_2.lower(): scaled_v3,
    }

    # Assign vectors to make the final cell diagonal
    new_cell[dir_map['x']] = vec_map['x']
    new_cell[dir_map['y']] = vec_map['y']
    new_cell[dir_map['z']] = vec_map['z']
    
    modified_atoms.set_cell(new_cell)

    return modified_atoms

if __name__ == "__main__":
    # 1. File Inputs
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()

    # 2. Parameter Inputs
    ini_axial = input("Initial axial direction (e.g. 'a'): ").strip()
    fin_dirs_str = input("Final directions v1 v2 v3 (e.g. 'z x y' will orient the Initial axial direction to z_hat, etc): ").strip()
    lat_consts_str = input("New lattice constants for secondary axes (e.g. '24.0 16.0'): ").strip()

    # 3. Format Inputs
    fin_dirs = fin_dirs_str.split()
    lat_consts = [float(x) for x in lat_consts_str.split()]

    # 4. Process and Save
    ase_atoms = read(file_in, format=file_fmt)

    new_atoms = reorient_1d_structure(
        atoms=ase_atoms,
        ini_axialcell_d=ini_axial,
        fin_cart_d=fin_dirs[0],
        fin_cart_d_plus_1=fin_dirs[1],
        fin_cart_d_plus_2=fin_dirs[2],
        lattice_constant_d_plus_1=lat_consts[0],
        lattice_constant_d_plus_2=lat_consts[1]
    )

    new_atoms.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')