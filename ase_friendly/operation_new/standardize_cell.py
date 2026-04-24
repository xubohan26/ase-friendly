import sys
import numpy as np
from ase.io import read, write

import numpy as np

def cell_matrix_to_parameters(cell):
    """
    Extracts lengths (a, b, c) and angles (alpha, beta, gamma) from a cell matrix.
    Refactored version of CellMatToCellParameters_2dlist.
    """
    cell = np.array(cell, dtype=float)
    v1, v2, v3 = cell[0], cell[1], cell[2]
    
    # Lengths
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    n3 = np.linalg.norm(v3)
    
    # Angles (in degrees)
    # Clip dot products to [-1, 1] to prevent NaN errors from floating point noise
    dot23 = np.clip(np.dot(v2/n2, v3/n3), -1.0, 1.0)
    dot31 = np.clip(np.dot(v3/n3, v1/n1), -1.0, 1.0)
    dot12 = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
    
    alpha = np.degrees(np.arccos(dot23))
    beta  = np.degrees(np.arccos(dot31))
    gamma = np.degrees(np.arccos(dot12))
    
    # Return as numpy array: [a, b, c, alpha, beta, gamma]
    return np.array([n1, n2, n3, alpha, beta, gamma])

def cell_parameters_to_matrix(params):
    """
    Constructs the Lower Triangular cell matrix from parameters.
    Refactored version of CellParamtersToCellMat_List (preserving exact math).
    
    Output Format (Lower):
    a is along X.
    b is in XY plane.
    """
    a, b, c, alpha, beta, gamma = params
    
    # Convert degrees to radians for calculation
    # Using your original logic: (angle - 90)
    alphaM90 = np.radians(alpha - 90)
    betaM90  = np.radians(beta - 90)
    gammaM90 = np.radians(gamma - 90)
    
    # --- Vector 1 (a) ---
    # Aligned with X axis
    v1 = [a, 0.0, 0.0]
    
    # --- Vector 2 (b) ---
    # Your algorithm: [-b*sin(g-90), b*cos(g-90), 0]
    # Note: sin(x-90) = -cos(x), cos(x-90) = sin(x)
    # This results in [b*cos(gamma), b*sin(gamma), 0]
    v2_x = -b * np.sin(gammaM90)
    v2_y =  b * np.cos(gammaM90)
    v2 = [v2_x, v2_y, 0.0]
    
    # --- Vector 3 (c) ---
    # Calculate v32 (y-component of c)
    # Your math: c * (cos(alpha-90+90) - sin(gamma-90)*sin(beta-90)) / cos(gamma-90)
    # Simplified: c * (cos(alpha) - cos(gamma)*cos(beta)) / sin(gamma)
    numerator = np.cos(alphaM90 + np.pi/2) - np.sin(gammaM90) * np.sin(betaM90)
    denominator = np.cos(gammaM90)
    v32 = c * numerator / denominator
    
    # Calculate v31 (x-component of c)
    # Your math: -c * sin(beta-90)  => c * cos(beta)
    v31 = -c * np.sin(betaM90)
    
    # Calculate v33 (z-component of c)
    # Your math: sqrt( (c*cos(beta-90))^2 - v32^2 )
    term_c_beta = c * np.cos(betaM90)
    v33 = np.sqrt(np.square(term_c_beta) - np.square(v32))
    
    v3 = [v31, v32, v33]
    
    return np.array([v1, v2, v3])

def triangular_diagonalized_cell(cell, mode='lower'):
    """
    Transforms a crystal cell matrix into a specific form using the tested
    reconstruction algorithm.
    
    Parameters:
    -----------
    cell : array_like (3x3)
    mode : str ('lower', 'upper', 'mid')
    """
    # 1. Extract invariant parameters from the input cell
    # params = [a, b, c, alpha, beta, gamma]
    params = cell_matrix_to_parameters(cell)
    
    if mode == 'lower':
        # Default behavior of your algorithm:
        # a || x, b in xy plane.
        return cell_parameters_to_matrix(params)
        
    elif mode == 'mid':
        # Goal: b || y, a in xy plane.
        # Strategy: Swap a/b parameters, generate Lower, then swap X/Y axes.
        
        # Swap lengths (a <-> b) and corresponding angles (alpha <-> beta)
        # Original: [a, b, c, alpha, beta, gamma]
        # Swapped:  [b, a, c, beta, alpha, gamma]
        mid_params = params.copy()
        mid_params[0], mid_params[1] = params[1], params[0] # Swap a, b
        mid_params[3], mid_params[4] = params[4], params[3] # Swap alpha, beta
        
        # Generate Lower matrix (now 'b' is on X, 'a' is in XY)
        matrix = cell_parameters_to_matrix(mid_params)
        
        # Swap X and Y axes (Columns 0 and 1) to put 'b' on Y
        matrix = matrix[:, [1, 0, 2]]
        
        # Swap Row 0 and Row 1 to restore vector order [a, b, c]
        matrix = matrix[[1, 0, 2], :]
        
        return matrix
        
    elif mode == 'upper':
        # Goal: c || z, b in yz plane.
        # Strategy: Map (c->a, b->b, a->c). Generate Lower. Swap axes (x->z).
        
        # Permutation: a(old)->c(new), b(old)->b(new), c(old)->a(new)
        # Lengths: [c, b, a]
        # Angles:  [gamma, beta, alpha] 
        # (Angle between new b/c (old b/a) is gamma)
        
        upper_params = np.zeros(6)
        upper_params[0] = params[2] # New a = old c
        upper_params[1] = params[1] # New b = old b
        upper_params[2] = params[0] # New c = old a
        
        upper_params[3] = params[5] # New alpha (b-c) = old gamma (b-a)
        upper_params[4] = params[4] # New beta  (a-c) = old beta  (c-a)
        upper_params[5] = params[3] # New gamma (a-b) = old alpha (c-b)
        
        # Generate Lower matrix (now 'c' is on X, 'b' is in XY)
        matrix = cell_parameters_to_matrix(upper_params)
        
        # Swap X and Z axes (Columns 0 and 2) to put 'c' on Z
        matrix = matrix[:, [2, 1, 0]]
        
        # Swap Row 0 and Row 2 to restore vector order [a, b, c]
        matrix = matrix[[2, 1, 0], :]
        
        return matrix


def standardize_cell(atoms, mode='lower'):
    """
    Standardizes the unit cell of the Atoms object by calculating the 
    transformed cell matrix and applying it while fixing fractional coordinates.
    """
    old_cell = atoms.get_cell()
    new_cell_matrix = triangular_diagonalized_cell(old_cell, mode=mode)
    atoms.set_cell(new_cell_matrix, scale_atoms=True)
    
    return atoms

if __name__ == "__main__":
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()
    file_out = input('file_output: ').strip()
    out_fmt = input('output_format: ').strip()

    atoms = read(file_in, format=file_fmt)
    mode_str = input("Enter mode (upper/lower/mide): ").strip()

    atoms = standardize_cell(atoms, mode=mode_str)
    
    atoms.write(file_out, format=out_fmt)
    print(f'Standardized cell ({mode_str}) saved to {file_out}')