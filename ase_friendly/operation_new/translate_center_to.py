import numpy as np
from ase import Atoms
from ase.io import read, write

def translate_center_to(atoms, target_position=[0,0,0], target_position_fractional=True):
    """
    Calculates the geometric center and shifts it to a target position.

    The target position can be specified in either Cartesian or fractional
    coordinates. This function computes the mean of the Cartesian coordinates
    of all atoms and translates the structure so this center moves to the target.

    Parameters:
        atoms (ase.Atoms): The input ASE Atoms object.
        target_position (list or np.ndarray, optional): A 3-element list or array
            representing the target coordinates. If None, defaults to the
            origin [0, 0, 0].
        target_position_fractional (bool, optional): If True (default), the
            `target_position` is treated as fractional coordinates relative to
            the cell vectors. If False, it's treated as Cartesian coordinates.

    Returns:
        ase.Atoms: A new Atoms object shifted to the new position.
    """
    # Create a copy to avoid modifying the original Atoms object
    modified_atoms = atoms.copy()

    # Determine the target position in absolute Cartesian coordinates
    if target_position is None:
        target_cartesian = np.array([0.0, 0.0, 0.0])
    else:
        target_position = np.array(target_position)
        if target_position.shape != (3,):
            raise ValueError("target_position must be a 3-element list or NumPy array.")

        if target_position_fractional:
            print(f"Interpreting target {target_position} as fractional coordinates.")
            # Convert fractional target to Cartesian using the cell matrix
            cell = modified_atoms.get_cell()
            target_cartesian = np.dot(target_position, cell)
        else:
            print(f"Interpreting target {target_position} as Cartesian coordinates.")
            target_cartesian = target_position

    # Calculate the geometric center (mean of all atomic positions in Cartesian)
    geometric_center = np.mean(modified_atoms.get_positions(), axis=0)
    
    # print(f"Original geometric center (Cartesian): {geometric_center}")
    # print(f"Target position (Cartesian):         {target_cartesian}")

    # Calculate the vector needed to shift the center to the target
    shift_vector = target_cartesian - geometric_center
    
    # print(f"Calculated shift vector:               {shift_vector}")

    # Apply the shift to all atoms in the copied object
    modified_atoms.translate(shift_vector)
    
    # Verify the new center for confirmation
    new_center = np.mean(modified_atoms.get_positions(), axis=0)
    # print(f"New geometric center (Cartesian):      {new_center}")

    return modified_atoms

if __name__ == "__main__":
    print('WARNING: you might want to wrap atom position across cell appropriately first')
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()
    target_str = input("Target position [x y z] (e.g. '0.5 0.5 0.5'): ").strip()
    frac_str = input("Is target fractional? (True/False): ").strip()
    target_coord = [float(x) for x in target_str.split()]
    is_fractional = frac_str.lower() in ['true', '1', 't', 'yes']
    atoms = read(file_in, format=file_fmt)
    atoms_shifted = translate_center_to(
        atoms,
        target_position=target_coord,
        target_position_fractional=is_fractional
    )
    atoms_shifted.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')
