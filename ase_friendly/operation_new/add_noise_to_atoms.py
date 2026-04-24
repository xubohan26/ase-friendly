import numpy as np
from ase import Atoms

def add_noise_to_atoms(atoms, std_1=0.4, axis_1=[0, 0, 1], std_2=0.02, axis_2=[1, 0, 0], std_3=0.02):
    """
    Add random Gaussian displacement to atomic coordinates along specified axes.

    Parameters:
        atoms (ase.Atoms): ASE Atoms object representing the molecule.
        axis_1 (list): Reference axis for std_1 Gaussian noise.
        std_1 (float): Standard deviation for Gaussian noise along axis_1.
        axis_2 (list): Secondary axis for std_2 Gaussian noise after projection.
        std_2 (float): Standard deviation for Gaussian noise along adjusted axis_2.
        std_3 (float): Standard deviation for Gaussian noise along axis_3 (cross product of axis_1 and axis_2).

    Returns:
        atoms (ase.Atoms): Atoms object with modified coordinates.
    """
    # Normalize axes
    axis_1 = np.array(axis_1) / np.linalg.norm(axis_1)
    axis_2 = np.array(axis_2) - np.dot(np.array(axis_2), axis_1) * axis_1  # Project away axis_1 component
    axis_2 /= np.linalg.norm(axis_2)
    axis_3 = np.cross(axis_1, axis_2)
    axis_3 /= np.linalg.norm(axis_3)

    # Construct a transformation matrix for the axes
    transformation_matrix = np.array([axis_1, axis_2, axis_3]).T

    # Get atomic positions
    positions = atoms.get_positions()

    # Generate random displacements
    noise = np.random.normal(0, [std_1, std_2, std_3], positions.shape)

    # Transform the noise to Cartesian coordinates
    noise_cartesian = np.dot(noise, transformation_matrix.T)

    # Add noise to positions
    new_positions = positions + noise_cartesian
    atoms.set_positions(new_positions)

    return atoms

if __name__ == "__main__":
    # Read your molecule from an input file
    from ase.io import read, write
    atoms = read('input.vasp')  # Changeable

    # Add random displacement to atoms
    #atoms = add_noise_to_atoms(atoms, axis_1=[0, 0, 1], std_1=0.4, axis_2=[1, 0, 0], std_2=0.02, std_3=0.02)
    #atoms = add_noise_to_atoms(atoms, axis_1=[ 2.097,   0.0, 3.552], std_1=0.25, axis_2=[1, 0, 0], std_2=0.0, std_3=0.0) #R1 0 1st-rotation
    #atoms = add_noise_to_atoms(atoms, axis_1=[-1.041, 1.816,-3.552], std_1=0.25, axis_2=[1, 0, 0], std_2=0.0, std_3=0.0) #R2 60 1st-rotation
    #atoms = add_noise_to_atoms(atoms, axis_1=[-1.057,-1.816,-3.533], std_1=0.25, axis_2=[1, 0, 0], std_2=0.0, std_3=0.0) #R2 60 1st-rotation

    # Save the modified structure to a file
    write('output.vasp', atoms)

