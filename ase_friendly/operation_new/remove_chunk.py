import numpy as np
from ase.io import read, write

def remove_chunk(atoms, intervals):
    """
    Removes atoms inside a specified fractional coordinate box using
    a shift-and-wrap method to handle boundary conditions.
    
    intervals: list of tuples [(min_a, max_a), (min_b, max_b), (min_c, max_c)]
    """
    modified_atoms = atoms.copy()
    frac_coords = modified_atoms.get_scaled_positions()
    n_atoms = len(modified_atoms)
    
    # Start assuming all atoms are inside the removal box (True),
    # then narrow it down axis by axis.
    inside_mask = np.ones(n_atoms, dtype=bool)

    for i in range(3):
        start, end = intervals[i]
        
        # 1. Calculate the target width of the interval
        # (end - start) % 1.0 correctly handles wrap-around (e.g. -0.2 to 0.2 -> width 0.4)
        width = (end - start) % 1.0
        
        # 2. Shift the coordinate so the interval starts at 0.0
        shifted = frac_coords[:, i] - start
        
        # 3. Wrap the shifted coordinate to [0, 1)
        wrapped = shifted % 1.0
        
        # 4. Check if the atom falls within the [0, width) range
        dim_mask = wrapped < width
        
        # Combine with the main mask (Logical AND)
        inside_mask &= dim_mask

    # Delete atoms that fell inside the box for all 3 dimensions
    del modified_atoms[inside_mask]
    
    return modified_atoms

if __name__ == "__main__":
    # 1. File Inputs
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()
    line_a = input("Interval for axis a (e.g. '0.25 0.75' or '-0.1 0.1'): ").strip()
    line_b = input("Interval for axis b: ").strip()
    line_c = input("Interval for axis c: ").strip()
    parts_a = line_a.split()
    parts_b = line_b.split()
    parts_c = line_c.split()
    intervals = [
        (float(parts_a[0]), float(parts_a[1])),
        (float(parts_b[0]), float(parts_b[1])),
        (float(parts_c[0]), float(parts_c[1]))
    ]
    ase_atoms = read(file_in, format=file_fmt)
    new_atoms = remove_chunk(ase_atoms, intervals)
    new_atoms.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')