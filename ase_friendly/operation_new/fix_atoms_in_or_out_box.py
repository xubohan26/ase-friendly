import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms

def fix_atoms_in_or_out_box(atoms, intervals, fix_in=True):
    """
    Applies FixAtoms constraint to atoms either inside or outside a specified 
    fractional coordinate box. Handles boundary wrapping (e.g. -0.1 to 0.1).
    
    Parameters:
        atoms (ase.Atoms): The structure to modify.
        intervals (list): List of tuples [(min_a, max_a), (min_b, max_b), (min_c, max_c)].
        fix_in (bool): If True, fix atoms INSIDE the box. 
                       If False, fix atoms OUTSIDE the box.
    """
    # Create a copy so we don't mutate the original if we don't want to
    # (Though typically constraints are applied in-place, returning a copy is safer for pipelines)
    modified_atoms = atoms.copy()
    
    frac_coords = modified_atoms.get_scaled_positions()
    n_atoms = len(modified_atoms)
    
    # Start assuming all atoms are "inside" until proven otherwise
    inside_mask = np.ones(n_atoms, dtype=bool)

    for i in range(3):
        start, end = intervals[i]
        
        # 1. Calculate effective width (handles wrap-around)
        width = (end - start) % 1.0
        
        # 2. Shift coordinates so interval starts at 0.0
        shifted = frac_coords[:, i] - start
        
        # 3. Wrap to [0, 1)
        wrapped = shifted % 1.0
        
        # 4. Check if atom falls within [0, width)
        dim_mask = wrapped < width
        
        # Combine with main mask (Logical AND)
        inside_mask &= dim_mask

    # Determine which indices to fix
    if fix_in:
        # Fix atoms that are inside the box
        indices_to_fix = np.where(inside_mask)[0]
    else:
        # Fix atoms that are OUTSIDE the box (invert mask)
        indices_to_fix = np.where(~inside_mask)[0]
        
    # Apply constraint
    if len(indices_to_fix) > 0:
        constraint = FixAtoms(indices=indices_to_fix)
        modified_atoms.set_constraint(constraint)
    else:
        print("Warning: No atoms matched the criteria. No constraints applied.")

    return modified_atoms

if __name__ == "__main__":
    # 1. File Inputs
    file_in = input('file_input: ').strip()
    file_fmt = input('file_format: ').strip()

    print("Enter intervals as pairs (e.g. '0.2 0.8' or '-0.1 0.1')")
    line_a = input("Interval for axis a: ").strip()
    line_b = input("Interval for axis b: ").strip()
    line_c = input("Interval for axis c: ").strip()
    fix_mode_str = input("Fix atoms INSIDE box, instead of outside? (true/false, or 1,yes,y,t): ").strip()
    parts_a = line_a.split()
    parts_b = line_b.split()
    parts_c = line_c.split()
    intervals = [
        (float(parts_a[0]), float(parts_a[1])),
        (float(parts_b[0]), float(parts_b[1])),
        (float(parts_c[0]), float(parts_c[1]))
    ]
    fix_in = fix_mode_str.lower() in ['true', '1', 't', 'yes','y']


    ase_atoms = read(file_in, format=file_fmt)
    new_atoms = fix_atoms_in_or_out_box(ase_atoms, intervals, fix_in=fix_in)
    
    new_atoms.write('zz_out.vasp', format='vasp')
    print('Saved to zz_out.vasp')