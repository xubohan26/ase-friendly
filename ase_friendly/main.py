#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
import ase.io
from ase import Atoms


# 1. Find the true physical path of this script, cutting through any symlinks
current_file = Path(__file__).resolve()
# 2. Step up two directories to reach the root 'ase-friendly' folder
root_dir = current_file.parent.parent
# 3. Inject the root directory into Python's search path
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


#import ase_friendly
from ase_friendly.operation_new.sort_atoms_by_element import sort_atoms_by_element
from ase_friendly.operation_new.unwrap_by_bond_connectivity import unwrap_by_bond_connectivity
from ase_friendly.operation_new.translate_center_to import translate_center_to
from ase_friendly.operation_new.standardize_cell import standardize_cell
from ase_friendly.operation_new.randomly_remove_atoms import randomly_remove_atoms
from ase_friendly.operation_new.randomly_replace_atoms import randomly_replace_atoms
from ase_friendly.operation_new.remove_chunk import remove_chunk
from ase_friendly.operation_new.add_noise_to_atoms import add_noise_to_atoms
from ase_friendly.operation_new.reorient_1d_structure import reorient_1d_structure
from ase_friendly.operation_new.fix_atoms_in_or_out_box import fix_atoms_in_or_out_box




# ==========================================
# 1. Input Handler (Records inputs for automation)  
# ==========================================

class InputHandler:
    def __init__(self, silence_mode=False):
        self.silence_mode = silence_mode
        self.history = []

    def get_input(self, prompt_text, helper_func=None):
        """
        Handles user input, runs helpers on empty input, and records valid inputs.
        """
        while True:
            if not self.silence_mode:
                print(prompt_text)
                print(">> ", end='', flush=True)
            
            try:
                line = sys.stdin.readline()
                if not line:  # EOF detected
                    sys.stderr.write("\n[FATAL ERROR] Input stream ended unexpectedly (EOF) while waiting for input.\n")
                    sys.exit(1) 
                user_in = line.strip()
            except KeyboardInterrupt:
                sys.stderr.write("\n[Aborted by user]\n")
                sys.exit(130)

            # Helper logic: if empty and helper exists, run helper and repeat prompt
            # the helper_func might be: press <Enter> to show all operations, or show ls, or show ase supported formats.
            if user_in == "" and helper_func is not None:
                if not self.silence_mode:
                    print("\n--- Helper Output ---")
                    helper_func()
                    print("---------------------\n")
                continue
            
            # Record meaningful input (ignore empty enters if they didn't trigger a helper or if they were just blank)
            # Actually, for automation, we need the actual values provided.
            self.history.append(user_in)
            return user_in

    def print_automation_tip(self):
        if not self.silence_mode and self.history:
            # Construct the printf command string
            # Escape spaces if necessary, though simple for now
            args = " ".join([f'"{x}"' if " " in x else x for x in self.history])
            print("\n" + "="*60)
            print("You could have automated this operation by:")
            print(f"printf '%s\\n' {args} | {os.path.basename(sys.argv[0])} -s  # -s for silence")
            print("="*60 + "\n")

# ==========================================
# 2. Helpers
# ==========================================

def show_ls():
    try:
        files = os.listdir('.')
        for f in sorted(files):
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")

def show_formats():
    keys = sorted(ase.io.formats.all_formats.keys())
    print(" ".join(keys))

# ==========================================
# 3. Operation Definitions
# ==========================================

# Format: (Key, Description, Extra Args Count)
# Extra Args Count = -1 means variable/unknown, handled manually inside execution logic if needed.
# For the new 'translate', we set args to 0 here because we will ask for them interactively one by one. 

OPS_1XX = [
    ("format_conversion", "all ase supported formats", 0),
    ("to_cartesian", "atoms to cartsian coordinate", 0),
    ("to_fractional", "atoms to direct coordinate", 0),
    ("wrap_0_1", "atoms coordinates to [0,1)", 0),
    ("wrap_n0p5_0p5", "atoms coordinates to [-0.5,0.5)", 0),
    ("unwrap_by_bond_connectivity", "wrap positions so bond connectivity is maximally preserved", 0),
    ("sort_atoms_by_element", "same-element-atoms would have adjacent index", 0),
]

OPS_2XX = [
    ("standardize_cell", "cell to 'upper' diagonal or 'lower' 'mid'; 'upper' c only has z_hat", 0), 
    ("translate", "translate ; add (A,B,C) to fractional coordinates", 0), 
    ("translate_center_to", "geo center to fractional or cartisan coordinates", 0), 
    ("reflect", "reflect a cartisean coordinate. option to preserve cell right hand rule/parity or not", 0), 
    ("cell_vec_rearrange_linear_comb", "new cell vectors by linear-combining old", 0), 
    ("cell_vec_rearrange_near_90_degree", "linear-combining cell vec so angles close to 90deg", 0), 
    ("atom_rearrange", "rearrange atom index", 0), 
]

OPS_3XX = [
    ("super_cell", "super cell; sort atoms or not; eg: '2 2 1' true", 0),
    ("super_cell_partial", "super cell, but only fill one slot", 0), 
    ("combine", "combine; sort atoms or not", 0),
]

OPS_4XX = [
    ("change_cell_entries", "change cell entries oversimplified 0 0 2*x+10 means double a1x entry and add 10", 0), 
    ("rotate", "rotate positions in various ways. follow the prompts for details", 0), 
    # old codes
    ("ChangeCell1D", "OldCode; L1 L2 are new length for the 2 other direction", 3), #old
]

OPS_5XX = [#
    ("remove_atoms", "Remove atoms of element; eg, Fe 1 2 4", 0),
    ("randomly_remove_atoms", "Randomly remove elements by num. eg: Li 3 Fe 1", 0),
    ("randomly_replace_atoms", "Randomly replace element with another by num. eg: Si P 2", 0),
    ("remove_chunk", "Remove atoms in the chunk specified by 3 intervals", 0),
    ("add_noise_to_atoms", "add gaussian atom displacements along three axis", 0), 
]

OPS_9XX = [#
    ("remove_velo", "remove atom and cell velocity", 0),
    ("add_velo_by_diff", "add velocity by substracting positions of two structures", 0),
    ("reorient_1d_structure", "eg, rotate 1d structure so (a b_proj c_proj) map to (z_hat x_hat y_hat)", 0), 
    ("fix_atoms_in_or_out_box", "eg, constrain dynamics inside/outside of an input box", 0), 
    ("remove_constraint", "remove atom position constraint", 0),
    # OldCode
    ("AddBiGaus", "OldCode; Add rand(+-)*Gaus(mean,s) to atoms along rotated [1,0,0]", 6), #old
    ("ToNear90ByChangeOneAxis", "OldCode; change cell so one angle is close to 90; 1, 2, or 3 as positional arg", 1), 
]

OPERATIONS = {}
def register_ops(start_idx, op_list):
    idx = start_idx
    for name, desc, n_args in op_list:
        OPERATIONS[str(idx)] = {"name": name, "desc": desc, "n_args": n_args}
        OPERATIONS[name.lower()] = OPERATIONS[str(idx)]
        OPERATIONS[name] = OPERATIONS[str(idx)]
        idx += 1

register_ops(101, OPS_1XX)
register_ops(201, OPS_2XX)
register_ops(301, OPS_3XX)
register_ops(401, OPS_4XX)
register_ops(501, OPS_5XX)
register_ops(901, OPS_9XX)

def print_menu():
    print("="*50)
    # Helper to print sections
    def print_section(title, start, end):
        print(f"\n{title}")
        for k in sorted(OPERATIONS.keys()):
            if k.isdigit() and start < int(k) < end:
                print(f"({k}) {OPERATIONS[k]['name']:<30} {OPERATIONS[k]['desc']}")

    print_section("1xx operations - very trivial (not changing crystal structure)", 100, 200)
    print_section("2xx operations - trivial (not changing crystal structure)", 200, 300)
    print_section("3xx operations - non-trivial (changing crystal structure)", 300, 400)
    print_section("4xx operations - very non-trivial", 400, 500)
    print_section("5xx operations - very very non-trivial", 500, 600)
    print_section("9xx operations - niche", 900, 999)
    print("="*50)

# ==========================================
# 4. Execution Logic
# ==========================================

def run_operation(op_info, input_handler, ase_atoms, input_path, output_path, output_fmt):
    """
    op_info: Dict containing operation metadata
    input_handler: The InputHandler instance to ask for specific args
    ase_atoms: The loaded ASE Atoms object (for new style ops)
    input_path: Raw string path (for old style ops)
    output_path: Raw string path for saving
    output_fmt: String format for saving
    """
    name = op_info['name']
    n_args = op_info['n_args']
    
    # === NEW STYLE OPERATIONS (Using ASE) ===

    #### 1xx
    if name == "format_conversion":
            print('format_conversion is implicitly done for all operations.')
            print('You can just just do to_cartesian, to_fractional, translate, etc, to convert format')
            print('Anyway... I still did the conversion for you :]')
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
            return

    if name == "to_cartesian":
            # ASE uses Cartesian internally. We explicitly force Cartesian write for VASP files.
            is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': False} if is_vasp else {}))
            return

    if name == "to_fractional":
            # ASE uses Cartesian internally. We explicitly force Direct write for VASP files.
            is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
            return

    if name == "wrap_0_1":
        # Wrap positions to [0, 1) relative to the cell # work for .vasp
        ase_atoms.wrap()
        is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
        return

    if name == "wrap_n0p5_0p5":
        # Get scaled positions, wrap to [-0.5, 0.5), then set back # work for .vasp
        scaled = ase_atoms.get_scaled_positions(wrap=False)  # can use wrap=False for faster
        scaled = (scaled + 0.5) % 1 - 0.5
        ase_atoms.set_scaled_positions(scaled)
        is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
        return

    if name == "unwrap_by_bond_connectivity":
        # 2. Parameter Inputs
        g_mult_str = input_handler.get_input('bond cutoff global_multiply (e.g. 1.0): ')
        g_add_str = input_handler.get_input('bond cutoff global_add (e.g. 0.1): ')
        u_near_str = input_handler.get_input('unwrap_near fractional coordinate (e.g. "0.25 0.25 0.25"): ')
        global_multiply = float(g_mult_str)
        global_add = float(g_add_str)
        unwrap_near = [float(x) for x in u_near_str.split()]   
        # fractional_coords = ase_atoms.get_scaled_positions()  #wrap to [0,1] first might reduce bug?
        # wrapped_coords = np.fmod(fractional_coords, 1.0)
        # ase_atoms.set_scaled_positions(wrapped_coords)     
        ase_atoms = unwrap_by_bond_connectivity(
            ase_atoms, 
            global_multiply=global_multiply, 
            global_add=global_add, 
            unwrap_near=unwrap_near, 
            unwrap_near_cartesian=False
        )
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "sort_atoms_by_element":  # untested 
        ase_atoms=sort_atoms_by_element(ase_atoms)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return


    #### 2xx 
    if name == "standardize_cell":
        mode_diagonal_str = input_handler.get_input("Please choose 'upper' 'lower' or 'mid'; upper diagonal makes c along z_hat:")
        ase_atoms = standardize_cell(ase_atoms, mode=mode_diagonal_str)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "translate":
        frac_str = input_handler.get_input("Will input be in fractional coordinate? (true/false, or 1,yes,y,t): ")
        shift_str = input_handler.get_input("translation [x y z] (e.g. '0.5 0.5 0.5'): ")
        shift = [float(x) for x in shift_str.split()]
        is_fractional = frac_str.lower() in ['true', '1', 't', 'yes', 'y']
        cell = ase_atoms.get_cell()
        if is_fractional:
            cart_shift = np.dot(shift, cell)
        else:
            cart_shift = shift
        ase_atoms.translate(cart_shift)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "translate_center_to":
        print('WARNING: you might want to wrap atom position across cell appropriately first')
        frac_str = input_handler.get_input("Will target input be in fractional coordinate? (true/false, or 1,yes,y,t): ")
        target_str = input_handler.get_input("Target position [x y z] (e.g. '0.5 0.5 0.5'): ")
        target_coord = [float(x) for x in target_str.split()]
        is_fractional = frac_str.lower() in ['true', '1', 't', 'yes', 'y']
        ase_atoms = translate_center_to(
            ase_atoms,
            target_position=target_coord,
            target_position_fractional=is_fractional
        )
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "cell_vec_rearrange_linear_comb":
        mat_str = input_handler.get_input("Please enter the new cell matrix (flattened 9 numbers). Eg: '1 0 0  0 1 0  -1 0 1' for [a, b, c-a]: ")
        entries = [float(x) for x in mat_str.split()]
        M = np.array(entries).reshape(3, 3)
        old_cell = ase_atoms.get_cell()
        new_cell = np.dot(M, old_cell)
        ase_atoms.set_cell(new_cell, scale_atoms=False)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "cell_vec_rearrange_near_90_degree":
        ase_atoms=standardize_cell(ase_atoms, mode='lower')
        old_cell = ase_atoms.get_cell()
        def trans( cell, m , n , i , j ):
            out=np.identity( 3, dtype=float)
            out[m,n]=-np.rint( cell[m,n]/cell[i,j] )
            return out
        mat= trans( old_cell, 2, 1, 1, 1 ) @ trans( old_cell, 1, 0, 0, 0 ) @ trans( old_cell, 2, 0, 0, 0 )
        new_cell= mat @ old_cell
        ase_atoms.set_cell(new_cell, scale_atoms=False)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "reflect":
            axis_str = input_handler.get_input("Reflect across plane perpendicular to axis (x, y, or z): ").strip().lower()
            parity_str = input_handler.get_input("Restore right-handed cell parity? (yes/no) [Recommended: yes]: ").strip().lower()
            # parse
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            if axis_str not in axis_map:
                print("Error: Axis must be x, y, or z.")
                sys.exit(1)
            ax_idx = axis_map[axis_str]
            fix_parity = parity_str in ['yes', 'y', 'true', '1']
            # ase
            pos = ase_atoms.get_positions()
            pos[:, ax_idx] *= -1
            ase_atoms.set_positions(pos)
            cell = ase_atoms.get_cell()[:]
            cell[:, ax_idx] *= -1
            if fix_parity:
                # Find the lattice vector (row 0, 1, or 2) most aligned with the reflection axis
                # by checking the absolute value of the components along ax_idx
                aligned_vec_idx = np.argmax(np.abs(cell[:, ax_idx]))
                cell[aligned_vec_idx, :] *= -1
            ase_atoms.set_cell(cell, scale_atoms=False)
            vel = ase_atoms.get_velocities()
            if vel is not None:
                vel[:, ax_idx] *= -1
                ase_atoms.set_velocities(vel)
            ase_atoms.wrap()
            is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
            return

    if name == "atom_rearrange":
            mode_str = input_handler.get_input("Rearrange mode ('seq' for sequential swaps, 'one' for in-one-go mapping that guarentee index locations): ").strip().lower()
            pairs_str = input_handler.get_input("Enter index pairs (1-based). e.g., '1 2 3 4' means 1->2, 3->4: ")
            # parse
            nums = [int(x) - 1 for x in pairs_str.split()]
            if len(nums) % 2 != 0:
                print("Error: You must provide an even number of indices (pairs).")
                sys.exit(1)
            pairs = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            N = len(ase_atoms)
            for s, d in pairs:
                if s < 0 or s >= N or d < 0 or d >= N:
                    print(f"Error: Index out of bounds. Must be between 1 and {N}.")
                    sys.exit(1)
            if mode_str.startswith('seq'):
                # Sequential Swaps (literal swapping of the atoms currently at those slots)
                index_array = list(range(N))
                for pos1, pos2 in pairs:
                    index_array[pos1], index_array[pos2] = index_array[pos2], index_array[pos1]
            elif mode_str.startswith('one'):
                # In-one-go Mapping (Old Position -> New Position)
                mapping = {}
                for s, d in pairs:
                    if s in mapping:
                        print(f"Error: Contradiction! Atom {s+1} is mapped to multiple destinations.")
                        sys.exit(1)
                    mapping[s] = d
                if len(set(mapping.values())) != len(mapping.values()):
                    print("Error: Contradiction! Multiple atoms mapped to the same destination.")
                    sys.exit(1)
                # Pin untouched atoms in place
                sources = set(mapping.keys())
                destinations = set(mapping.values())
                untouched = set(range(N)) - sources - destinations
                for u in untouched:
                    mapping[u] = u
                leftover_src = sorted(list(set(range(N)) - set(mapping.keys())))
                leftover_dest = sorted(list(set(range(N)) - set(mapping.values())))
                for s, d in zip(leftover_src, leftover_dest):
                    mapping[s] = d
                index_array = [0] * N
                for old_pos, new_pos in mapping.items():
                    index_array[new_pos] = old_pos
            else:
                print("Error: Mode must be 'seq' or 'one'.")
                sys.exit(1)
            ase_atoms = ase_atoms[index_array]
            is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
            return

    #### 3xx
    if name == "super_cell":
        rep_str = input_handler.get_input("Please enter supercell multipliers (e.g. '2 2 1'):")
        sort_atoms_bool = input_handler.get_input("Want atoms index sorted by element? true or 1 or... :")
        reps = [int(x) for x in rep_str.split()]
        ase_atoms = ase_atoms.repeat(reps)
        if sort_atoms_bool.lower() in ['true', '1', 't', 'yes', 'y']: ase_atoms=sort_atoms_by_element(ase_atoms)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return
    if name == "super_cell_partial":
        mult_str = input_handler.get_input("Please enter supercell multipliers (e.g. '2 2 1'):")
        slot_str = input_handler.get_input("Please enter slot indices (e.g. '0 1 0' to shift to 2nd slot in y):")
        mult = np.array([int(x) for x in mult_str.split()])
        slot = np.array([int(x) for x in slot_str.split()])
        old_cell = ase_atoms.get_cell()
        shift_vector = np.dot(slot, old_cell)
        ase_atoms.translate(shift_vector)
        new_cell = old_cell * mult[:, np.newaxis]
        ase_atoms.set_cell(new_cell, scale_atoms=False)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        if not input_handler.silence_mode: print('WARNING: If wrong, might because coordinates need to be wrapped first')
        return
    if name == "combine":
        sec_path = input_handler.get_input("Please type secondary input file path. Type only <Enter> to show 'ls':", helper_func=show_ls)
        sec_fmt = input_handler.get_input("Please type secondary input file format. Type only <Enter> to show formats:", helper_func=show_formats)
        sort_atoms_bool = input_handler.get_input("Want atoms index sorted by element? true or 1 or... :")
        second_atoms = ase.io.read(sec_path, format=sec_fmt if sec_fmt else None)
        ase_atoms.extend(second_atoms)
        if sort_atoms_bool.lower() in ['true', '1', 't', 'yes', 'y']: ase_atoms=sort_atoms_by_element(ase_atoms)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    #### 4xx
    if name == "change_cell_entries":
            keep_str = input_handler.get_input("Keep atom coordinates Cartesian or Fractional? (cart/frac): ")
            center_str = input_handler.get_input("Center of wrapping ('origin', 'mid', or 'x y z' in fractional): ")
            args_str = input_handler.get_input("Enter Row Col Xfunc triplets (e.g., '0 0 1.2*x 1 1 x+5'): ")
            # Parse input
            keep_cart = keep_str.lower().startswith('c')
            center_str = center_str.strip().lower()
            if center_str == 'origin':
                center = np.array([0.0, 0.0, 0.0])
            elif center_str == 'mid':
                center = np.array([0.5, 0.5, 0.5])
            else:
                try:
                    center = np.array([float(x) for x in center_str.split()])
                    if len(center) != 3:
                        raise ValueError
                except:
                    print("Error: Center must be 'origin', 'mid', or exactly three numbers.")
                    sys.exit(1)
            args_list = args_str.split()
            if len(args_list) % 3 != 0 or len(args_list) == 0:
                print("Error: Input must be in triplets of (Row, Col, Xfunc).")
                sys.exit(1)
                
            #ase
            scaled = ase_atoms.get_scaled_positions()
            scaled = (scaled - center + 0.5) % 1 - 0.5 + center
            ase_atoms.set_scaled_positions(scaled)
            cell = np.array(ase_atoms.get_cell())
            for i in range(0, len(args_list), 3):
                row, col = int(args_list[i]), int(args_list[i+1])
                expr = args_list[i+2]
                try:
                    # Safely evaluate the math string with 'x' as the variable
                    cell[row, col] = float(eval(expr, {"__builtins__": None}, {"x": cell[row, col]}))
                except Exception as e:
                    print(f"Error evaluating formula '{expr}' for entry [{row}, {col}]: {e}")
                    sys.exit(1)
            ase_atoms.set_cell(cell, scale_atoms=not keep_cart)
            is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
            ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
            return

    if name == "rotate":
            group_str = input_handler.get_input("Center/Grouping for rotation ('none', 'origin', 'mid', 'cart x y z', 'frac x y z'): ")
            axis_str = input_handler.get_input("Rotation axis [x y z] (will be normalized): ")
            angle_str = input_handler.get_input("Rotation angle in degrees: ")
            try:
                # Parse 
                from scipy.spatial.transform import Rotation
                axis = np.array([float(x) for x in axis_str.split()])
                norm = np.linalg.norm(axis)
                if norm == 0:
                    raise ValueError("Rotation axis cannot be a zero vector.")
                axis = axis / norm
                angle_deg = float(angle_str)
                angle_rad = angle_deg * np.pi / 180.0
                rot_vec = axis * angle_rad
                r = Rotation.from_rotvec(rot_vec)
                rot_mat = r.as_matrix()
                
                group_str = group_str.strip().lower()
                cell = ase_atoms.get_cell()
                # Parse Grouping / Center coordinates
                if group_str == 'none':
                    center_cart = np.array([0.0, 0.0, 0.0])
                    do_wrap = False
                else:
                    do_wrap = True
                    if group_str == 'origin':
                        center_frac = np.array([0.0, 0.0, 0.0])
                        center_cart = np.array([0.0, 0.0, 0.0])
                    elif group_str == 'mid':
                        center_frac = np.array([0.5, 0.5, 0.5])
                        center_cart = np.dot(center_frac, cell)
                    elif group_str.startswith('cart'):
                        center_cart = np.array([float(x) for x in group_str.split()[1:]])
                        # Cart to Frac conversion: Frac = Cart @ inv(Cell)
                        center_frac = np.dot(center_cart, np.linalg.inv(cell))
                    elif group_str.startswith('frac'):
                        center_frac = np.array([float(x) for x in group_str.split()[1:]])
                        # Frac to Cart conversion: Cart = Frac @ Cell
                        center_cart = np.dot(center_frac, cell)
                    else:
                        raise ValueError("Invalid grouping format.")
                # ase
                if do_wrap:
                    scaled = ase_atoms.get_scaled_positions()
                    scaled = (scaled - center_frac + 0.5) % 1 - 0.5 + center_frac
                    ase_atoms.set_scaled_positions(scaled)
                pos = ase_atoms.get_positions()
                pos = pos - center_cart
                pos = np.dot(pos, rot_mat.T)
                pos = pos + center_cart
                ase_atoms.set_positions(pos)
                vel = ase_atoms.get_velocities()
                if vel is not None:
                    ase_atoms.set_velocities(np.dot(vel, rot_mat.T))
                is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
                ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
                
            except Exception as e:
                print(f"Error in rotation: {e}")
                sys.exit(1)
            return

    #### 5xx
    if name == "remove_atoms":
        n_str = input_handler.get_input("Please enter Element Index (1-based block):")
        idx_str = input_handler.get_input("Please enter atom indices to remove (eg. 1 2 removes first 2 atoms):")
        nelem = int(n_str) - 1
        idx_rel = [int(x) - 1 for x in idx_str.split()]
        # Identify element block boundaries (e.g., [0, 10, 15] for 10 Al and 5 O)
        syms = ase_atoms.get_chemical_symbols()
        bounds = [0] + [i for i in range(1, len(syms)) if syms[i] != syms[i-1]] + [len(syms)]
        # Calculate global indices to remove based on block start
        start = bounds[nelem]
        global_remove = {start + i for i in idx_rel}
        ase_atoms = ase_atoms[[i for i in range(len(ase_atoms)) if i not in global_remove]] 
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return
    if name == "randomly_remove_atoms":
        remove_str = input_handler.get_input("Please enter elements and counts to remove (e.g. 'Li 11 Fe 1'):")
        seed_str = input_handler.get_input("Please enter random seed (or 'None' for no seed control):")
        seed = None if str(seed_str)=='None' else int(seed_str)
        parts = remove_str.split()
        atoms_to_remove = {parts[i]: int(parts[i+1]) for i in range(0, len(parts), 2)}
        ase_atoms = randomly_remove_atoms(ase_atoms, atoms_to_remove, seed=seed)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return
    if name == "randomly_replace_atoms":
        rep_parts = input_handler.get_input("Enter replacement (e.g., 'Si P 1'):")
        seed_str = input_handler.get_input("Random seed (or 'None'): ")
        parts=rep_parts.split()
        replace_config = [parts[0], parts[1], int(parts[2])]
        # replace_config = ['Si', 'P', 1]
        seed = None if seed_str.lower() == 'none' else int(seed_str)
        ase_atoms = randomly_replace_atoms(ase_atoms, replace_config, seed=seed)
        #ase_atoms=sort_atoms_by_element(ase_atoms) # don't need
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return
    if name == "remove_chunk":
        line_a = input_handler.get_input("Interval for axis a (e.g. '0.25 0.75' or '-0.1 0.1'): ").strip()
        line_b = input_handler.get_input("Interval for axis b: ").strip()
        line_c = input_handler.get_input("Interval for axis c: ").strip()
        parts_a = line_a.split()
        parts_b = line_b.split()
        parts_c = line_c.split()
        intervals = [
            (float(parts_a[0]), float(parts_a[1])),
            (float(parts_b[0]), float(parts_b[1])),
            (float(parts_c[0]), float(parts_c[1]))
        ]
        ase_atoms = remove_chunk(ase_atoms, intervals)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return
    if name == "add_noise_to_atoms":
        std_1_str = input_handler.get_input(r"standard deviation of noise along axis 1 (e.g. 0.4): ")
        axis_1_str = input_handler.get_input(r"axis 1 using a b c (e.g. '1 0 0' means add std_1 along a): ")
        std_2_str = input_handler.get_input(r"standard deviation of noise along axis 2 (e.g. 0.4): ")
        axis_2_str = input_handler.get_input(r"axis 2 using a b c (e.g. '0 1 0' means add std_2 along b_projected): ")
        std_3_str = input_handler.get_input(r"standard deviation of noise along axis 3 orthogonal to axis 1 and 2 (e.g. 0.4): ")
        std_1 = float(std_1_str)
        std_2 = float(std_2_str)
        std_3 = float(std_3_str)
        axis_1 = [float(x) for x in axis_1_str.split()]
        axis_2 = [float(x) for x in axis_2_str.split()]
        ase_atoms = add_noise_to_atoms(ase_atoms, std_1=std_1, axis_1=axis_1, std_2=std_2, axis_2=axis_2, std_3=std_3)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    #9xx
    if name == "remove_velo":  # untested
        ase_atoms.set_velocities(None)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    if name == "add_velo_by_diff":
        sec_path = input_handler.get_input("Please type [secondary input file path]. Type only <Enter> to show 'ls':", helper_func=show_ls)
        sec_fmt = input_handler.get_input("Please type [secondary input file format]. Type only <Enter> to show formats:", helper_func=show_formats)
        atoms_end = ase.io.read(sec_path, format=sec_fmt if sec_fmt else None)
        if ase_atoms.get_chemical_formula() != atoms_end.get_chemical_formula() or \
           not np.array_equal(ase_atoms.get_chemical_symbols(), atoms_end.get_chemical_symbols()):
            print("Error: The element sequences (indices) of the two files do not match.")
            sys.exit(1)
        diff_velo = atoms_end.get_positions() - ase_atoms.get_positions() # not working
        ase_atoms.set_velocities(diff_velo) # not working
        ase.io.vasp.write_vasp(output_path, ase_atoms, direct=True, vasp5=True, ) #no velocities argument 
        # brute force add velocity now
        diff_velo_frac = atoms_end.get_scaled_positions(wrap=False) - ase_atoms.get_scaled_positions(wrap=False)
        with open(output_path, "a") as f:
            f.write("\n")
            for v in diff_velo_frac:
                f.write(f"{v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}\n")
        if not input_handler.silence_mode: print('WARNING: this operation only works with vasp format')
        return

    if name == "reorient_1d_structure":
        ini_axial = input_handler.get_input("Initial axial direction (e.g. 'a'): ")
        fin_dirs_str = input_handler.get_input("Final directions v1 v2 v3 (e.g. 'z x y' will orient the Initial axial direction to z_hat, etc): ")
        lat_consts_str = input_handler.get_input("New lattice constants for secondary axes (e.g. '24.0 16.0'): ")
        fin_dirs = fin_dirs_str.split()
        lat_consts = [float(x) for x in lat_consts_str.split()]
        ase_atoms = reorient_1d_structure(
            atoms=ase_atoms,
            ini_axialcell_d=ini_axial,
            fin_cart_d=fin_dirs[0],
            fin_cart_d_plus_1=fin_dirs[1],
            fin_cart_d_plus_2=fin_dirs[2],
            lattice_constant_d_plus_1=lat_consts[0],
            lattice_constant_d_plus_2=lat_consts[1]
        )
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        print('WARNING: might need to unwrap position across box using unwrap_by_bond_connectivity')
        return

    if name == "fix_atoms_in_or_out_box":
        print("Enter intervals as pairs (e.g. '0.2 0.8' or '-0.1 0.1')")
        line_a = input_handler.get_input("Interval for axis a: ")
        line_b = input_handler.get_input("Interval for axis b: ")
        line_c = input_handler.get_input("Interval for axis c: ")
        fix_mode_str = input_handler.get_input("Fix atoms INSIDE box, instead of outside? (true/false, or 1,yes,y,t): ")
        parts_a = line_a.split()
        parts_b = line_b.split()
        parts_c = line_c.split()
        intervals = [
            (float(parts_a[0]), float(parts_a[1])),
            (float(parts_b[0]), float(parts_b[1])),
            (float(parts_c[0]), float(parts_c[1]))
        ]
        fix_in = fix_mode_str.lower() in ['true', '1', 't', 'yes', 'y']
        is_vasp = output_fmt == 'vasp' or (output_fmt is None and 'POSCAR' in output_path)
        ase_atoms = fix_atoms_in_or_out_box(ase_atoms, intervals, fix_in=fix_in)
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None, **({'direct': True} if is_vasp else {}))
        return

    if name == "remove_constraint":
        # Setting constraint to None/empty removes all constraints (e.g. T T T / F F F flags)
        ase_atoms.set_constraint()
        ase_atoms.write(output_path, format=output_fmt if output_fmt else None)
        return

    # === OLD STYLE OPERATIONS (Legacy Code) ===   remove constraint
    
    # 1. Gather generic extra arguments for old ops if defined in OPS list
    # (Only if n_args > 0 or -1. If 0, no generic prompt is shown)
    extra_args = []
    if n_args != 0:
        prompt_msg = f"Operation '{name}' requires additional parameters."
        if n_args > 0:
            prompt_msg += f" Please enter {n_args} arguments separated by space:"
        else:
            prompt_msg += f" Please enter arguments separated by space:"
            
        args_str = input_handler.get_input(prompt_msg)
        extra_args = args_str.split()
        
        if n_args > 0 and len(extra_args) != n_args:
            print(f"Error: Expected {n_args} arguments, got {len(extra_args)}.")
            sys.exit(1)

    # 2. Initialize Old Object
    try:
        p = ase_friendly.Poscar.FromFile(input_path)
    except Exception as e:
        print(f"Error loading file {input_path} via old method: {e}")
        sys.exit(1)

    # 3. Dispatch to old methods
    # Note: Output is strictly stdout (p.pr()) for now unless specific save logic is added to old code.
    # The user instruction says "Get rid of dummy output", so for old ops we just run p.pr().
    # If you want to capture p.pr() to the output_path, we need redirect_stdout.

    with open(output_path, 'w') if output_path else sys.stdout as f_out:
        # If output_path is provided, we redirect stdout to that file
        # If output_path is empty/None, we print to console (sys.stdout)
        
        # We need to handle the context manager conditionally or just swap sys.stdout
        context = redirect_stdout(f_out) if output_path else redirect_stdout(sys.stdout)
        
        with context:
            # 1xx
            # all converted to new code style
                
            # 2xx
            if name == "TransformCell":
                matdf = pd.read_csv(extra_args[0], delimiter=r'\s+', skipinitialspace=True, engine='python', header=None)
                mat = matdf[0:3].to_numpy()
                p.TransformCell_Matrix(mat)
                p.pr()
            elif name == "ToNear90ByChangeOneAxis":
                p.ToNear90ByChangeOneAxis_Xyz(extra_args[0])
                p.pr()

            # 3xx
            # 4xx

            # 5xx
            # elif name == "RemoveChunk":
            #     p.RemoveChunk_XminXmaxYYZZ(*extra_args)
            #     p.pr()
            elif name == "AddBiGaus":
                p.AddBiGaus_MeanStdDegRxRyRz(*extra_args)
                p.pr()
            elif name == "AddGaus":
                p.AddGaus_MeanStdDegRxRyRz(*extra_args)
                p.pr()
            # elif name == "AddVeloByDiff":
            #     p2 = ase_friendly.Poscar.FromFile(extra_args[0])
            #     p.AddVeloByDiff_TheotherSign(p2, extra_args[1])
            #     p.pr()


# ==========================================
# 5. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='ase-friendly interactive interface')
    parser.add_argument('-s', '--silence', action='store_true', help="Silence mode for automation")
    args = parser.parse_args()
    
    # Initialize Input Handler
    ih = InputHandler(silence_mode=args.silence)

    if not args.silence:
        print("Welcome to [ ase-friendly ]. It manipulates crystal/molecular structures through ASE.")
        print("manually with command-line prompts, or automated through pipping strings to standard input")
        print("Use -s for silence mode during automation\n")

    # A. Select Operation
    prompt_op = r"Please type [operation index or operation name]. Type only <Enter> to see all options"
    op_input = ih.get_input(prompt_op, helper_func=print_menu)
    
    op_data = OPERATIONS.get(op_input)
    if not op_data:
        if not args.silence:
            print("Invalid operation selected. Exiting.")
        sys.exit(1)

    if not args.silence:
        print(f"Selected: {op_data['name']}")

    # B. Primary Input Path
    prompt_in_path = r"Please type primary [input file path]. Type only <Enter> to show 'ls'"
    in_path = ih.get_input(prompt_in_path, helper_func=show_ls)
    
    # C. Primary Input Format
    prompt_in_fmt = r"Please type primary [input file format]. Type only <Enter> to show all ase compatible formats"
    in_fmt = ih.get_input(prompt_in_fmt, helper_func=show_formats)
    
    # Read file using ASE for "New Style" ops (generic read)
    # Even if old ops don't use it, we strictly follow the instruction:
    # "Make the generic read file to read into ase atoms object."
    ase_atoms = None
    try:
        # If format is empty, ase tries to guess
        fmt = in_fmt if in_fmt else None
        ase_atoms = ase.io.read(in_path, format=fmt)
    except Exception as e:
        if not args.silence:
            print(f"Warning!!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  !!!!  ")
            print(f"Warning: ASE failed to read input file '{in_path}': {e}")
            print("Processing will continue, but 'New Style' operations (e.g. translate) will fail if selected.")
        # We don't exit here because Old Style ops read the file themselves later.

    # D. Primary Output Path
    prompt_out_path = r"Please type primary [output file path]. Type only <Enter> to show 'ls'. The output file will be OVERWRITTEN."
    out_path = ih.get_input(prompt_out_path, helper_func=show_ls)

    # E. Primary Output Format
    prompt_out_fmt = r"Please type primary [output file format]. Type only <Enter> to show all ase compatible formats"
    out_fmt = ih.get_input(prompt_out_fmt, helper_func=show_formats)

    # F. Run
    run_operation(op_data, ih, ase_atoms, in_path, out_path, out_fmt)

    # G. Print Automation Tip
    ih.print_automation_tip()

if __name__ == "__main__":
    main()
