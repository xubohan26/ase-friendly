import numpy as np
from ase import Atoms
from ase.io import read, write
import sys
from ase.formula import Formula

def sort_atoms_by_element(image):
    copied = image.copy()
    atoms = Atoms()
    atoms.cell = copied.cell
    w = copied.symbols
    formula = Formula(f'{w}')
    count = list(formula.count())
    for _ , symbol in enumerate(count):
        for i in range(len(copied)):
            if copied[i].symbol == symbol:
                atoms.append(copied[i])
    return atoms

if __name__ == "__main__":
    file_input = input('file_input: ').strip()
    file_format = input('file_format: ').strip()
    ase_atoms = read(file_input, format=file_format)
    ase_atoms = sort_atoms_by_element(ase_atoms)
    ase_atoms.write('zz_out.vasp', format='vasp')
    print('saved to zz_out.vasp')