################################# package description

This package, ase-friendly, is a user-friendly command line prompt-based tool to manipulate crystal structure such as translate, supercell, sort atom by element, make cell matrix upper or lower diagonal, unwrap_by_bond_connectivity, formate_conversion, etc.

The advantage of ase-friendly is that NO document-reading or memorization on how to use the tool, is required. Users will naturally know how to use the tool through prompts. One can choose to either go along the prompts which requires no memorization, or pip the standard input into the code for automation.


#################################  Example: translating atoms

```text
USER: ase-friendly 

ASE-FRIENDLY:
Welcome to [ ase-friendly ]. It manipulates crystal/molecular structures through ASE.
manually with command-line prompts, or automated through pipping strings to standard input
Use -s for silence mode during automation

Please type [operation index or operation name]. Type only <Enter> to see all options

USER: <Enter>

ASE-FRIENDLY:
--- Helper Output ---
==================================================

1xx operations - very trivial (not changing crystal structure)
(101) format_conversion              all ase supported formats
(102) to_cartesian                   atoms to cartsian coordinate
(103) to_fractional                  atoms to direct coordinate
(104) wrap_0_1                       atoms coordinates to [0,1)
(105) wrap_n0p5_0p5                  atoms coordinates to [-0.5,0.5)
(106) unwrap_by_bond_connectivity    wrap positions so bond connectivity is maximally preserved
(107) sort_atoms_by_element          same-element-atoms would have adjacent index

2xx operations - trivial (not changing crystal structure)
(201) standardize_cell               cell to 'upper' diagonal or 'lower' 'mid'; 'upper' c only has z_hat
(202) translate                      translate ; add (A,B,C) to fractional coordinates
...

USER: translate

ASE-FRIENDLY:
Selected: translate
Please type primary [input file path]. Type only <Enter> to show 'ls'

USER: in.vasp

ASE-FRIENDLY:
Please type primary [input file format]. Type only <Enter> to show all ase compatible formats

USER: <ENTER>

ASE-FRIENDLY:
--- Helper Output ---
abinit-gsr abinit-in abinit-out acemolecule-input acemolecule-out aims aims-output bundletrajectory castep-castep castep-cell castep-geom castep-md castep-phonon cfg cif cjson cmdft cp2k-dcd cp2k-restart crystal cube dacapo-text db dftb dlp-history dlp4 dmol-arc dmol-car dmol-incoor elk elk-in eon eps espresso-in espresso-out exciting extxyz findsym gamess-us-in gamess-us-out gamess-us-punch gaussian-in gaussian-out gen gif gpaw-out gpumd gpw gromacs gromos html json jsv lammps-data lammps-dump-binary lammps-dump-text magres mol mp4 mustem mysql netcdftrajectory nomad-json nwchem-in nwchem-out octopus-in onetep-in onetep-out png postgresql pov prismatic proteindatabank py qbox res rmc6f sdf siesta-xv struct struct_out sys traj turbomole turbomole-gradient v-sim vasp vasp-out vasp-xdatcar vasp-xml vti vtu wout x3d xsd xsf xtd xyz

USER: vasp

ASE-FRIENDLY:
Please type primary [output file path]. Type only <Enter> to show 'ls'. The output file will be OVERWRITTEN.

USER: out.vasp

ASE-FRIENDLY:
Please type primary [output file format]. Type only <Enter> to show all ase compatible formats

USER: vasp

ASE-FRIENDLY:
Will input be in fractional coordinate? (true/false, or 1,yes,y,t):

USER: y

ASE-FRIENDLY: translation [x y z] (e.g. '0.5 0.5 0.5'):

USER: 0 0 0.1

ASE-FRIENDLY: 
============================================================
You could have automated this operation by:
printf '%s\n' translate zzin.vasp vasp zzout.vasp vasp y "0 0 0.1" | ase-friendly.py -s  # -s for silence
============================================================

[job finished]
```

########################## installation
### Installation

`ase-friendly` can be installed in a few different ways depending on your experience level and how you manage your Python packages.

#### Option 1: Direct Pip Install (Simplest, but not recommended)
**WARNING:** This method installs the package into your active/global Python environment. This can sometimes cause dependency conflicts over time.

    pip install git+https://github.com/xubohan26/ase-friendly.git

#### Option 2: Segregated Conda Environment (Recommended)
This is the safest method for scientific workflows. It completely isolates the tool and prevents heavy dependencies from messing up your `base` Anaconda system.

    # 1. Create and activate a fresh environment with core dependencies AND pip
    conda create -n ase_env python=3.10 numpy pandas scipy networkx pip
    conda activate ase_env
    # 2. Install ASE via conda-forge (prevents C-library conflicts)
    conda install -c conda-forge ase
    # 3. Verify that the 'pip' command belongs to your new conda environment
    which pip 
    # 4. Install ase-friendly directly from GitHub, using the conda segregated pip
    pip install git+https://github.com/xubohan26/ase-friendly.git

#### Option 3: Conda Base + Direct Executable (Manual Setup)
Use this if you prefer to keep all your scientific packages in your `base` Conda environment and want to avoid using `pip` entirely.

    # PREREQUISITES: You must already have Anaconda (providing numpy, pandas, scipy, networkx) and ASE installed in your base environment.
    # 1. Download the source code
    git clone https://github.com/xubohan26/ase-friendly.git
    # 2. Enter the directory
    cd ase-friendly
    # NOTE: The main executable file is located at ase_friendly/main.py
    # 3. Make the script executable
    chmod +x ase_friendly/main.py
    # 4. Create a local bin directory (if it doesn't exist) and link the executable
    mkdir -p ~/.local/bin
    ln -s $(pwd)/ase_friendly/main.py ~/.local/bin/ase-friendly
    # 5. Add the local bin to your PATH (To make permanent, add to ~/.bashrc)
    export PATH="$HOME/.local/bin:$PATH"