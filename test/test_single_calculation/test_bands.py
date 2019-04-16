# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/cp2k/aiida-cp2k        #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""Run simple Band Structure calculation"""

from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
from ase.atoms import Atoms

from aiida.engine import run
from aiida.orm import Code, Dict, StructureData
from aiida.common import NotExistent
from aiida_cp2k.calculations import Cp2kCalculation

# ==============================================================================
if len(sys.argv) != 2:
    print("Usage: test_dft.py <code_name>")
    sys.exit(1)

codename = sys.argv[1]
try:
    code = Code.get_from_string(codename)
except NotExistent:
    print("The code '{}' does not exist".format(codename))
    sys.exit(1)

print("Computing Band Structure of Si...")

# structure
positions = [
    [0.0000000000, 0.0000000000, 2.6954627656],
    [4.0431941484, 4.0431941484, 4.0431941484],
]
cell = [
    [0.0, 2.69546276561, 2.69546276561],
    [2.69546276561, 0.0, 2.69546276561],
    [2.69546276561, 2.69546276561, 0.0],
]
atoms = Atoms("Si2", positions=positions, cell=cell)
structure = StructureData(ase=atoms)

# parameters
parameters = Dict(
    dict={
        "FORCE_EVAL": {
            "METHOD": "Quickstep",
            "DFT": {
                "CHARGE": 0,
                "KPOINTS": {
                    "SCHEME MONKHORST-PACK": "1 1 1",
                    "SYMMETRY": "OFF",
                    "WAVEFUNCTIONS": "REAL",
                    "FULL_GRID": ".TRUE.",
                    "PARALLEL_GROUP_SIZE": 0,
                },
                "MGRID": {"CUTOFF": 600, "NGRIDS": 4, "REL_CUTOFF": 50},
                "UKS": False,
                "BASIS_SET_FILE_NAME": "BASIS_MOLOPT",
                "POTENTIAL_FILE_NAME": "GTH_POTENTIALS",
                "QS": {"METHOD": "GPW", "EXTRAPOLATION": "USE_GUESS"},
                "POISSON": {"PERIODIC": "XYZ"},
                "SCF": {
                    "EPS_SCF": 1.0e-4,
                    "ADDED_MOS": 1,
                    "SMEAR": {"METHOD": "FERMI_DIRAC", "ELECTRONIC_TEMPERATURE": 300},
                    "DIAGONALIZATION": {"ALGORITHM": "STANDARD", "EPS_ADAPT": 0.01},
                    "MIXING": {
                        "METHOD": "BROYDEN_MIXING",
                        "ALPHA": 0.2,
                        "BETA": 1.5,
                        "NBROYDEN": 8,
                    },
                },
                "XC": {"XC_FUNCTIONAL": {"_": "PBE"}},
                "PRINT": {
                    "MO_CUBES": {  # this is to print the band gap
                        "STRIDE": "1 1 1",
                        "WRITE_CUBE": "F",
                        "NLUMO": 1,
                        "NHOMO": 1,
                    },
                    "BAND_STRUCTURE": {
                        "KPOINT_SET": [
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": ["GAMMA 0.0 0.0 0.0", "X 0.5 0.0 0.5"],
                                "UNITS": "B_VECTOR",
                            },
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": [
                                    "X 0.5 0.0 0.5",
                                    "U 0.625 0.25 0.625",
                                ],
                                "UNITS": "B_VECTOR",
                            },
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": [
                                    "K 0.375 0.375 0.75",
                                    "GAMMA 0.0 0.0 0.0",
                                ],
                                "UNITS": "B_VECTOR",
                            },
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": ["GAMMA 0.0 0.0 0.0", "L 0.5 0.5 0.5"],
                                "UNITS": "B_VECTOR",
                            },
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": ["L 0.5 0.5 0.5", "W 0.5 0.25 0.75"],
                                "UNITS": "B_VECTOR",
                            },
                            {
                                "NPOINTS": 10,
                                "SPECIAL_POINT": ["W 0.5 0.25 0.75", "X 0.5 0.0 0.5"],
                                "UNITS": "B_VECTOR",
                            },
                        ]
                    },
                },
            },
            "SUBSYS": {
                "KIND": [
                    {
                        "_": "Si",
                        "BASIS_SET": "DZVP-MOLOPT-SR-GTH",
                        "POTENTIAL": "GTH-PBE-q4",
                    }
                ]
            },
            "PRINT": {  # this is to print forces (may be necessary for problems
                # detection)
                "FORCES": {"_": "ON"}
            },
        },
        "GLOBAL": {"EXTENDED_FFT_LENGTHS": True},  # Needed for large systems
    }
)

options = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
    "max_wallclock_seconds": 1 * 3 * 60,
}
inputs = {
    "structure": structure,
    "parameters": parameters,
    "code": code,
    "metadata": {"options": options},
}
print("submitted calculation...")
calc = run(Cp2kCalculation, **inputs)

bands = calc["output_bands"]

# check bands
expected_gamma_kpoint = np.array(
    [-5.71237757, 6.5718575, 6.5718575, 6.5718575, 8.88653953]
)

if bands.get_kpoints().shape == (66, 3):
    print("OK, got expected kpoints set size.")
else:
    print("Got unexpected kpoints set.")
    sys.exit(3)

if bands.get_bands().shape == (66, 5):
    print("OK, got expected bands set size.")
else:
    print("Got unexpected bands set.")
    sys.exit(3)

if abs(max(bands.get_bands()[0] - expected_gamma_kpoint)) < 1e-7:
    print("Ok, got expected energy levels at GAMMA point.")
else:
    print("Got unexpected energy levels at GAMMA point.")
    sys.exit(3)

sys.exit(0)

# EOF