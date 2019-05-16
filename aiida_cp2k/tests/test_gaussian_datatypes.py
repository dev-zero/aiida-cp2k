# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""Test simple DFT calculations"""

from __future__ import print_function
from __future__ import absolute_import

from io import StringIO

import pytest

from . import get_computer, get_code, gaussian_datatypes_available

pytestmark = pytest.mark.skipif(
    not gaussian_datatypes_available(), reason="Gaussian Datatypes are not available"
)

BSET_INPUT = """\
 H  DZVP-MOLOPT-GTH DZVP-MOLOPT-GTH-q1
 1
 2 0 1 7 2 1
     11.478000339908  0.024916243200 -0.012512421400  0.024510918200
      3.700758562763  0.079825490000 -0.056449071100  0.058140794100
      1.446884268432  0.128862675300  0.011242684700  0.444709498500
      0.716814589696  0.379448894600 -0.418587548300  0.646207973100
      0.247918564176  0.324552432600  0.590363216700  0.803385018200
      0.066918004004  0.037148121400  0.438703133000  0.892971208700
      0.021708243634 -0.001125195500 -0.059693171300  0.120101316500

 O  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q6
 1
 2 0 2 5 2 2 1
     10.389228018317  0.126240722900  0.069215797900 -0.061302037200 -0.026862701100  0.029845227500
      3.849621072005  0.139933704300  0.115634538900 -0.190087511700 -0.006283021000  0.060939733900
      1.388401188741 -0.434348231700 -0.322839719400 -0.377726982800 -0.224839187800  0.732321580100
      0.496955043655 -0.852791790900 -0.095944016600 -0.454266086000  0.380324658600  0.893564918400
      0.162491615040 -0.242351537800  1.102830348700 -0.257388983000  1.054102919900  0.152954188700
"""


@pytest.mark.process_execution
def test_gaussian_basisset_validation(new_workdir):
    """Testing CP2K with the Basis Set stored in gaussian.basisset"""

    import ase.build

    from aiida.engine import run
    from aiida.plugins import CalculationFactory, DataFactory
    from aiida.orm import Dict, StructureData

    computer = get_computer(workdir=new_workdir)
    code = get_code(entry_point="cp2k", computer=computer)

    # structure
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=2.0)
    structure = StructureData(ase=atoms)

    BasisSet = DataFactory("gaussian.basisset")

    fhandle = StringIO(BSET_INPUT)

    bsets = {b.element: b for b in BasisSet.from_cp2k(fhandle)}

    # parameters
    parameters = Dict(
        dict={
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "QS": {
                        "EPS_DEFAULT": 1.0e-12,
                        "WF_INTERPOLATION": "ps",
                        "EXTRAPOLATION_ORDER": 3,
                    },
                    "MGRID": {"NGRIDS": 4, "CUTOFF": 280, "REL_CUTOFF": 30},
                    "XC": {"XC_FUNCTIONAL": {"_": "LDA"}},
                    "POISSON": {"PERIODIC": "none", "PSOLVER": "MT"},
                },
                "SUBSYS": {
                    "KIND": [
                        {
                            "_": "O",
                            "POTENTIAL": "GTH-LDA-q6",
                            "BASIS_SET": bsets["O"].name,
                        },
                        {
                            "_": "H",
                            "POTENTIAL": "GTH-LDA-q1",
                            "BASIS_SET": bsets["H"].name,
                        },
                    ]
                },
            }
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
        "basissets": {
            "FORCE_EVAL_0": {
                element: {"ORB_0": bset} for element, bset in bsets.items()
            }
        },
    }

    run(CalculationFactory("cp2k"), **inputs)


@pytest.mark.process_execution
def test_gaussian_basisset_validation_failure(new_workdir):
    """Testing CP2K with the Basis Set stored in gaussian.basisset but missing"""

    import ase.build

    from aiida.engine import run
    from aiida.plugins import CalculationFactory, DataFactory
    from aiida.orm import Dict, StructureData
    from aiida.common.exceptions import InputValidationError

    computer = get_computer(workdir=new_workdir)
    code = get_code(entry_point="cp2k", computer=computer)

    # structure
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=2.0)
    structure = StructureData(ase=atoms)

    BasisSet = DataFactory("gaussian.basisset")

    fhandle = StringIO(BSET_INPUT)
    bsets = {b.element: b for b in BasisSet.from_cp2k(fhandle)}

    # parameters
    parameters = Dict(
        dict={
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {
                    "QS": {
                        "EPS_DEFAULT": 1.0e-12,
                        "WF_INTERPOLATION": "ps",
                        "EXTRAPOLATION_ORDER": 3,
                    },
                    "MGRID": {"NGRIDS": 4, "CUTOFF": 280, "REL_CUTOFF": 30},
                    "XC": {"XC_FUNCTIONAL": {"_": "LDA"}},
                    "POISSON": {"PERIODIC": "none", "PSOLVER": "MT"},
                },
                "SUBSYS": {
                    "KIND": [
                        {
                            "_": "O",
                            "POTENTIAL": "GTH-LDA-q6",
                            "BASIS_SET": bsets["O"].name,
                        },
                        {
                            "_": "H",
                            "POTENTIAL": "GTH-LDA-q1",
                            "BASIS_SET": bsets["H"].name,
                        },
                    ]
                },
            }
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
        # add only one of the basis sets to inputs
        "basissets": {"FORCE_EVAL_0": {"H": {"ORB_0": bsets["H"]}}},
    }

    with pytest.raises(InputValidationError):
        run(CalculationFactory("cp2k"), **inputs)
