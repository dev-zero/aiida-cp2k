# -*- coding: utf-8 -*-

"""Command line interface script to launch CP2K workchains"""

from __future__ import absolute_import
import re

import click

from aiida.cmdline.params import types, options
from aiida.cmdline.utils import decorators

from . import calculations


@calculations.command("oneshot")
@options.CODE(required=True, type=types.CodeParamType(entry_point="cp2k"))
@click.argument(
    "files", nargs=-1, type=click.Path(exists=True, resolve_path=True), required=True
)
@click.option(
    "-m",
    "--max-num-machines",
    type=click.INT,
    default=1,
    show_default=True,
    help="The maximum number of machines (compute nodes)",
)
@click.option(
    "-p",
    "--max-num-mpiprocs-per-machine",
    type=click.INT,
    default=1,
    show_default=True,
    help="The maximum number of MPI processes per machine",
)
@click.option(
    "-w",
    "--max-wallclock-seconds",
    type=click.INT,
    default=1800,
    show_default=True,
    help="the maximum wallclock time in seconds",
)
@decorators.with_dbenv()
def oneshot(
    code, files, max_num_machines, max_num_mpiprocs_per_machine, max_wallclock_seconds
):
    """Run a one-shot CP2K workchain with the given input file (first) and additional data files"""

    from aiida.orm import SinglefileData, Dict
    from aiida.engine import launch

    datafiles = [SinglefileData(file=f) for f in files]
    cp2k_input = datafiles[0]

    def safe_linkname(fname):
        return re.sub("[^0-9a-zA-Z_]+", "_", fname)

    from aiida.plugins import CalculationFactory

    inputs = {
        "code": code,
        "parameters": Dict(),
        "file": {safe_linkname(f.filename): f for f in datafiles},
        "metadata": {
            "options": {
                "input_filename": cp2k_input.filename,
                "resources": {
                    "num_machines": max_num_machines,
                    "num_mpiprocs_per_machine": max_num_mpiprocs_per_machine,
                },
                "max_wallclock_seconds": max_wallclock_seconds,
            }
        },
    }

    click.echo("Running CP2K calculation...")
    _, node = launch.run_get_node(CalculationFactory("cp2k"), **inputs)
