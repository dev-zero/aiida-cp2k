# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel
"""Base workchain command"""

import click

from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators, echo

from .base import launch_process
from . import cmd_launch
from ..utils import cp2k_options


@cmd_launch.command('eos')
@options.CODE(required=True, type=types.CodeParamType(entry_point="cp2k"))
@click.argument("cp2k-input-file", type=click.File("r"), required=True)
@click.argument("structures", type=types.DataParamType(sub_classes=('aiida.data:structure',)), required=True, nargs=-1)
@cp2k_options.DAEMON()
@decorators.with_dbenv()
def cmd_launch_workflow(code, daemon, cp2k_input_file, structures):
    """Run an Equation-of-States workflow with the given input and structures"""

    from aiida.orm import Dict
    from aiida.plugins import WorkflowFactory
    from cp2k_input_tools.parser import CP2KInputParserSimplified

    parser = CP2KInputParserSimplified(key_trafo=str.upper)
    tree = parser.parse(cp2k_input_file)

    if tree["FORCE_EVAL"]["SUBSYS"].pop("COORD", None):
        echo.echo_warning("any structures specified in the input file will be ignored")

    builder = WorkflowFactory('cp2k.eos').get_builder()

    builder.cp2k.structure = structures[0]
    builder.structures = {f"struct_{i}": s for i, s in enumerate(structures[1:])}
    builder.cp2k.code = code
    builder.cp2k.metadata.options = {
        "resources": {
            "num_machines": 1,
        },
    }
    builder.cp2k.parameters = Dict(dict=tree)

    launch_process(builder, daemon)
