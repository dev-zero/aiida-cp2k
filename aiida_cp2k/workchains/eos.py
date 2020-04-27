# -*- coding: utf-8 -*-
"""Work chain to obtain Burch-Murnaghan EOS parameters from CP2K."""

import operator as op
import itertools

from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.plugins import CalculationFactory
from aiida import orm

from .base import Cp2kBaseWorkChain

Cp2kCalculation = CalculationFactory('cp2k')  # pylint: disable=invalid-name


@calcfunction
def extract_singlepoint_data(output_parameters):
    print(f"{output_parameters}")
    return orm.Dict(dict={"energy": output_parameters.get_dict().get("energy", None)})


@calcfunction
def ev_curve(**ev):
    return orm.Dict(dict={l: v["energy"] for l, v in ev.items()})


class Cp2kEosWorkChain(WorkChain):
    """Workflow to run EOS calculations with CP2K"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # expose the base work chain parameters, but not a structure since we're going to override that
        spec.expose_inputs(Cp2kBaseWorkChain)
        spec.input_namespace('structures',
                             dynamic=True,
                             required=True,
                             help="Explicit list of structures to use for calculation")
        spec.output('ev', valid_type=orm.Dict, required=True, help="The raw E(V) data")
        spec.output('eos_params',
                    valid_type=orm.Dict,
                    required=False,
                    help="The Birch-Murnaghan EOS parameters if the fit was successful")

        spec.outline(cls.setup, cls.run_calculations, cls.results)

    def setup(self):
        self.ctx.ref_cell_vol = self.inputs.cp2k.structure.get_cell_volume()

        # list of (cell_fraction, label, structure) sorted by cell_fraction
        self.ctx.structures = sorted(itertools.chain(
            [(1.0, "ref_struct", self.inputs.cp2k.structure)],
            ((s.get_cell_volume() / self.ctx.ref_cell_vol, l, s) for l, s in self.inputs.structures.items())),
                                     key=op.itemgetter(0))

    def run_calculations(self):
        """run the calculations"""

        calcs = {}

        for fraction, label, structure in self.ctx.structures:
            # get the inputs for the workchain doing the calculation
            inputs = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain))

            # ... and replace the structure
            inputs["cp2k"]["structure"] = structure

            future = self.submit(Cp2kBaseWorkChain, **inputs)
            self.report(f"Running Cp2kBaseWorkChain<{future.pk}> for volume fraction {fraction} ({label})")
            calcs[label] = future

        return ToContext(**calcs)

    def results(self):
        """collect results from calculations"""

        self.report("Calculations finished, reporting data")

        results = {}

        for _, label, _ in self.ctx.structures:
            workchain = self.ctx[label]
            result = extract_singlepoint_data(workchain.outputs.output_parameters)
            results[label] = result

        self.out('ev', ev_curve(**results))
