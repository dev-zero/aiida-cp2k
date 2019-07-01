# -*- coding: utf-8 -*-

from __future__ import absolute_import
import click


@click.group("aiida-cp2k", context_settings={"help_option_names": ["-h", "--help"]})
def root():
    """CLI for the `aiida-cp2k` plugin"""


@root.group("calc")
def calculations():
    """Manage calculations"""


# we need the groups to be defined first
from .calculations import *  # noqa: F401, F403
