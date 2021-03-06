# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""Check versions"""

from __future__ import print_function
from __future__ import absolute_import

import sys
import json
import re


def check_version():
    """Check if versions in setup.json and in plugin are consistent"""

    with open("./aiida_cp2k/__init__.py", "r") as fhandle:
        match = re.search(r"^__version__ = \"([^\"]+)\"", fhandle.read(), re.MULTILINE)
        version1 = match.group(1)

    with open("./setup.json") as fhandle:
        version2 = json.load(fhandle)["version"]

    if version1 != version2:
        print(
            "ERROR: Versions in aiida_cp2k/__init__.py and setup.json are inconsistent: '%s' vs '%s'"
            % (version1, version2)
        )
        sys.exit(3)


if __name__ == "__main__":
    check_version()
