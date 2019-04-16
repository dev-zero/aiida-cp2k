# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""AiiDA-CP2K input plugin"""

from __future__ import absolute_import


class Cp2kInput:
    """Transforms dictionary into CP2K input"""

    DISCLAIMER = "!!! Generated by AiiDA !!!"

    def __init__(self, params=None):
        if not params:
            self._params = {}
        else:
            self._params = params

    def add_keyword(self, kwpath, value):
        self._add_keyword_low(kwpath.split("/"), value, self._params)

    def render(self):
        output = [self.DISCLAIMER]
        self._render_section(output, self._params)
        return "\n".join(output)

    @staticmethod
    def _add_keyword_low(kwpath, value, params):
        """Adds keyword"""
        if len(kwpath) == 1:
            params[kwpath[0]] = value
        elif kwpath[0] not in params.keys():
            new_subsection = {}
            params[kwpath[0]] = new_subsection
            Cp2kInput._add_keyword_low(kwpath[1:], value, new_subsection)
        else:
            Cp2kInput._add_keyword_low(kwpath[1:], value, params[kwpath[0]])

    @staticmethod
    def _render_section(output, params, indent=0):
        """
        It takes a dictionary and recurses through.

        For key-value pair it checks whether the value is a dictionary
        and prepends the key with &
        It passes the valued to the same function, increasing the indentation
        If the value is a list, I assume that this is something the user
        wants to store repetitively
        eg:
            dict['KEY'] = ['val1', 'val2']
            ===>
            KEY val1
            KEY val2

            or

            dict['KIND'] = [{'_': 'Ba', 'ELEMENT':'Ba'},
                            {'_': 'Ti', 'ELEMENT':'Ti'},
                            {'_': 'O', 'ELEMENT':'O'}]
            ====>
                  &KIND Ba
                     ELEMENT  Ba
                  &END KIND
                  &KIND Ti
                     ELEMENT  Ti
                  &END KIND
                  &KIND O
                     ELEMENT  O
                  &END KIND
        """

        for key, val in sorted(params.items()):
            if key.upper() != key:
                raise ValueError("keyword '%s' not upper case" % key)
            if key.startswith("@") or key.startswith("$"):
                raise ValueError("CP2K preprocessor not supported")
            if isinstance(val, dict):
                line = "%s&%s" % (" " * indent, key)
                if "_" in val:  # if there is a section parameter, add it
                    line += " %s" % val["_"]
                output.append(line)
                Cp2kInput._render_section(output, val, indent + 3)
                output.append("%s&END %s" % (" " * indent, key))
            elif isinstance(val, list):
                for listitem in val:
                    Cp2kInput._render_section(output, {key: listitem}, indent)
            elif isinstance(val, bool):
                val_str = ".true." if val else ".false."
                output.append("%s%s  %s" % (" " * indent, key, val_str))
            else:
                output.append("%s%s  %s" % (" " * indent, key, val))