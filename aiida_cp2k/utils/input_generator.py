# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""AiiDA-CP2K input generator"""

from __future__ import absolute_import
from __future__ import division

from copy import deepcopy

import six
if six.PY2:
    from collections import Mapping, Sequence  # pylint: disable=import-error, no-name-in-module
else:
    from collections.abc import Mapping, Sequence  # pylint: disable=import-error, no-name-in-module


class Cp2kInput:  # pylint: disable=old-style-class
    """Transforms dictionary into CP2K input"""

    DISCLAIMER = "!!! Generated by AiiDA !!!"

    def __init__(self, params=None):
        """Initializing Cp2kInput object"""
        if not params:
            self._params = {}
        else:
            # always make a full copy to avoid that add_keyword() changes the
            # passed-in dictionary
            self._params = deepcopy(params)

    def __getitem__(self, key):
        return self._params[key]

    def add_keyword(self, kwpath, value, override=True, conflicting_keys=None):
        """
        Add a value for the given keyword.

        Args:

            kwpath: Can be a single keyword, a path with `/` as divider for sections & key,
                    or a sequence with sections and key.

            value: the value to set the given key to

            override: whether to override the key if it is already present in the self._params

            conflicting_keys: list of keys that cannot live together with the provided key
            (SOMETHING1/[..]/SOMETHING2/KEY). In case override is True, all conflicting keys will
            be removed, if override is False and conflicting_keys are present the new key won't be
            added.
        """

        if isinstance(kwpath, six.string_types):
            kwpath = kwpath.split("/")

        Cp2kInput._add_keyword(kwpath, value, self._params, ovrd=override, cfct=conflicting_keys)

    def render(self):
        output = [self.DISCLAIMER]
        self._render_section(output, deepcopy(self._params))
        return u"\n".join(output)

    @staticmethod
    def _add_keyword(kwpath, value, params, ovrd, cfct):
        """Add keyword into the given nested dictionary"""
        conflicting_keys_present = []
        # key/value for the deepest level
        if len(kwpath) == 1:
            if cfct:
                conflicting_keys_present = [key for key in cfct if key in params]
            if ovrd:  # if ovrd is True, set the new element's value and remove the conflicting keys
                params[kwpath[0]] = value
                for key in conflicting_keys_present:
                    params.pop(key)
            # if ovrd is False, I only add the new key if (1) it wasn't present beforeAND (2) it does not conflict
            # with any key that is currently present
            elif not conflicting_keys_present and kwpath[0] not in params:
                params[kwpath[0]] = value

        # the key was not present in the dictionary, and we are not yet at the deepest level,
        # therefore a subdictionary should be added
        elif kwpath[0] not in params:
            params[kwpath[0]] = {}
            Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)

        # if it is a list, loop over its elements
        elif isinstance(params[kwpath[0]], Sequence) and not isinstance(params[kwpath[0]], six.string_types):
            for element in params[kwpath[0]]:
                Cp2kInput._add_keyword(kwpath[1:], value, element, ovrd, cfct)

        # if the key does NOT point to a dictionary and we are not yet at the deepest level,
        # therefore, the element should be replaced with an empty dictionary unless ovrd is False
        elif not isinstance(params[kwpath[0]], Mapping):
            if ovrd:
                params[kwpath[0]] = {}
                Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)
        # if params[kwpath[0]] points to a sub-dictionary, enter into it
        else:
            Cp2kInput._add_keyword(kwpath[1:], value, params[kwpath[0]], ovrd, cfct)

    @staticmethod
    def _render_section(output, params, indent=0):
        """It takes a dictionary and recurses through.

        For key-value pair it checks whether the value is a dictionary and prepends the key with & (CP2K section).
        It passes the valued to the same function, increasing the indentation. If the value is a list, I assume
        that this is something the user wants to store repetitively
        eg:
        .. highlight::

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
            # keys are not case-insensitive, ensure that they follow the current scheme
            if key.upper() != key:
                raise ValueError("keyword '{}' not upper case".format(key))

            if key.startswith(("@", "$")):
                raise ValueError("CP2K preprocessor directives not supported")

            if isinstance(val, Mapping):
                line = "{}&{}".format(' ' * indent, key)
                if "_" in val:  # if there is a section parameter, add it
                    line += " {}".format(val.pop('_'))
                output.append(line)
                Cp2kInput._render_section(output, val, indent + 3)
                output.append('{}&END {}'.format(' ' * indent, key))

            elif isinstance(val, Sequence) and not isinstance(val, six.string_types):
                for listitem in val:
                    Cp2kInput._render_section(output, {key: listitem}, indent)

            elif isinstance(val, bool):
                val_str = '.TRUE.' if val else '.FALSE.'
                output.append('{}{} {}'.format(' ' * indent, key, val_str))

            else:
                output.append('{}{} {}'.format(' ' * indent, key, val))
