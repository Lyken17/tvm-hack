# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Defines functions to analyze available opcodes in the ARM ISA."""

import tvm.target


ARM_MPROFILE_DSP_SUPPORT_LIST = [
    "cortex-m7",
    "cortex-m4",
    "cortex-m33",
    "cortex-m35p",
    "cortex-m55",
]


class IsaAnalyzer(object):
    """Checks ISA support for given target"""

    def __init__(self, target):
        self.target = tvm.target.Target(target)

    @property
    def has_dsp_support(self):
        return self.target.mcpu is not None and self.target.mcpu in ARM_MPROFILE_DSP_SUPPORT_LIST
