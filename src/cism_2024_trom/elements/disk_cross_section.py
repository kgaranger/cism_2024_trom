# Copyright 2024 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from dataclasses import InitVar, dataclass, field

from .circular_cross_section import CircularCrossSection


@dataclass(kw_only=True)
class DiskCrossSection(CircularCrossSection):
    radius: InitVar[float]

    def __post_init__(self, radius: float) -> None:
        self.area = math.pi * radius**2
        self.second_moment_of_area = math.pi / 4 * radius**4
