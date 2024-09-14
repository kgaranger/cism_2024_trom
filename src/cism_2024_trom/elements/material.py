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


from dataclasses import dataclass


@dataclass(kw_only=True)
class Material:
    name: str = ""
    youngs_modulus: float
    density: float

    def __post_init__(self):
        if self.youngs_modulus <= 0:
            raise ValueError(
                f"Expected youngs_modulus to be positive but got {self.youngs_modulus}"
            )
        if self.density <= 0:
            raise ValueError(f"Expected density to be positive but got {self.density}")
