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


from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy import floating
from numpy.typing import ArrayLike, NDArray

from .material import Material
from .typing import Float


@dataclass(kw_only=True)
class Element(ABC):
    material: Material
    dim: int

    @abstractmethod
    def elastic_forces(self, coordinates: ArrayLike) -> NDArray:
        pass

    @abstractmethod
    def damping_forces(self, coordinates: ArrayLike, velocities: ArrayLike) -> NDArray:
        pass

    def forces(self, coordinates: ArrayLike, velocities: ArrayLike) -> NDArray:
        return self.elastic_forces(coordinates) + self.damping_forces(
            coordinates, velocities
        )

    @abstractmethod
    def elastic_energy(self, coordinates: ArrayLike) -> Float:
        pass

    @abstractmethod
    def kinetic_energy(self, coordinates: ArrayLike, velocities: ArrayLike) -> Float:
        pass

    def energy(self, coordinates: ArrayLike, velocities: ArrayLike) -> Float:
        return self.elastic_energy(coordinates) + self.kinetic_energy(
            coordinates, velocities
        )

    @abstractmethod
    def elastic_forces_gradient(self, coordinates: ArrayLike) -> NDArray:
        pass
