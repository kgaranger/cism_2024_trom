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


from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circular_cross_section import CircularCrossSection
from .element import Element
from .material import Material
from .typing import Float


@dataclass(kw_only=True)
class Cable(Element):
    length: float
    cross_section: CircularCrossSection
    stiffness: float = field(init=False)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, but got {self.length}")
        self.stiffness = (
            self.material.youngs_modulus * self.cross_section.area / self.length
        )

    def elastic_forces(self, coordinates: ArrayLike) -> NDArray:
        coordinates = np.asarray(coordinates).reshape((2, self.dim))

        ###### Instructions ######
        # Compute the elastic forces acting on the nodes at the ends of the cable.
        ### Input:  `coordinates` is an array containing the coordinates of the nodes at the ends of the cable.
        #           The dimension of the coordinates is given by the attribute `self.dim`.
        #           The shape of `coordinates` is therefore (2, self.dim),
        #           so the coordinates of the first node are `coordinates[0, :]`
        #           and the coordinates of the second node are `coordinates[1, :]`.
        ### Output: `nodes_forces` should represent the forces acting on the nodes at the ends of the cable.
        #           This variable should be defined in your code as either:
        #               - an array of shape (2, self.dim),
        #               - or a list of two lists of length `self.dim` each.
        ### Notes:
        #           - The square root function is available in the numpy module as `np.sqrt`.
        #           - The dot product of two vectors `a` and `b` can be computed as `np.dot(a, b)`.
        #           - Your implementation can assume that self.dim is equal to 3 for the tests to pass.
        #           - The variable `self.stiffness` is the stiffness of the cable.

        # YOUR CODE STARTS HERE

        # Compute the vector from the first node to the second node
        cable_vec = coordinates[1, :] - coordinates[0, :]

        # Compute the actual length of the cable
        vec_norm = np.linalg.norm(cable_vec)

        # Check whether the cable is in tension or compression
        if vec_norm < self.length:
            # The cable is in compression, so the force is zero
            nodes_forces = np.zeros((2, self.dim))
        else:
            # The cable is in tension, so the force is the linear spring force
            force_magnitude = self.stiffness * (vec_norm - self.length)
            force_vec = force_magnitude * cable_vec / vec_norm
            nodes_forces = np.array([force_vec, -force_vec])

        # YOUR CODE ENDS HERE, THE VARIABLE `nodes_forces` SHOULD HAVE BEEN DEFINED

        return np.asarray(nodes_forces)

    def damping_forces(self, coordinates: ArrayLike, velocities: ArrayLike) -> NDArray:
        return np.zeros((2, self.dim))

    def elastic_energy(self, coordinates: ArrayLike) -> Float:
        coordinates = np.asarray(coordinates).reshape((2, self.dim))

        ###### Instructions ######
        # Compute the elastic energy stored in the cable.
        ### Input:  `coordinates` as in the `elastic_forces` method.
        ### Output: `elastic_energy` should represent the elastic energy stored in the cable.

        # YOUR CODE STARTS HERE

        elastic_energy = (
            0.5
            * self.stiffness
            * max(
                np.linalg.norm(coordinates[1, :] - coordinates[0, :]) - self.length, 0
            )
            ** 2
        )

        # YOUR CODE ENDS HERE, THE VARIABLE `elastic_energy` SHOULD HAVE BEEN DEFINED

        return elastic_energy

    def kinetic_energy(self, coordinates: ArrayLike, velocities: ArrayLike) -> Float:
        velocities = np.asarray(velocities).reshape((2, self.dim))
        return (
            0.5
            * self.material.density
            * self.cross_section.area
            * self.length
            * (
                np.linalg.norm(velocities[0, :]) ** 2
                + np.linalg.norm(velocities[1, :]) ** 2
                + np.dot(velocities[0, :], velocities[1, :])
            )
        )

    def elastic_forces_gradient(self, coordinates: ArrayLike) -> NDArray:
        raise NotImplementedError


@dataclass(kw_only=True)
class Cable2D(Cable):
    dim: int = 2


@dataclass(kw_only=True)
class Cable3D(Cable):
    dim: int = 3
