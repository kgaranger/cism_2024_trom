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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circular_cross_section import CircularCrossSection
from .element import Element
from .material import Material
from .typing import Float


@dataclass(kw_only=True)
class Bar(Element):
    length: float
    cross_section: CircularCrossSection
    k1: float = field(init=False)
    k2: float = field(init=False)
    kt: float = field(init=False)
    m1: float = field(init=False)
    m2: float = field(init=False)
    l1: float = field(init=False)
    l2: float = field(init=False)
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, but got {self.length}")
        if self.alpha < 0 or self.alpha > 1 / math.sqrt(3):
            raise ValueError(
                f"alpha must be between 0 and 1/sqrt(3), but got {self.alpha}"
            )
        self.k1 = (
            2
            * self.material.youngs_modulus
            * self.cross_section.area
            / self.length
            / (1 - self.alpha)
        )
        self.k2 = (
            self.material.youngs_modulus
            * self.cross_section.area
            / self.length
            / self.alpha
        )
        self.kt = (
            (1 - self.alpha)
            / 2
            * math.pi**2
            * self.material.youngs_modulus
            * self.cross_section.second_moment_of_area
            / self.length
            * (
                1
                - self.cross_section.second_moment_of_area
                * math.pi**2
                / self.cross_section.area
                / self.length**2
            )
        )
        self.m1 = (
            1
            / 6
            * self.material.density
            * self.cross_section.area
            * self.length
            * (1 - 3 * self.alpha**2)
            / (1 - self.alpha**2)
        )
        self.m2 = (
            1
            / 3
            * self.material.density
            * self.cross_section.area
            * self.length
            / (1 - self.alpha**2)
        )
        self.l1 = (1 - self.alpha) / 2 * self.length
        self.l2 = self.alpha * self.length

    def elastic_forces(self, coordinates: ArrayLike) -> NDArray:
        coordinates = np.asarray(coordinates).reshape((4, self.dim))

        ###### Instructions ######
        # Compute the elastic forces acting on all the four nodes of the bar.
        ### Input:  `coordinates` is an array containing the coordinates of the nodes (or masses) of the bar.
        #           The dimension of the coordinates is given by the attribute `self.dim`.
        #           The shape of `coordinates` is therefore (4, self.dim),
        #           so the coordinates of the first node are `coordinates[0, :]`,
        #           the coordinates of the second node are `coordinates[1, :]`, etc.
        ### Output: `nodes_forces` should represent the forces acting on the four nodes.
        #           This variable should be defined in your code as either:
        #               - an array of shape (4, self.dim),
        #               - or a list of four lists of length `self.dim` each.
        ### Notes:
        #           - The arctangent of x/y can be computed with `np.atan2(x, y)`.
        #           - Your implementation can assume that self.dim is equal to 3 for the tests to pass.
        #           - The `self.k1` and `self.k2` attributes contain the stiffness coefficients of the end and middle
        #             bar segments, respectively.
        #           - The `self.l1` and `self.l2` attributes contain the lengths of the end and middle bar segments,
        #             respectively.
        #           - The `self.kt` attribute contains the torsional stiffness coefficient of the angular springs.

        # YOUR CODE STARTS HERE

        # Compute the vectors between each consecutive pair of nodes (`t` vectors).
        bars_vecs = coordinates[1:, :] - coordinates[:-1, :]

        # Compute the actual lengths of the bar segments
        bars_vecs_norms = np.linalg.norm(bars_vecs, axis=1)

        # Compute the normalized vectors along the bar segments (`u` vectors).
        n_bars_vecs = bars_vecs / bars_vecs_norms[:, None]

        # Compute the magnitudes of the elastic forces of the axial springs
        ax_forces_mags = np.array([self.k1, self.k2, self.k1]) * (
            np.array([self.l1, self.l2, self.l1]) - bars_vecs_norms
        )

        # Compute the force vectors of the axial springs
        ax_forces_vecs = n_bars_vecs[:, :] * ax_forces_mags[:, None]

        # Compute the cross products between the successive `u` vectors
        vecs_cross_prods = np.cross(n_bars_vecs[:-1, :], n_bars_vecs[1:, :])

        # Compute the dot products between the successive `u` vectors
        vecs_dot_prods = (n_bars_vecs[:-1, :] * n_bars_vecs[1:, :]).sum(axis=1)

        # Compute the angles between the successive bar segments
        thetas = np.arctan2(np.linalg.norm(vecs_cross_prods, axis=1), vecs_dot_prods)

        # Compute the torque of each angular spring divided by the sine of the corresponding angle
        # (k_t * theta_i / sin(theta_i))
        torques_per_sint = self.kt / (
            1 - thetas**2 / 6 + thetas**4 / 120 - thetas**6 / 5040 + thetas**8 / 362880
        )

        # Compute the vectors that when multiplied by their respective torque give the forces of each angular spring on
        # its adjacent nodes
        n_bars_vecs_per_n = n_bars_vecs / bars_vecs_norms[:, None]
        torque_directions = np.cross(
            np.vstack(
                [n_bars_vecs_per_n[:-1, :].T, n_bars_vecs_per_n[1:, :].T]
            ).T.reshape((-1, 2, self.dim)),
            vecs_cross_prods[:, None, :],
        )

        nodes_forces = np.zeros((4, self.dim))

        for i in range(3):
            # Add the forces of spring i to the forces of its two adjacent nodes
            nodes_forces[i, :] -= ax_forces_vecs[i, :]
            nodes_forces[i + 1, :] += ax_forces_vecs[i, :]
        for i in range(2):
            # Add the forces of the angular spring i to the three forces it acts on
            f1 = (
                np.cross(n_bars_vecs[i, :] / bars_vecs_norms[i], vecs_cross_prods[i, :])
                * torques_per_sint[i]
            )

            nodes_forces[i, :] += f1
            nodes_forces[i + 1, :] -= f1
            f2 = (
                np.cross(
                    n_bars_vecs[i + 1, :] / bars_vecs_norms[i + 1],
                    vecs_cross_prods[i, :],
                )
                * torques_per_sint[i]
            )
            nodes_forces[i + 2, :] += f2
            nodes_forces[i + 1, :] -= f2

        # YOUR CODE ENDS HERE, THE VARIABLE `nodes_forces` SHOULD HAVE BEEN DEFINED

        return nodes_forces

    def damping_forces(self, coordinates: ArrayLike, velocities: ArrayLike) -> NDArray:
        return np.zeros((4, self.dim))

    def elastic_energy(self, coordinates: ArrayLike) -> Float:
        coordinates = np.asarray(coordinates).reshape((4, self.dim))

        ###### Instructions ######
        # Compute the elastic energy stored in the bar.
        ### Input:  `coordinates` as in the `elastic_forces` method.
        ### Output: `elastic_energy` should represent the elastic energy stored in the bar.

        # YOUR CODE STARTS HERE

        bars_vecs = coordinates[1:, :] - coordinates[:-1, :]
        bars_vecs_norms = np.linalg.norm(bars_vecs, axis=1)
        n_bars_vecs = bars_vecs / bars_vecs_norms[:, np.newaxis]
        ax_energies = (
            0.5
            * np.array([self.k1, self.k2, self.k1])
            * (np.array([self.l1, self.l2, self.l1]) - bars_vecs_norms) ** 2
        )

        vecs_cross_prods = np.cross(n_bars_vecs[:-1, :], n_bars_vecs[1:, :])
        vecs_dot_prods = (n_bars_vecs[:-1, :] * n_bars_vecs[1:, :]).sum(axis=1)
        thetas = np.arctan2(np.linalg.norm(vecs_cross_prods, axis=1), vecs_dot_prods)
        ang_energies = 0.5 * self.kt * thetas**2

        elastic_energy = ax_energies.sum() + ang_energies.sum()

        # YOUR CODE ENDS HERE, THE VARIABLE `elastic_energy` SHOULD HAVE BEEN DEFINED

        return elastic_energy

    def kinetic_energy(self, coordinates: ArrayLike, velocities: ArrayLike) -> Float:
        velocities = np.asarray(velocities).reshape((4, self.dim))
        velocities_norms = np.linalg.norm(velocities, axis=1)
        return 0.5 * (
            self.m1 * (velocities_norms[0] ** 2 + velocities_norms[3] ** 2)
            + self.m2 * (velocities_norms[1] ** 2 + velocities_norms[2] ** 2)
        )

    def elastic_forces_gradient(self, coordinates: ArrayLike) -> NDArray:
        raise NotImplementedError


@dataclass(kw_only=True)
class Bar2D(Bar):
    dim: int = 2


@dataclass(kw_only=True)
class Bar3D(Bar):
    dim: int = 3
