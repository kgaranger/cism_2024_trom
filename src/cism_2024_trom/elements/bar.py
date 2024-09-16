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

        bar_vecs = coordinates[1:, :] - coordinates[:-1, :]
        vecs_norms = np.linalg.norm(bar_vecs, axis=1)
        n_bar_vecs = bar_vecs / vecs_norms[:, np.newaxis]
        ax_forces_n = np.array([self.k1, self.k2, self.k1]) * (
            np.array([self.l1, self.l2, self.l1]) - vecs_norms
        )
        ax_forces = np.tensordot(
            [[-1], [1]],
            n_bar_vecs[None, :, :] * ax_forces_n[None, :, None],
            axes=(1, 0),
        )

        vecs_cross_prods = np.cross(n_bar_vecs[:-1, :], n_bar_vecs[1:, :])
        vecs_dot_prods = (n_bar_vecs[:-1, :] * n_bar_vecs[1:, :]).sum(axis=1)
        thetas = np.arctan2(np.linalg.norm(vecs_cross_prods, axis=1), vecs_dot_prods)
        torques_per_sint = self.kt / (
            1 - thetas**2 / 6 + thetas**4 / 120 - thetas**6 / 5040 + thetas**8 / 362880
        )
        n_bar_vecs_per_n = n_bar_vecs / vecs_norms[:, None]
        torque_directions = np.cross(
            np.vstack(
                [n_bar_vecs_per_n[:-1, :].T, n_bar_vecs_per_n[1:, :].T]
            ).T.reshape((-1, 2, self.dim)),
            vecs_cross_prods[:, None, :],
        )
        ang_forces = np.tensordot(
            np.array([[1, 0], [-1, -1], [0, 1]]),
            torques_per_sint[:, None, None] * torque_directions,
            (1, 1),
        )

        forces = np.zeros((4, self.dim))

        for i in range(3):
            forces[i : i + 2, :] += ax_forces[:, i, :]
        for i in range(2):
            forces[i : i + 3, :] += ang_forces[:, i, :]

        return forces

    def damping_forces(self, coordinates: ArrayLike, velocities: ArrayLike) -> NDArray:
        return np.zeros((4, self.dim))

    def elastic_energy(self, coordinates: ArrayLike) -> Float:
        coordinates = np.asarray(coordinates).reshape((4, self.dim))

        bar_vecs = coordinates[1:, :] - coordinates[:-1, :]
        vecs_norms = np.linalg.norm(bar_vecs, axis=1)
        n_bar_vecs = bar_vecs / vecs_norms[:, np.newaxis]
        ax_energies = (
            0.5
            * np.array([self.k1, self.k2, self.k1])
            * (np.array([self.l1, self.l2, self.l1]) - vecs_norms) ** 2
        )

        vecs_cross_prods = np.cross(n_bar_vecs[:-1, :], n_bar_vecs[1:, :])
        vecs_dot_prods = (n_bar_vecs[:-1, :] * n_bar_vecs[1:, :]).sum(axis=1)
        thetas = np.arctan2(np.linalg.norm(vecs_cross_prods, axis=1), vecs_dot_prods)
        ang_energies = 0.5 * self.kt * thetas**2

        return ax_energies.sum() + ang_energies.sum()

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
