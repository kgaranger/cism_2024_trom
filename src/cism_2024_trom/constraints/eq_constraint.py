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

import numpy as np
from numpy.typing import NDArray


@dataclass
class EqConstraint:

    A: NDArray
    b: NDArray
    Apinv = property(lambda self: np.linalg.pinv(self.A))
    Apinvproj = property(lambda self: self.Apinv @ self.A)
    Apinvb = property(lambda self: self.Apinv @ self.b)

    def __post_init__(self):
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"A and b must have the same number of rows but got {self.A.shape[0]} and {self.b.shape[0]}"
            )
        if np.linalg.matrix_rank(self.A) < np.linalg.matrix_rank(
            np.hstack([self.A, self.b.reshape(-1, 1)])
        ):
            print("Warning: b is not in the column space of A")

    def apply(self, x: NDArray) -> NDArray:
        return x - self.Apinvproj @ x + self.Apinvb

    def __add__(self, other: "EqConstraint") -> "EqConstraint":
        return EqConstraint(
            A=np.vstack([self.A, other.A]),
            b=np.vstack([self.b, other.b]),
        )
