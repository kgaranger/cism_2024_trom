import os
import unittest
from itertools import product

import numpy as np

from cism_2024_trom.elements import (Cable3D, DiskCrossSection,
                                     HollowDiskCrossSection, Material)


class TestCable3D(unittest.TestCase):
    seed: int = 0
    samples_cnt: int = 10
    places: int = 4
    stretch_range = (0.999, 1.001)

    def setUp(self):
        self.generator = np.random.default_rng(self.seed)
        self.places = 6

        self.ref_material = Material(
            name="Titanium",
            youngs_modulus=91e9,
            density=4480,
        )
        self.ref_cross_section = HollowDiskCrossSection(
            r1=19e-3 / 2 - 1e-3,
            r2=19e-3 / 2,
        )
        self.ref_length = 0.58

        self.all_materials = [
            self.ref_material,
            Material(
                name="Aluminium",
                youngs_modulus=70e9,
                density=2700,
            ),
            Material(
                name="Steel",
                youngs_modulus=210e9,
                density=7800,
            ),
        ]
        self.all_cross_sections = [
            self.ref_cross_section,
            DiskCrossSection(radius=10e-3),
            DiskCrossSection(radius=15e-3),
        ]
        self.all_lengths = [self.ref_length, 0.5, 1.0]

    def test_init(self):
        cable = Cable3D(
            length=self.ref_length,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertEqual(cable.length, self.ref_length)
        self.assertEqual(cable.cross_section, self.ref_cross_section)
        self.assertEqual(cable.material, self.ref_material)

        self.assertRaises(
            ValueError,
            Cable3D,
            length=-1,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertRaises(
            TypeError,
            Cable3D,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertRaises(
            TypeError,
            Cable3D,
            length=self.ref_length,
            cross_section=self.ref_cross_section,
        )
        self.assertRaises(
            TypeError, Cable3D, length=self.ref_length, material=self.ref_material
        )

    def test_stiffness(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            cable = Cable3D(length=l, cross_section=cs, material=mat)
            self.assertAlmostEqual(
                cable.stiffness, mat.youngs_modulus * cs.area / l, places=self.places
            )

    def test_elastic_forces(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            cable = Cable3D(length=l, cross_section=cs, material=mat)
            for stretch, dir_vec, pos in zip(
                self.generator.uniform(*self.stretch_range, size=self.samples_cnt),
                self.generator.multivariate_normal(
                    mean=np.zeros(3),
                    cov=np.eye(3),
                    size=self.samples_cnt,
                ),
                self.generator.multivariate_normal(
                    mean=np.zeros(3),
                    cov=np.eye(3),
                    size=self.samples_cnt,
                ),
            ):
                dir_vec /= np.linalg.norm(dir_vec)
                coordinates = np.array([pos, pos + stretch * l * dir_vec])

                with self.subTest(
                    material=mat.name,
                    cross_section=cs.__class__.__name__,
                    length=l,
                    stretch=stretch,
                    dir_vec=dir_vec,
                    pos=pos,
                ):
                    forces = cable.elastic_forces(coordinates)
                    self.assertAlmostEqual(
                        float(np.linalg.norm(forces[0, :] + forces[1, :])),
                        0.0,
                        places=self.places,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(forces[0, :])),
                        max(0, cable.stiffness * (stretch - 1) * l),
                        places=self.places,
                    )
                    self.assertAlmostEqual(
                        np.linalg.norm(forces[0, :]),
                        np.dot(forces[0, :], dir_vec),
                        places=self.places,
                    )

    def test_elastic_energy(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            cable = Cable3D(length=l, cross_section=cs, material=mat)
            for stretch, dir_vec, pos in zip(
                self.generator.uniform(*self.stretch_range, size=self.samples_cnt),
                self.generator.multivariate_normal(
                    mean=np.zeros(3),
                    cov=np.eye(3),
                    size=self.samples_cnt,
                ),
                self.generator.multivariate_normal(
                    mean=np.zeros(3),
                    cov=np.eye(3),
                    size=self.samples_cnt,
                ),
            ):
                dir_vec /= np.linalg.norm(dir_vec)
                coordinates = np.array([pos, pos + stretch * l * dir_vec])

                with self.subTest(
                    material=mat.name,
                    cross_section=cs.__class__.__name__,
                    length=l,
                    stretch=stretch,
                    dir_vec=dir_vec,
                ):
                    energy = cable.elastic_energy(coordinates)
                    self.assertAlmostEqual(
                        energy,
                        0.5 * cable.stiffness * (max(0, stretch - 1) * l) ** 2,
                        places=self.places,
                    )


if __name__ == "__main__":
    TestCable3D.seed = int(os.getenv("SEED", TestCable3D.seed))
    TestCable3D.samples_cnt = int(os.getenv("SAMPLES_CNT", TestCable3D.samples_cnt))

    unittest.main(failfast=True)
