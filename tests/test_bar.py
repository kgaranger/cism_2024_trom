import math
import os
import unittest
from itertools import product

import numpy as np

from cism_2024_trom.elements import (Bar3D, DiskCrossSection,
                                     HollowDiskCrossSection, Material)


class TestBar3D(unittest.TestCase):
    seed: int = 0
    samples_cnt: int = 10
    places: int = 4
    stretch_range = (0.999, 1.001)
    angular_range = (-math.pi / 6, math.pi / 6)

    def setUp(self):
        self.generator = np.random.default_rng(self.seed)

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
        bar = Bar3D(
            length=self.ref_length,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertEqual(bar.length, self.ref_length)
        self.assertEqual(bar.cross_section, self.ref_cross_section)
        self.assertEqual(bar.material, self.ref_material)

        self.assertRaises(
            ValueError,
            Bar3D,
            length=-1,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertRaises(
            TypeError,
            Bar3D,
            cross_section=self.ref_cross_section,
            material=self.ref_material,
        )
        self.assertRaises(
            TypeError,
            Bar3D,
            length=self.ref_length,
            cross_section=self.ref_cross_section,
        )
        self.assertRaises(
            TypeError, Bar3D, length=self.ref_length, material=self.ref_material
        )

    def test_lengths(self):
        for l in self.all_lengths:
            bar = Bar3D(
                length=l,
                cross_section=self.ref_cross_section,
                material=self.ref_material,
            )
            self.assertAlmostEqual(l, 2 * bar.l1 + bar.l2, places=self.places)

    def test_masses(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            bar = Bar3D(
                length=l,
                cross_section=cs,
                material=mat,
            )
            self.assertAlmostEqual(
                cs.area * l * mat.density, 2 * bar.m1 + 2 * bar.m2, places=self.places
            )

    def test_stiffnesses(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            bar = Bar3D(
                length=l,
                cross_section=cs,
                material=mat,
            )
            self.assertAlmostEqual(
                cs.area * mat.youngs_modulus / l,
                1 / (2 / bar.k1 + 1 / bar.k2),
                places=self.places,
            )

    def test_unbuckled_elastic_forces(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            bar = Bar3D(
                length=l,
                cross_section=cs,
                material=mat,
            )
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
                coordinates = np.array(
                    [
                        pos,
                        pos + stretch * bar.l1 * dir_vec,
                        pos + stretch * (bar.l1 + bar.l2) * dir_vec,
                        pos + stretch * bar.length * dir_vec,
                    ]
                )

                with self.subTest(
                    material=mat.name,
                    cross_section=cs.__class__.__name__,
                    length=l,
                    stretch=stretch,
                    dir_vec=dir_vec,
                    pos=pos,
                ):
                    forces = bar.elastic_forces(coordinates)
                    self.assertAlmostEqual(
                        float(np.linalg.norm(forces[0, :] + forces[-1, :])),
                        0.0,
                        places=self.places,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(forces[0, :])),
                        abs(cs.area * mat.youngs_modulus / l * (stretch - 1) * l),
                        places=self.places,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(forces[1:-1, :])), 0.0, places=self.places
                    )
                    self.assertAlmostEqual(
                        np.dot(forces[0, :], dir_vec),
                        cs.area * mat.youngs_modulus / l * (stretch - 1) * l,
                        places=self.places,
                    )

    def test_unstretched_elastic_forces(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            bar = Bar3D(
                length=l,
                cross_section=cs,
                material=mat,
            )
            for ang0, dirang0, ang1, dirang1, u0, pos in zip(
                self.generator.uniform(*self.angular_range, size=self.samples_cnt),
                self.generator.uniform(-math.pi, math.pi, size=self.samples_cnt),
                self.generator.uniform(*self.angular_range, size=self.samples_cnt),
                self.generator.uniform(-math.pi, math.pi, size=self.samples_cnt),
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
                u0 /= np.linalg.norm(u0)
                x0 = pos
                x1 = x0 + bar.l1 * u0
                u0_cross = np.cross(np.array([[1, 0, 0], [0, 1, 0]]), u0)
                v00 = u0_cross[np.argmax(np.linalg.norm(u0_cross, axis=1)), :]
                v00 /= np.linalg.norm(v00)
                v01 = np.cross(u0, v00)
                v01 /= np.linalg.norm(v01)
                dir1 = np.cos(dirang0) * v00 + np.sin(dirang0) * v01
                u1 = np.cos(ang0) * u0 + np.sin(ang0) * dir1
                x2 = x1 + bar.l2 * u1
                u1_cross = np.cross(np.array([[1, 0, 0], [0, 1, 0]]), u1)
                v10 = u1_cross[np.argmax(np.linalg.norm(u1_cross, axis=1)), :]
                v10 /= np.linalg.norm(v10)
                v11 = np.cross(u1, v10)
                v11 /= np.linalg.norm(v11)
                dir2 = np.cos(dirang1) * v10 + np.sin(dirang1) * v11
                u2 = np.cos(ang1) * u1 + np.sin(ang1) * dir2
                x3 = x2 + bar.l1 * u2
                coordinates = np.array([x0, x1, x2, x3])

                with self.subTest(
                    material=mat.name,
                    cross_section=cs.__class__.__name__,
                    length=l,
                    ang0=ang0,
                    dirang0=dirang0,
                    ang1=ang1,
                    dirang1=dirang1,
                    u0=u0,
                    pos=pos,
                ):
                    forces = bar.elastic_forces(coordinates)
                    self.assertAlmostEqual(
                        float(np.linalg.norm(np.sum(forces, axis=0))),
                        0.0,
                        places=self.places,
                    )
                    fdir0 = np.cross(u0, np.cross(u0, u1))
                    fdir0 /= np.linalg.norm(fdir0)
                    self.assertAlmostEqual(
                        np.dot(forces[0, :], fdir0),
                        bar.kt * abs(ang0) / bar.l1,
                        places=self.places,
                    )
                    fdir3 = np.cross(u2, np.cross(u1, u2))
                    fdir3 /= np.linalg.norm(fdir3)
                    self.assertAlmostEqual(
                        np.dot(forces[-1, :], fdir3),
                        bar.kt * abs(ang1) / bar.l1,
                        places=self.places,
                    )

    def test_elastic_energy(self):
        for mat, cs, l in product(
            self.all_materials, self.all_cross_sections, self.all_lengths
        ):
            bar = Bar3D(
                length=l,
                cross_section=cs,
                material=mat,
            )
            for (
                stretch0,
                ang0,
                dirang0,
                stretch1,
                ang1,
                dirang1,
                stretch2,
                u0,
                pos,
            ) in zip(
                self.generator.uniform(*self.stretch_range, size=self.samples_cnt),
                self.generator.uniform(*self.angular_range, size=self.samples_cnt),
                self.generator.uniform(-math.pi, math.pi, size=self.samples_cnt),
                self.generator.uniform(*self.stretch_range, size=self.samples_cnt),
                self.generator.uniform(*self.angular_range, size=self.samples_cnt),
                self.generator.uniform(-math.pi, math.pi, size=self.samples_cnt),
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
                u0 /= np.linalg.norm(u0)
                x0 = pos
                x1 = x0 + l * (1 - bar.alpha) / 2 * stretch0 * u0
                u0_cross = np.cross(np.array([[1, 0, 0], [0, 1, 0]]), u0)
                v00 = u0_cross[np.argmax(np.linalg.norm(u0_cross, axis=1)), :]
                v00 /= np.linalg.norm(v00)
                v01 = np.cross(u0, v00)
                v01 /= np.linalg.norm(v01)
                dir1 = np.cos(dirang0) * v00 + np.sin(dirang0) * v01
                u1 = np.cos(ang0) * u0 + np.sin(ang0) * dir1
                x2 = x1 + l * bar.alpha * stretch1 * u1
                u1_cross = np.cross(np.array([[1, 0, 0], [0, 1, 0]]), u1)
                v10 = u1_cross[np.argmax(np.linalg.norm(u1_cross, axis=1)), :]
                v10 /= np.linalg.norm(v10)
                v11 = np.cross(u1, v10)
                v11 /= np.linalg.norm(v11)
                dir2 = np.cos(dirang1) * v10 + np.sin(dirang1) * v11
                u2 = np.cos(ang1) * u1 + np.sin(ang1) * dir2
                x3 = x2 + l * (1 - bar.alpha) / 2 * stretch2 * u2
                coordinates = np.array([x0, x1, x2, x3])

                with self.subTest(
                    material=mat.name,
                    cross_section=cs.__class__.__name__,
                    length=l,
                    ang0=ang0,
                    dirang0=dirang0,
                    ang1=ang1,
                    dirang1=dirang1,
                    u0=u0,
                    pos=pos,
                ):
                    energy = bar.elastic_energy(coordinates)
                    self.assertAlmostEqual(
                        energy,
                        0.5
                        * bar.k1
                        * bar.l1**2
                        * ((stretch0 - 1) ** 2 + (stretch2 - 1) ** 2)
                        + 0.5 * bar.k2 * (bar.l2 * (stretch1 - 1)) ** 2
                        + 0.5 * bar.kt * (ang0**2 + ang1**2),
                        places=self.places,
                    )


if __name__ == "__main__":
    TestBar3D.seed = int(os.getenv("SEED", TestBar3D.seed))
    TestBar3D.samples_cnt = int(os.getenv("SAMPLES_CNT", TestBar3D.samples_cnt))
    unittest.main(failfast=True)
