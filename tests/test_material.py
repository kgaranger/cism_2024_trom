import unittest
from itertools import product

import numpy as np

from cism_2024_trom.elements import Material


class TestMaterial(unittest.TestCase):
    def setUp(self):
        self.ref_name = "Titanium"
        self.ref_youngs_modulus = 91e9
        self.ref_density = 4480

    def test_init(self):
        material = Material(
            name=self.ref_name,
            youngs_modulus=self.ref_youngs_modulus,
            density=self.ref_density,
        )
        self.assertEqual(material.name, self.ref_name)
        self.assertEqual(material.youngs_modulus, self.ref_youngs_modulus)
        self.assertEqual(material.density, self.ref_density)

        self.assertRaises(
            TypeError,
            Material,
            name=self.ref_name,
            youngs_modulus=self.ref_youngs_modulus,
        )

        self.assertRaises(
            TypeError,
            Material,
            name=self.ref_name,
            density=self.ref_density,
        )

        self.assertRaises(
            ValueError,
            Material,
            name=self.ref_name,
            youngs_modulus=-1,
            density=self.ref_density,
        )

        self.assertRaises(
            ValueError,
            Material,
            name=self.ref_name,
            youngs_modulus=self.ref_youngs_modulus,
            density=-1,
        )


if __name__ == "__main__":
    unittest.main()
