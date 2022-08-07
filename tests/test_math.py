import os
import sys
import numpy as np
import copy

import unittest
from spatialmath import SE3
from spatialmath import base as sb
from itmobotics_sim.utils.math import SE32vec, vec2SE3

from itmobotics_sim.pybullet_env.urdf_editor import URDFEditor

class testMath(unittest.TestCase):
    def setUp(self):
        self.__tf = SE3(0.0, 1.0, 0.0) @ SE3.Rx(np.pi/2)
        self.__vec = np.array([0.0, 1.0, 0.0, np.pi/2, 0.0, 0.0])

    def test_tf_to_vec(self):
        np.testing.assert_array_equal(SE32vec(self.__tf), self.__vec)
        self.assertTrue(self.__tf == vec2SE3(self.__vec))
    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
