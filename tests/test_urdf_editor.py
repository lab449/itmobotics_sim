import os
import sys
import copy
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.pybullet_env.urdf_editor import URDFEditor


class testUrdfEditor(unittest.TestCase):
    def setUp(self):
        self.__editor = URDFEditor('tests/urdf/ur5e_pybullet.urdf')
        self.__editor2 = URDFEditor('tests/urdf/table2.urdf')

    def test_join_urdf(self):
        self.__editor.joinURDF(self.__editor2, 'ee_tool', SE3(0.0,0.0,0.1).A)
        self.__editor.save('new.urdf')
        os.remove('new.urdf')
    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
