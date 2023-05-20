import copy
import time
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE


class testPyBulletCollision(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01, time_scale=1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', save=True)
        self.__robot = self.__sim.add_robot('tests/urdf/iiwa14_pybullet.urdf', SE3(0,0,0.725) , 'robot')

    def test_collision(self):
        self.__sim.reset()
        self.__sim.sim_step()
        self.assertEqual(len(self.__sim.is_collide('robot')), 0)
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
        print("Collision list: ", self.__sim.is_collide('robot'))
        self.assertNotEqual(len(self.__sim.is_collide('robot')), 0)
        self.assertTrue('table' in self.__sim.is_collide('robot'))

def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
