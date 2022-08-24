import numpy as np

import unittest
from spatialmath import SE3
from spatialmath import base as sb
from itmobotics_sim.utils.math import SE32vec, vec2SE3

from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE

class testUtils(unittest.TestCase):
    def setUp(self):
        self.__tf = SE3(0.0, 1.0, 0.0) @ SE3.Rx(np.pi/2)
        self.__vec = np.array([0.0, 1.0, 0.0, np.pi/2, 0.0, 0.0])

    def test_tf_to_vec(self):
        np.testing.assert_array_equal(SE32vec(self.__tf), self.__vec)
        self.assertTrue(self.__tf == vec2SE3(self.__vec))
    
    def test_clip_joint_state(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1)
        self.__robot = self.__sim.add_robot('tests/urdf/iiwa14_pybullet.urdf', SE3(0,0,0.625) , 'robot1')
        self.__sim.sim_step()
        
        js = self.__robot.joint_state
        self.assertIsNotNone(js)
        self.__robot.joint_limits.clip_joint_state(js)
        
        self.assertTrue(np.all(js.joint_positions > self.__robot.joint_limits.limit_positions[0]))
        self.assertTrue(np.all(js.joint_positions < self.__robot.joint_limits.limit_positions[1]))

        self.assertTrue(np.all(js.joint_velocities > self.__robot.joint_limits.limit_velocities[0]))
        self.assertTrue(np.all(js.joint_velocities < self.__robot.joint_limits.limit_velocities[1]))

        self.assertTrue(np.all(js.joint_torques > self.__robot.joint_limits.limit_torques[0]))
        self.assertTrue(np.all(js.joint_torques < self.__robot.joint_limits.limit_torques[1]))
    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
