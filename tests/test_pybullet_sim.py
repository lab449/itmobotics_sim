import numpy as np
import copy

import time

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
import unittest
from spatialmath import SE3
from spatialmath import base as sb


controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05, 0.05]) * 40}

target_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_ee_state = EEState.from_tf(target_tf, 'iiwa_link_ee')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0, 0.0]))

target_motion = Motion.from_states(target_joint_state, target_ee_state)
target_motion2 = copy.deepcopy(target_motion)

class testPyBulletSim(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf')
        self.__robot = self.__sim.add_robot('tests/urdf/iiwa14_pybullet.urdf', SE3(0,0,0.625) , 'robot')
        self.__robot.joint_controller_params = controller_params

    def test_ee_state(self):
        self.__sim.reset()
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
            ee_state = self.__robot.ee_state('iiwa_link_ee')
            self.assertIsNotNone(ee_state)
            self.assertTrue(np.any(np.abs(ee_state.twist)>np.zeros(6)))
        self.assertTrue(np.any(np.abs(ee_state.force_torque, np.array([0.0, 0.0, 9.82, 0.0, 0.0, 0.0])) > np.ones(6)*1e-6 ))

    def test_joint_state(self):
        self.__sim.reset()
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
            joint_state = self.__robot.joint_state
            self.assertIsNotNone(joint_state)
            self.assertEqual(joint_state.num_joints, 7)
            self.assertTrue(np.any(np.abs(joint_state.joint_velocities) > np.ones(7)*1e-6))

def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
