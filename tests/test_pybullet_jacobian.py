import os
import sys
import copy
import unittest
import time
import pybullet as p

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))


np.set_printoptions(precision=2)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.00001)
    
    def test_jacobi_nonzero_orient(self):

        robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,-0.3,0.625), 'robot1', True)
        robot.reset_joint_state(target_joint_state)
        # self.__sim.sim_step()
        jac = robot.jacobian(robot.joint_state.joint_positions, 'ee_tool')
        self.__sim.remove_robot('robot1')
        self.__sim.reset()

        base_orient = SE3.Rz(np.pi)
        robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,-0.3,0.625)@base_orient, 'robot1')
        robot.reset_joint_state(target_joint_state)
        # self.__sim.sim_step()
        jac2 = np.kron(np.eye(2,2), base_orient.R) @ robot.jacobian(robot.joint_state.joint_positions, 'ee_tool')

        np.testing.assert_almost_equal(jac, jac2)

def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
