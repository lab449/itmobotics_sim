import os
import sys
import time
import copy
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import SimulationException
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController

controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

test_joint_pose = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
test_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_joint_state = JointState.from_position(test_joint_pose)

peg_link_name = 'peg_target_link'

target_ee_state = EEState.from_tf(test_tf, peg_link_name)
target_motion = Motion.from_states(copy.deepcopy(target_joint_state), copy.deepcopy(target_ee_state))


final_time = 3.0
n_samples = 5

class testPyBulletAddTool(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1.0)
        self.__sim.add_object('hole_round', 'tests/urdf/hole_round.urdf', base_transform = SE3(0.3, -0.5, 0.8), fixed = True, save = True, scale_size=1.1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', fixed = True, save = True)

        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0, -0.3, 0.625), 'robot1')
        self.__robot.joint_controller_params = controller_params

        self.__controller_ee_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_ee_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_ee_pose.connect_controller(self.__controller_ee_speed)

        self.__sim.add_object('peg_round2', 'tests/urdf/peg_round.urdf', base_transform = SE3(0.1, 0.0, 0.675), fixed = True, save = True)
        
    def __reset(self):
        self.__sim.reset()

    def test_add_tool(self):
        self.__robot.connect_tool('peg' ,'tests/urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1))
        self.__sim.sim_step()
        self.assertIsNotNone(self.__robot.ee_state(peg_link_name))
        self.__sim.sim_step()
        self.__robot.remove_tool('peg')
        self.__sim.sim_step()
        self.assertRaises(KeyError, self.__robot.ee_state, peg_link_name)

        self.__robot.connect_tool('peg' ,'tests/urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1), save=True)
        self.__sim.sim_step()
        self.__robot.reset_ee_state(target_ee_state)
        self.assertIsNotNone(self.__robot.ee_state(peg_link_name))

        np.allclose(target_ee_state.tf.A, self.__robot.ee_state(peg_link_name).tf.A, atol=1e-5)
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
