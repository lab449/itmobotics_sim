import os
import sys
import time

import numpy as np
import copy

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
import unittest
from spatialmath import SE3
from spatialmath import base as sb
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController


target_ee_state = EEState.from_tf( SE3(0.3, -0.5, 1.0) @ SE3.Rx(np.pi), 'ee_tool')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])

target_ee_state2 = EEState.from_tf(SE3(0.3, 0.1, 1.2) @ SE3.Rx(np.pi), 'ee_tool')
target_ee_state2.twist = np.array([0,0,-0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
target_joint_state = JointState.from_torque(np.zeros(6))

target_motion = Motion.from_states(target_joint_state, target_ee_state)
target_motion_s = copy.deepcopy(target_motion)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld('plane.urdf',gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01, time_scale=10.0)
        self.__sim.add_object('hole_round', 'urdf/hole_round.urdf', base_transform = SE3(0.3, -0.5, 0.8), fixed = True, save = True, scale_size=1.1)
        self.__sim.add_object('table', 'urdf/table.urdf', fixed = True, save = True)
        self.__robot = PyBulletRobot('urdf/ur5e_pybullet.urdf', SE3(0, -0.3, 0.625))

        self.__sim.add_robot(self.__robot, 'robot1')

        self.__controller_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_pose.connect_controller(self.__controller_speed)

        self.__sim.add_object('peg_round2', 'urdf/peg_round.urdf', base_transform = SE3(0.1, 0.0, 0.675), fixed = True, save = True)
        
    def __reset(self):
        self.__sim.reset()

    def test_add_tool(self):

        self.__robot.connect_tool('peg' ,'urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1))
        self.__sim.sim_step()

        print(self.__robot.ee_state('peg_target_link'))
        # time.sleep(10.0)

        while self.__sim.sim_time<55.0:
            ok = self.__controller_pose.send_control_to_robot(target_motion)
            self.assertTrue(ok)
            self.__sim.sim_step()
            print(self.__robot.ee_state('peg_link'))
        
        self.__robot.remove_tool('peg')
        self.__sim.sim_step()
        self.__robot.connect_tool('peg' ,'urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1), save=True)
        self.__sim.sim_step()
        while self.__sim.sim_time<80.0:
            ok = self.__controller_pose.send_control_to_robot(target_motion)
            self.assertTrue(ok)
            self.__sim.sim_step()

        self.__reset()
        while self.__sim.sim_time<100.0:
            ok = self.__controller_pose.send_control_to_robot(target_motion)
            self.assertTrue(ok)
            self.__sim.sim_step()
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
