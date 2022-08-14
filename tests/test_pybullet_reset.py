import os
import sys
import numpy as np
import copy

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
import unittest
from spatialmath import SE3
from spatialmath import base as sb
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController


target_ee_state = EEState.from_tf( SE3(0.3, -0.5, 1.2) @ SE3.Rx(np.pi) , 'ee_tool')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])

target_ee_state2 = EEState.from_tf(SE3(0.3, 0.1, 1.2) @ SE3.Rx(np.pi), 'ee_tool')
target_ee_state2.twist = np.array([0,0,-0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
target_joint_state = JointState.from_torque(np.zeros(6))

target_motion = Motion.from_states(target_joint_state,target_ee_state)
target_motion_s = copy.deepcopy(target_motion)

target_motion2 = Motion.from_states(target_joint_state,target_ee_state2)
target_motion2_s = copy.deepcopy(target_motion2)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01)
        self.__sim.add_object('hole_round', 'urdf/hole_round.urdf', base_transform = SE3(0.0, 0.0, 0.675), fixed = True, save = True)
        self.__sim.add_object('table', 'urdf/table.urdf', fixed =True, save=True)
        self.__robot = PyBulletRobot('urdf/ur5e_pybullet.urdf', SE3(0,-0.3,0.625))
        self.__robot2 = PyBulletRobot('urdf/ur5e_pybullet.urdf', SE3(0.0,0.3,0.625))
        self.__sim.add_robot(self.__robot, 'robot1')
        self.__sim.add_robot(self.__robot2, 'robot2')

        self.__controller_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_pose.connect_controller(self.__controller_speed)

        self.__controller_speed2 = EEVelocityToJointVelocityController(self.__robot2)
        self.__controller_pose2 = EEPositionToEEVelocityController(self.__robot2)
        self.__controller_pose2.connect_controller(self.__controller_speed2)

        self.__sim.add_object('hole_round2', 'urdf/hole_round.urdf', base_transform = SE3(0.1, 0.0, 0.675), fixed = True, save = True, scale_size = 1.1)


    def __reset(self):
        self.__sim.reset()
        self.__robot.reset()
        self.__robot2.reset()

    def test_reset(self):
        self.__robot.ee_state('ee_tool')
        self.__reset()
        self.assertNotEqual(self.__robot.robot_id, self.__robot2.robot_id)
        self.assertIsNotNone(self.__robot.robot_id)
        self.assertIsNotNone(self.__robot2.robot_id)
        while self.__sim.sim_time<10.0:
            ok = self.__controller_pose.send_control_to_robot(target_motion)
            ok &= self.__controller_pose2.send_control_to_robot(target_motion2)
            self.assertTrue(ok)
            self.__sim.sim_step()

        jstate = self.__robot.joint_state
        self.__reset()
        self.assertNotEqual(self.__robot.robot_id, self.__robot2.robot_id)
        self.assertIsNotNone(self.__robot.robot_id)
        self.assertIsNotNone(self.__robot2.robot_id)
        self.__robot.reset_joint_state(jstate)
        result = self.__robot.joint_state
        self.assertEqual(jstate, result)
        
        self.__robot2.reset_ee_state(target_motion2.ee_state)
        result = self.__robot2.ee_state(target_motion2.ee_state.ee_link)
        self.assertEqual(target_motion2.ee_state, result)

        while self.__sim.sim_time<10.0:
            ok = self.__controller_pose.send_control_to_robot(target_motion)
            ok &= self.__controller_pose2.send_control_to_robot(target_motion2)
            self.assertTrue(ok)
            self.__sim.sim_step()
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
