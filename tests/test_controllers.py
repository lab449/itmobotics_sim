import os
import sys
import copy
from typing import final
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.math import vec2SE3
from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointPositionsController, JointTorquesController, JointVelocitiesController


controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

test_joint_pose = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
test_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_joint_state = JointState.from_position(test_joint_pose)

target_ee_state = EEState.from_tf(test_tf, 'ee_tool')
target_motion = Motion.from_states(copy.deepcopy(target_joint_state), copy.deepcopy(target_ee_state))

final_time = 3.0
n_samples = 5

class testControllers(unittest.TestCase):
    def setUp(self):

        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1.0)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', save=True)
        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,0,0.625), 'robot1')

        self.__robot.joint_controller_params = controller_params
        
        self.__controller_joint_torque = JointTorquesController(self.__robot)
        self.__controller_joint_speed = JointVelocitiesController(self.__robot)
        self.__controller_joint_pose = JointPositionsController(self.__robot)

        self.__controller_ee_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_ee_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_ee_pose.connect_controller(self.__controller_ee_speed)

        self.assertIsNotNone(self.__robot.joint_limits)
    

    def test_joint_torque_controller(self):
        for i in range(0, n_samples):
            # self.__sim.reset()
            self.__robot.reset_joint_state(target_joint_state)
            random_target_state = np.random.uniform(
                np.max(self.__robot.joint_limits.limit_torques[0]),
                np.min(self.__robot.joint_limits.limit_torques[1]),
                test_joint_pose.shape
            )
            print(random_target_state)
            target_motion.joint_state.joint_torques = random_target_state
            while self.__sim.sim_time<final_time*(i+1):
                ok = self.__controller_joint_torque.send_control_to_robot(target_motion)
                self.assertTrue(ok)
                self.__sim.sim_step()
            np.testing.assert_allclose(self.__robot.joint_state.joint_torques, random_target_state, rtol= 0.0, atol=1e-4)
    
    def test_joint_speed_controller(self):
        for i in range(0, n_samples):
            # self.__sim.reset()
            self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))
            random_target_state = np.random.uniform(-0.1, 0.1, test_joint_pose.shape)
            target_motion.joint_state.joint_velocities = random_target_state
            while self.__sim.sim_time<final_time*(i+1):
                ok = self.__controller_joint_speed.send_control_to_robot(target_motion)
                self.assertTrue(ok)
                self.__sim.sim_step()
            np.testing.assert_allclose(self.__robot.joint_state.joint_velocities, random_target_state, rtol= 0.0, atol=1e-3)
        
    def test_joint_pose_controller(self):
        for i in range(0, n_samples):
            # self.__sim.reset()
            self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))
            random_target_state = test_joint_pose + np.random.uniform(-np.pi/5, np.pi/5, test_joint_pose.shape)
            target_motion.joint_state.joint_positions = random_target_state
            while self.__sim.sim_time<final_time*(i+1):
                ok = self.__controller_joint_pose.send_control_to_robot(target_motion)
                self.assertTrue(ok)
                self.__sim.sim_step()
            np.testing.assert_allclose(self.__robot.joint_state.joint_positions, random_target_state, rtol= 0.0, atol=3e-1)
    
    def test_ee_speed_controller(self):
        for i in range(0, n_samples):
            # self.__sim.reset()
            self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))
            random_target_state = np.random.uniform(-0.05, 0.05, 6)
            target_motion.ee_state.twist = random_target_state
            while self.__sim.sim_time<final_time*(i+1):
                ok = self.__controller_ee_speed.send_control_to_robot(target_motion)
                self.assertTrue(ok)
                self.__sim.sim_step()
            ee_state = self.__robot.ee_state(target_motion.ee_state.ee_link,target_motion.ee_state.ref_frame)
            np.testing.assert_allclose(ee_state.twist, random_target_state, rtol= 0.0, atol=1e-3)
    
    def test_ee_pose_controller(self):
        for i in range(0, n_samples):
            # self.__sim.reset()
            self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))
            random_target_state = test_tf @ vec2SE3(np.random.uniform(-0.2, 0.2, 6))
            target_motion.ee_state.tf = random_target_state
            while self.__sim.sim_time<final_time*(i+1):
                ok = self.__controller_ee_pose.send_control_to_robot(target_motion)
                self.assertTrue(ok)
                self.__sim.sim_step()
            ee_state = self.__robot.ee_state(target_motion.ee_state.ee_link,target_motion.ee_state.ref_frame)
            np.allclose(ee_state.tf.A, random_target_state.A, atol=1e-5)
    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()