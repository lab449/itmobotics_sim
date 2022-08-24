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

target_pose_motion = Motion.from_states(target_joint_state,target_ee_state)
target_speed_motion = copy.deepcopy(target_pose_motion)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01)
        self.__sim.add_object('hole_round', 'tests/urdf/hole_round.urdf', base_transform = SE3(0.0, 0.0, 0.675), fixed = True, save = True)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', fixed =True, save=True)
        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,-0.3,0.625), 'robot1')
        self.__robot2 = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0.0,0.3,0.625), 'robot2')

        self.__controller_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_pose.connect_controller(self.__controller_speed)

        self.__sim.add_object('hole_round2', 'tests/urdf/hole_round.urdf', base_transform = SE3(0.1, 0.0, 0.675), fixed = True, save = True, scale_size = 1.1)


    def test_reset(self):
        self.assertIsNotNone(self.__robot.ee_state('ee_tool'))
        self.__sim.reset()

        self.assertIsNotNone(self.__robot.robot_id)
        self.assertIsNotNone(self.__robot2.robot_id)
        self.assertNotEqual(self.__robot.robot_id, self.__robot2.robot_id)
        while self.__sim.sim_time<10.0:
            ok = self.__controller_pose.send_control_to_robot(target_pose_motion)
            self.assertTrue(ok)
            self.__sim.sim_step()
        np.allclose(target_pose_motion.ee_state.tf.A, self.__robot.ee_state(target_pose_motion.ee_state.ee_link).tf.A, atol=1e-5)

        new_joint_state = self.__robot.joint_state
        self.__sim.reset()
        self.assertNotEqual(self.__robot.robot_id, self.__robot2.robot_id)
        self.assertIsNotNone(self.__robot.robot_id)
        self.assertIsNotNone(self.__robot2.robot_id)

        self.__robot.reset_joint_state(new_joint_state)
        self.assertEqual(new_joint_state, self.__robot.joint_state)
        
        self.__robot.reset_ee_state(target_pose_motion.ee_state)
        self.assertEqual(target_pose_motion.ee_state, self.__robot.ee_state(target_pose_motion.ee_state.ee_link))

        while self.__sim.sim_time<10.0:
            ok = self.__controller_speed.send_control_to_robot(target_speed_motion)
            self.assertTrue(ok)
            self.__sim.sim_step()
        np.allclose(target_speed_motion.ee_state.twist, self.__robot.ee_state(target_speed_motion.ee_state.ee_link).twist, atol=1e-5)
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
