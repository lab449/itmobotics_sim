import copy
import unittest
import time

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

import cv2

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController

CAMERA_LINK_NAME = 'camera_link'
TEST_TF = SE3(-0.6, 0.0, 1.0) @ SE3.Ry(np.pi)
TEST_JOINT_POSE = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])

controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

target_ee_state = EEState.from_tf(TEST_TF, CAMERA_LINK_NAME)
target_joint_state = JointState.from_position(TEST_JOINT_POSE)
target_motion = Motion.from_states(copy.deepcopy(target_joint_state), copy.deepcopy(target_ee_state))

class testPyBulletSim(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', save=True)
        self.__sim.add_object('hole_round', 'tests/urdf/hole_round.urdf', base_transform = SE3(-0.6, 0.0, 0.625), fixed = True, save = True)

        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,0,0.625) , 'robot')
        self.__robot.joint_controller_params = controller_params

        self.__controller_ee_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_ee_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_ee_pose.connect_controller(self.__controller_ee_speed)


    def test_camera(self):
        self.__sim.reset()
        self.__sim.connect_camera('eyehand_cam', 'robot', CAMERA_LINK_NAME)
        lg = np.array([237, 130, 0])
        ug = np.array([255, 145, 0])
        
        img, _ = self.__sim.get_image('eyehand_cam')
        img = img.astype(np.uint8)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gmask = cv2.inRange(hsv_img, lg ,ug)
        self.assertFalse(cv2.countNonZero(gmask)>0)

        self.__robot.reset_joint_state(target_joint_state)
        while self.__sim.sim_time<5.0:
            self.__controller_ee_pose.send_control_to_robot(target_motion)
            self.__sim.sim_step()
        img, _ = self.__sim.get_image('eyehand_cam')
        img = img.astype(np.uint8)
    
        self.assertTrue(np.sum(np.logical_and(lg <= img[650, 550], img[650, 550] <= ug).astype(int)))


def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
