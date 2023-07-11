import copy
import unittest
import time

import cv2
import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
import open3d as o3d
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController


controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

test_joint_pose = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
test_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_joint_state = JointState.from_position(test_joint_pose)

target_ee_state = EEState.from_tf(test_tf, 'ee_tool')
target_motion = Motion.from_states(copy.deepcopy(target_joint_state), copy.deepcopy(target_ee_state))


class testPyBulletSim(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', save=True)
        self.__sim.add_object('peg', 'tests/urdf/peg_round.urdf',SE3(0.5,-0.3,0.65), save=True)
        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,0,0.625) , 'robot')
        self.__robot.joint_controller_params = controller_params
        self.__controller_ee_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_ee_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_ee_pose.connect_controller(self.__controller_ee_speed)

    @unittest.skip("Skip for CI")
    def test_camera(self):
        self.__sim.reset()
        self.__robot.connect_camera('base_cam', 'camera_link')
        self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))

        geometry = o3d.geometry.PointCloud()
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
            self.__controller_ee_pose.send_control_to_robot(target_motion)

            points, img = self.__robot.get_point_cloud('base_cam')
            geometry.points =  o3d.utility.Vector3dVector(points)
            if len(geometry.points) > 0:
                # cv2.imwrite(f"res/ex_{self.__sim.sim_time}.png",img)
                # o3d.io.write_point_cloud(f"res/ex_{self.__sim.sim_time}.pcd",geometry)
                print(geometry)


def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
