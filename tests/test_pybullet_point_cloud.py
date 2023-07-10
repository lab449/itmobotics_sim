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

controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

target_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)
test_joint_pose = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])

target_ee_state = EEState.from_tf(target_tf, 'ee_link')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0, 0.0]))

target_motion = Motion.from_states(target_joint_state, target_ee_state)
target_motion2 = copy.deepcopy(target_motion)

class testPyBulletSim(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01, time_scale=1)
        self.__sim.add_object('table', 'tests/urdf/table.urdf', save=True)
        self.__sim.add_object('peg', 'tests/urdf/peg_round.urdf',SE3(0.5,0,0.65), save=True)

        self.__robot = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0,0,0.625) , 'robot')

        self.__robot.joint_controller_params = controller_params

    # @unittest.skip("Skip for CI")
    def test_camera(self):
        self.__sim.reset()
        self.__robot.connect_camera('base_cam', 'camera_link')
        self.__robot.reset_joint_state(JointState.from_position(test_joint_pose))

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
            points = self.__robot.get_point_cloud('base_cam')
            geometry.points =  o3d.utility.Vector3dVector(points)
            geometry.remove_non_finite_points(remove_nan=True, remove_infinite=True)
            print(geometry)
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()


def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()