import os
import sys
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.robot import EEState, JointState, Motion
from pybullet_world import PyBulletWorld, GUI_MODE
from pybullet_robot import PyBulletRobot
import unittest
from spatialmath import SE3
from spatialmath import base as sb
from utils.controllers import CartPositionToCartVelocityController, CartVelocityToJointVelocityController

target_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)
target_tf2 = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_ee_state = EEState.from_tf(target_tf, 'ee_tool')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]))
target_joint_state = JointState.from_torque(np.zeros(6))

target_motion = Motion.from_states(target_joint_state,target_ee_state)
target_motion2 = copy.deepcopy(target_motion)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld('plane.urdf',gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01)
        self.__sim.add_object('table', 'urdf/table.urdf')
        self.__robot = PyBulletRobot('urdf/ur5e_pybullet.urdf', SE3(0,0,0.625))
        self.__robot.apply_force_sensor('ee_tool')
        # self.__robot.set_jointcontrol(init_joint_state,'joint_positions')
        self.__controller_speed = CartVelocityToJointVelocityController(self.__robot)
        self.__controller_pose = CartPositionToCartVelocityController(self.__robot, 'world')
        self.__controller_pose.connect_controller(self.__controller_speed)

    def test_sim(self):
        self.__robot.ee_state('ee_tool')
        while self.__sim.sim_time<10.0:
            self.__sim.sim_step()
            # print(self.__robot.joint_state)
            if self.__sim.sim_time>5.0:
                self.__controller_pose.send_control_to_robot(target_motion)
        while self.__sim.sim_time<40.0:
            self.__sim.sim_step()
            # print(self.__robot.joint_state)
            self.__controller_speed.send_control_to_robot(target_motion2)
        
        # while self.__sim.sim_time<30.0:
        #     self.__sim.sim_step()
        #     # print(self.__robot.joint_state)
        #     self.__controller_speed.send_control_to_robot(target_state)
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
