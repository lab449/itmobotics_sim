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


target_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)
target_tf2 = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_ee_state = EEState.from_tf(target_tf, 'iiwa_link_ee')
target_ee_state.twist = np.array([0,0,0.01,0,0,0])


target_joint_state = JointState.from_position(np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0, 0.0]))

target_motion = Motion.from_states(target_joint_state, target_ee_state)
target_motion2 = copy.deepcopy(target_motion)

class testPyBulletRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01, time_scale=5)
        self.__sim.add_object('table', 'tests/urdf/table.urdf')
        self.__robot = PyBulletRobot('tests/urdf/iiwa14_pybullet.urdf', SE3(0,0,0.625))
        self.__sim.add_robot(self.__robot, 'robot1')
        self.__controller_speed = EEVelocityToJointVelocityController(self.__robot)
        self.__controller_pose = EEPositionToEEVelocityController(self.__robot)
        self.__controller_pose.connect_controller(self.__controller_speed)
        self.__controller_torque = JointTorquesController(self.__robot)

        self.assertIsNotNone(self.__robot.joint_limits)
        print(self.__robot.joint_limits)

    @unittest.skip
    def test_clip_joint_state(self):
        self.__sim.sim_step()
        js = self.__robot.joint_state
        self.assertIsNotNone(js)
        self.__robot.joint_limits.clip_joint_state(js)
        self.assertTrue(np.all(js.joint_positions > self.__robot.joint_limits.limit_positions[0]))
        self.assertTrue(np.all(js.joint_positions < self.__robot.joint_limits.limit_positions[1]))

        self.assertTrue(np.all(js.joint_velocities > self.__robot.joint_limits.limit_velocities[0]))
        self.assertTrue(np.all(js.joint_velocities < self.__robot.joint_limits.limit_velocities[1]))

        self.assertTrue(np.all(js.joint_torques > self.__robot.joint_limits.limit_torques[0]))
        self.assertTrue(np.all(js.joint_torques < self.__robot.joint_limits.limit_torques[1]))

    def test_sim(self):
        self.assertIsNotNone(self.__robot.ee_state('iiwa_link_ee'))        
        
        while self.__sim.sim_time<15.0:
            self.__sim.sim_step()
            # print(self.__robot.joint_state)
            if self.__sim.sim_time>3.0:
                ok = self.__controller_pose.send_control_to_robot(target_motion)
                self.assertTrue(ok)
        while self.__sim.sim_time<35.0:
            self.__sim.sim_step()
            # print(self.__robot.joint_state)
            if self.__sim.sim_time>20.0:
                ok = self.__controller_speed.send_control_to_robot(target_motion2)
                self.assertTrue(ok)
        while self.__sim.sim_time<55.0:
            self.__sim.sim_step()
            # print(self.__robot.joint_state)
            if self.__sim.sim_time>40.0:
                ok = self.__controller_torque.send_control_to_robot(target_motion2)
                self.assertTrue(ok)
    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
