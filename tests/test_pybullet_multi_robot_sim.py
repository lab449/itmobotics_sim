import numpy as np
import time

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.utils.math import vec2SE3
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
import unittest
from spatialmath import SE3
from itmobotics_sim.utils.controllers import EEPositionToEEVelocityController, EEVelocityToJointVelocityController, JointTorquesController

controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

test_tf = SE3(-0.8, 0.0, 0.625)
test_ee_tf = SE3(0.2, 0.2, 0.625) @ SE3.Rx(np.pi)

test_joint_pose = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
target_joint_state = JointState.from_position(test_joint_pose)

final_time = 10.0
n_agents = 5

class testPyBulletMlultiRobot(unittest.TestCase):
    def setUp(self):
        self.__sim = PyBulletWorld(gui_mode = GUI_MODE.DIRECT, time_step = 0.01)
        self.__sim.add_object('table', 'tests/urdf/table.urdf')

        self.__agents = []
        for i in range(0, n_agents):
            roboti = self.__sim.add_robot('tests/urdf/ur5e_pybullet.urdf', SE3(0.4*i, 0.0, 0.0) @test_tf, 'robot' + str(i))
            roboti.joint_controller_params = controller_params

            controller_speed = EEVelocityToJointVelocityController(roboti)
            controller_pose = EEPositionToEEVelocityController(roboti)
            controller_pose.connect_controller(controller_speed)

            self.__agents.append(
                {
                    'robot': roboti,
                    'controoller_pose': controller_pose,
                    'conroller_speed': controller_speed
                }
            )

    # @unittest.skip("Temporal skip")
    def test_reset_state(self):
        new_ee_tf = vec2SE3(np.random.uniform(-0.2, 0.2, 6)) @test_ee_tf
        for a in self.__agents:
            target_state = EEState.from_tf(new_ee_tf, 'ee_tool', 'base_link')
            a['robot'].reset_ee_state(target_state)
            self.assertEqual(target_state, a['robot'].ee_state(target_state.ee_link, 'base_link'))

        new_joint_state = test_joint_pose + np.random.uniform(-np.pi/5, np.pi/5, test_joint_pose.shape)
        for a in self.__agents:
            target_state = JointState.from_position(new_joint_state)
            a['robot'].reset_joint_state(target_state)
            self.assertEqual(target_state, a['robot'].joint_state)
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
