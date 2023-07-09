import copy

import cv2
import numpy as np
from spatialmath import SE3

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.utils.controllers import JointVelocitiesController

controller_params = {'kp': np.array([12.0, 12.0, 12.0, 2.0, 2.0, 1.0]), 'kd': np.array([1.0, 5.0, 1.0, 0.05, 0.05, 0.05]) * 40}

test_joint_pose = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
test_tf = SE3(0.3, 0.0, 1.0) @ SE3.Rx(np.pi)

target_joint_state = JointState.from_position(test_joint_pose)

target_ee_state = EEState.from_tf(test_tf, 'ee_tool')
target_motion = Motion.from_states(copy.deepcopy(target_joint_state), copy.deepcopy(target_ee_state))

def main():
    __sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01, time_scale=1)
    __sim.add_object('table', 'tests/urdf/table.urdf', save=True)
    __sim.add_object('socket', 'example/urdf/socket.urdf', SE3(0.4,0.2,0.65), save=True)


    __robot = __sim.add_robot('example/urdf/ur5e_pybullet.urdf', SE3(0,0,0.625) , 'robot')
    __robot.joint_controller_params = controller_params
    __controller_joint_speed = JointVelocitiesController(__robot)

    __sim.reset()

    __robot.connect_tool('peg' ,'example/urdf/plug.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0), save=True)
    __robot.connect_camera('base_cam', 'camera_link')

    __robot.reset_joint_state(JointState.from_position(test_joint_pose))

    while True: 
        ok = __controller_joint_speed.send_control_to_robot(target_motion)
        __sim.sim_step()
        points = __robot.get_image('base_cam')
    
if __name__ == "__main__":
    main()
