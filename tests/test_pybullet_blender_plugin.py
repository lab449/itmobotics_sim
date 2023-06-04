import copy
import time
import unittest

import numpy as np
from spatialmath import SE3
from spatialmath import base as sb

from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE

def main():
    sim = PyBulletWorld(gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01, time_scale=1)
    sim.add_object('table', 'tests/urdf/table.urdf')
    robot = sim.add_robot('tests/urdf/iiwa_airhockey.urdf', SE3(0,-0.2,0.625) , 'robot')
    # robot2 = sim.add_robot('tests/urdf/ur5e_pybullet2.urdf', SE3(0,0.2,0.625) , 'robot2')
    sim.register_objects_for_record()
    while sim.sim_time<10.0:
        sim.sim_step()
        # ee_state = robot.ee_state('iiwa_1/striker_mallet_tip')
    sim.save_scene_record('blender_scene.pkl')
if __name__ == "__main__":
    main()
