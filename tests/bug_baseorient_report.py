import pybullet as p
import numpy as np
import unittest
from scipy.spatial.transform import Rotation as R

class TestBaseOrient(unittest.TestCase):
    def setUp(self):
        pc = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.82)
        p.setTimeStep(0.001)
        p.setPhysicsEngineParameter(fixedTimeStep=0.001, numSolverIterations=100, numSubSteps=4)

        self.__urdf_filename = 'urdf/iiwa14_pybullet.urdf'
        self.__ee_link_name = 'iiwa_link_ee'
        self.__base_pose = [0,0,0]
        self.__target_position = [0.6, 0.2, 0.4]
        self.__target_orient = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat().tolist()

    def test_zero_baseorient(self):
        base_orient = [0,0,0,1]
        robot_id = p.loadURDF(
            self.__urdf_filename,
            basePosition = self.__base_pose,
            baseOrientation = base_orient,
            useFixedBase = True
        )
        num_joints = p.getNumJoints(robot_id)

        actuators_id_list = []
        links_id_by_name = {}
        for _id in range(num_joints):
            joint_info = p.getJointInfo(robot_id, _id)
            _name = joint_info[12].decode('UTF-8')
            if joint_info[4] != -1:
                actuators_id_list.append(_id)
            links_id_by_name[_name] = _id
        print("Actuators id list: ", actuators_id_list)

        p.stepSimulation()
        ik_joints = p.calculateInverseKinematics(robot_id, links_id_by_name[self.__ee_link_name], self.__target_position, self.__target_orient)
        for i,a in actuators_id_list:
            p.resetJointState(robot_id, a, self._joint_state.joint_positions[i])
        fk_pose = p.
        p.resetSimulation()
    
    # def test_nonzero_baseorient(self):

    def solve_ik(self):

    
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()
