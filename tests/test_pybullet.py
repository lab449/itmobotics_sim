import pybullet as p
import time
import pybullet_data
import numpy as np


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startPos_hole = [0,0.3,0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("tests/urdf/ur5e_pybullet.urdf",startPos, startOrientation)
p.resetBasePositionAndOrientation(robot_id, startPos, startOrientation)
for i in range(0, p.getNumJoints(robot_id)):
    print(p.getJointInfo(robot_id, i))
p.stepSimulation()
obj_id = p.loadURDF("tests/urdf/hole_round.urdf", startPos_hole, startOrientation)
print(p.getNumJoints(obj_id))
print(p.getJointInfo(obj_id, 0))
joint_pose = np.zeros(6)
for i in range (1000):
    time.sleep(1./240.)
    jac_t, jac_r = p.calculateJacobian(
        robot_id, 6, [0,0,0],
        list(joint_pose), list(np.zeros(joint_pose.shape)),
        list(np.zeros(joint_pose.shape))
    )
    Jv = np.asarray(jac_t)
    Jw = np.asarray(jac_r)
        
    J = np.concatenate((Jv,Jw), axis=0)
p.disconnect()