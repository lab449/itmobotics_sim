import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import time
import platform
import pybullet_robot
import enum
import pkgutil
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R

class GUI_MODE(enum.Enum):
    DIRECT = enum.auto()
    SIMPLE_GUI = enum.auto()
    

class PyBulletWorld():
    def __init__(self, urdf_filename: str ='plane.urdf', gui_mode: GUI_MODE = GUI_MODE.SIMPLE_GUI, time_step:float = 1e-3):
        self.__urdf_filename = urdf_filename
        self.__time_step = time_step

        self.__pybullet_gui_mode = p.DIRECT
        
        if gui_mode == GUI_MODE.SIMPLE_GUI:
            self.__pybullet_gui_mode = p.GUI
            self.__client = p.connect(self.__pybullet_gui_mode)
        else:
            self.__client = p.connect(self.__pybullet_gui_mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation()
        p.setGravity(0, 0, -9.82)
        p.setTimeStep(self.__time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.__time_step, numSolverIterations=100, numSubSteps=10)
        p.setRealTimeSimulation(False)
        self.__world_model = p.loadURDF(self.__urdf_filename, useFixedBase=True)
        self.__sim_time = 0.0
        self.__start_sim_time = time.time()
        self.__objects = {}
    
    def add_object(self, name:str, urdf_filename: str, base_transform: SE3 = SE3(), fixed=True):
        if name in self.__objects.keys():
            p.removeBody(self.__objects[name])
            print('Replace object with name {:s}'.format(name))
        base_pose = base_transform.t.tolist() # World position [x,y,z]
        base_orient = R.from_matrix(base_transform.R).as_quat().tolist() # Quaternioun [x,y,z,w]

        obj_id = p.loadURDF(
            urdf_filename,
            basePosition=base_pose,
            baseOrientation=base_orient,
            useFixedBase=True
        )
        self.__objects[name] = obj_id
        
    
    def sim_step(self):
        inloptime = time.time()-self.__start_sim_time
        p.stepSimulation()
        self.__sim_time = time.time()-self.__start_sim_time
        time.sleep(self.__time_step)

    @property
    def sim_time(self) -> float:
        return self.__sim_time
