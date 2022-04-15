import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import time
import enum
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from .pybullet_robot import PyBulletRobot

class GUI_MODE(enum.Enum):
    DIRECT = enum.auto()
    SIMPLE_GUI = enum.auto()
    

class PyBulletWorld():
    def __init__(self, urdf_filename: str ='plane.urdf', gui_mode: GUI_MODE = GUI_MODE.SIMPLE_GUI, time_step:float = 1e-3, time_scale:float = 1):
        self.__urdf_filename = urdf_filename
        self.__time_step = time_step
        self.__time_scale = max(time_scale, 1.0)
        assert self.__time_scale < 1e3, "Large time scale doesn't support, please chose less than 1e3"
        assert self.__time_step < 1.0, "Large time step doesn't support, please chose less than 1.0 sec"
        self.__robots = []

        self.__pybullet_gui_mode = p.DIRECT
        
        if gui_mode == GUI_MODE.SIMPLE_GUI:
            self.__pybullet_gui_mode = p.GUI
        self.__client = p.connect(self.__pybullet_gui_mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.__objects = {}
        self.reset()
    
    def add_robot(self, robot: PyBulletRobot):
        self.__robots.append(robot)
    
    def add_object(self, name:str, urdf_filename: str, base_transform: SE3 = SE3(), fixed=True, save=False):
        if name in self.__objects.keys():
            self.remove_object(name)
            print('Replace object with name {:s}'.format(name))
        self.__append_object(name, urdf_filename, base_transform, fixed, save)
    
    def __append_object(self, name:str, urdf_filename: str, base_transform: SE3, fixed: bool, save: bool):
        base_pose = base_transform.t.tolist() # World position [x,y,z]
        base_orient = R.from_matrix(base_transform.R).as_quat().tolist() # Quaternioun [x,y,z,w]
        # print("Load %s, in %s", name, base_pose)
        obj_id = p.loadURDF(
            urdf_filename,
            basePosition=base_pose,
            baseOrientation=base_orient,
            useFixedBase=fixed
        )
        self.__objects[name] = {"id": obj_id, "urdf_filename": urdf_filename, "base_tf": base_transform, "fixed": fixed, "save": save}
    
    def remove_object(self, name:str):
        assert name in self.__objects, "Undefined object: {:s}".format(name)
        p.removeBody(self.__objects[name]["id"])
        del self.__objects[name]

    def sim_step(self):
        p.stepSimulation()
        self.__sim_time = time.time()-self.__start_sim_time
        time.sleep(self.__time_step/self.__time_scale)
    
    
    def reset(self):
        for r in self.__robots:
            r.remove_robot_body()
        p.resetSimulation()
        p.setGravity(0, 0, -9.82)
        p.setTimeStep(self.__time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.__time_step, numSolverIterations=100, numSubSteps=10)
        p.setRealTimeSimulation(False)

        self.__world_model = p.loadURDF(self.__urdf_filename, useFixedBase=True)
        self.__sim_time = 0.0
        self.__start_sim_time = time.time()
        for n in self.__objects:
            obj = dict(self.__objects[n])
            if obj["save"]:
                self.__append_object(n, obj["urdf_filename"], obj["base_tf"], obj["fixed"], obj["save"])
        
        print(self.__objects)

    @property
    def sim_time(self) -> float:
        return self.__sim_time

    @property
    def client(self) -> float:
        return self.__client
