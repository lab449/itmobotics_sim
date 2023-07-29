import os, sys
import time
import enum

import numpy as np
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation as R

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data

from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot, SimulationException
from itmobotics_sim.utils import robot
from itmobotics_sim.utils import converters
from itmobotics_sim.pybullet_env.pybullet_recorder import PyBulletRecorder

class GUI_MODE(enum.Enum):
    DIRECT = enum.auto()
    SIMPLE_GUI = enum.auto()

class PyBulletWorld():
    def __init__(self, gui_mode: GUI_MODE = GUI_MODE.SIMPLE_GUI, time_step:float = 1e-3, time_scale:float = 1):
        self.__time_step = time_step
        self.__time_scale = max(time_scale, 1.0)
        assert self.__time_scale < 1e5, "Large time scale doesn't support, please choose less than 1e5"
        assert self.__time_step < 1.0, "Large time step doesn't support, please choose less than 1.0 sec"
        self.__robots = {}

        self.__pybullet_gui_mode = pybullet.DIRECT
        self.__blender_recorder = None

        if gui_mode == GUI_MODE.SIMPLE_GUI:
            self.__pybullet_gui_mode = pybullet.GUI
        
        self.__blender_recorder = PyBulletRecorder()
        self.__recording = False

        self.__p = bc.BulletClient(connection_mode=self.__pybullet_gui_mode)

        self.__p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.additional_paths = [pybullet_data.getDataPath()]

        self.__objects = {}
        self.__cameras = {}
        self.reset()

    def __del__(self):
        print("Pybullet disconnecting")
        self.__p.disconnect()
        del self.__p
        del self.__robots
        del self.__objects
        del self.__cameras
    
    def add_robot(self, urdf_filename: str, base_transform: SE3 = SE3(), name: str = 'robot', fixed: bool = True, self_collide: bool = True) -> PyBulletRobot:
        if name in self.__robots.keys():
            raise SimulationException('A robot with that name ({:s}) already exists'.format(name))
        self.__robots[name] = PyBulletRobot(
            urdf_filename,
            self.__p,
            base_transform,
            additional_path = self.additional_paths,
            fixed_base=fixed,
            use_self_collision=self_collide
        )
        return self.__robots[name]
    
    def add_object(self, name:str, urdf_filename: str, base_transform: SE3 = SE3(), fixed: bool = True, save: bool = False, scale_size: float = 1.0):
        if name in self.__objects.keys():
            self.remove_object(name)
            print('Replace object with name {:s}'.format(name))
        self.__append_object(name, urdf_filename, base_transform, fixed, save, scale_size)
    
    def connect_camera(self, 
        name: str, 
        model_name: str, 
        link_name: str, 
        resolution: tuple = (1280, 1024), 
        clip: tuple = (0.001, 5.0), 
        intrinsic_matrix: np.ndarray = None,
        fps: int = 25
    ):
        if intrinsic_matrix is None:
            default_fov_x = resolution[0]/2.0*1.2
            default_fov_y = resolution[1]/2.0*1.2
            default_cx = resolution[0]/2
            default_cy = resolution[1]/2
            intrinsic_matrix = np.array([[default_fov_x,           0 , default_cx],
                                        [0,            default_fov_y , default_cy],
                                        [0,                        0 ,         1 ]])
        self.__cameras[name] = {
            'intrinsic_matrix': intrinsic_matrix,
            'link': link_name,
            'model': model_name,
            'resolution': resolution,
            'clip': clip,
            'fps': fps,
            'time_frame': -1,
            'last_frame': np.zeros(1)
        }
    
    def get_image(self, camera_name: str) -> np.ndarray:
        assert camera_name in self.__cameras, SimulationException(
            'Camera {:s} is not connected, please use connect_camera() before that!'. format(camera_name)
        )
        if (self.sim_time - self.__cameras[camera_name]['time_frame']) > 1.0/self.__cameras[camera_name]['fps']:
                    # camera view_matrix:
            view_matrix = converters.extrinsic2GLview_matrix(
                self.link_state(self.__cameras[camera_name]["model"],self.__cameras[camera_name]["link"]).tf.A
            )

            color, depth, segmask = self.__p.getCameraImage(
                width=self.__cameras[camera_name]['resolution'][0],
                height=self.__cameras[camera_name]['resolution'][1],
                viewMatrix=view_matrix,
                shadow=0,
                projectionMatrix=converters.intrinsic2GLprojection_matrix(
                    self.__cameras[camera_name]['intrinsic_matrix'],
                    self.__cameras[camera_name]['resolution'],
                    self.__cameras[camera_name]['clip']
                ),
                renderer=self.__p.ER_TINY_RENDERER,
                flags=self.__p.ER_NO_SEGMENTATION_MASK
            )[2:5]
            self.__cameras[camera_name]['last_frame'] = [
                (
                    np.reshape(color, 
                        (self.__cameras[camera_name]['resolution'][1], self.__cameras[camera_name]['resolution'][0], 4)
                    )[..., :3]
                ),
                np.reshape(
                    depth, (self.__cameras[camera_name]['resolution'][1], self.__cameras[camera_name]['resolution'][0])
                )
            ]
            self.__cameras[camera_name]['time_frame'] = self.sim_time
        return self.__cameras[camera_name]['last_frame']

    def get_point_cloud(self, camera_name: str) -> np.ndarray:
        self.get_image(camera_name)
        
        view_matrix = converters.extrinsic2GLview_matrix(
            self.link_state(self.__cameras[camera_name]["model"],self.__cameras[camera_name]["link"]).tf.A
        )

        depth = self.__cameras[camera_name]['last_frame'][1]
        proj_matrix = np.asarray(
            converters.intrinsic2GLprojection_matrix(
                self.__cameras[camera_name]['intrinsic_matrix'],
                self.__cameras[camera_name]['resolution'],
                self.__cameras[camera_name]['clip']
            )
        ).reshape([4, 4], order="F")
        Tc = np.array([[1,   0,    0,  0],
                    [0,  -1,    0,  0],
                    [0,   0,   -1,  0],
                    [0,   0,    0,  1]]).reshape(4,4)

        tran_pix_camera = np.linalg.pinv(np.matmul(proj_matrix, Tc))

        # create a grid with pixel coordinates and depth values
        width = self.__cameras[camera_name]['resolution'][0]
        height = self.__cameras[camera_name]['resolution'][1]
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < self.__cameras[camera_name]['clip'][1]]
        pixels = pixels[z > self.__cameras[camera_name]['clip'][0]]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to camera coordinates
        points = np.matmul(tran_pix_camera, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points
    
    def __append_object(self, name:str, urdf_filename: str, base_transform: SE3, fixed: bool, save: bool, scale_size: float, enable_ft: bool = False):
        base_pose = base_transform.t.tolist() # World position [x,y,z]
        base_orient = R.from_matrix(base_transform.R).as_quat().tolist() # Quaternioun [x,y,z,w]
        obj_id = self.__p.loadURDF(
            urdf_filename,
            basePosition=base_pose,
            baseOrientation=base_orient,
            useFixedBase=fixed,
            globalScaling=scale_size
        )
        num_joints = self.__p.getNumJoints(obj_id)
        link_id = {}
        for _id in range(num_joints):
            _name = self.__p.getJointInfo(obj_id, _id)[12].decode('UTF-8')
            link_id[_name] = _id
            if enable_ft:
                self.__p.enableJointForceTorqueSensor(obj_id, _id, 1)
        
        self.__objects[name] = {
            "id": obj_id,
            "urdf_filename": urdf_filename,
            "base_tf": base_transform,
            "fixed": fixed,
            "save": save,
            "link_id": link_id,
            "scale_size": scale_size,
            "enable_ft": enable_ft
        }
    
    def remove_object(self, name: str):
        assert name in self.__objects, "Undefined object: {:s}".format(name)
        self.__p.removeBody(self.__objects[name]["id"])
        del self.__objects[name]

    def remove_robot(self, name: str):
        assert name in self.__robots, "Undefined object: {:s}".format(name)
        self.__p.removeBody(self.__robots[name].robot_id)
        del self.__robots[name]

    def link_state(self, model_name: str, link: str, reference_model_name: str = "", reference_link: str = "global") -> robot.EEState:
        link_state = robot.EEState.from_tf(SE3(0.0, 0.0, 0.0), ee_link=link, ref_link=reference_link)
        if link != 'global':
            try:
                if model_name in self.__objects:
                    pr = self.__p.getLinkState(self.__objects[model_name]["id"], self.__objects[model_name]["link_id"][link], computeLinkVelocity=1)
                    pb_joint_state = self.__p.getJointState(self.__objects[model_name]["id"], self.__objects[model_name]["link_id"][link])
                elif model_name in self.__robots:
                    pr = self.__p.getLinkState(self.__robots[model_name].robot_id, self.__robots[model_name].link_id(link), computeLinkVelocity=1)
                    pb_joint_state = self.__p.getJointState(self.__robots[model_name].robot_id, self.__robots[model_name].link_id(link))
                else:
                    raise KeyError(
                        'Unknown model name. Please check that object or robot model has been added to the simulator \
                        with name: {:s}.\n List of added robot models: {:s}.\n List of added object models: {:s}'.format(
                            model_name,
                            str(list(self.__robots.keys())),
                            str(list(self.__objects.keys()))
                        )
                    )
            except KeyError:
                raise KeyError(
                    "Unknown link id for link: {:s} in model: {:s}. \Please check target link and model name. Check that required tool was connected".format(
                        link,
                        model_name
                    )
                )
            _,_,_,_, link_frame_pos, link_frame_rot, link_frame_pos_vel, link_frame_rot_vel = pr
            link_state.tf = SE3(*link_frame_pos) @ SE3(SO3(R.from_quat(link_frame_rot).as_matrix(), check=False))
            link_state.twist = np.concatenate([link_frame_pos_vel, link_frame_rot_vel])
            link_state.force_torque = np.array(pb_joint_state[2])
            
        if reference_model_name == "" or reference_link == "global":
            return link_state

        if reference_model_name in self.__objects:
            pr = self.__p.getLinkState(self.__objects[reference_model_name]["id"], self.__objects[reference_model_name]["link_id"][reference_link], computeLinkVelocity=1)
        elif reference_model_name in self.__robots:
            pr = self.__p.getLinkState(self.__robots[reference_model_name].robot_id, self.__robots[reference_model_name].link_id(reference_link), computeLinkVelocity=1)
        else:
            raise SimulationException(
                'Unknown reference model name. Please check that object or robot model has been added to the simulator\
                with name: {:s}.\n List of added robot models: {:s}.\n List of added object models: {:s}'.format(
                    reference_model_name,
                    str(list(self.__robots.keys())),
                    str(list(self.__objects.keys()))
                )
            )
        _,_,_,_, ref_frame_pos, ref_frame_rot, ref_frame_pos_vel, ref_frame_rot_vel = pr
        ref_frame_twist = np.concatenate([ref_frame_pos_vel, ref_frame_rot_vel])

        link_state.tf = (SE3(*ref_frame_pos) @ SE3(SO3(R.from_quat(ref_frame_rot).as_matrix(), check=False))).inv() @ link_state.tf
        rotation_6d = np.kron(np.eye(2,dtype=int), R.from_quat(ref_frame_rot).inv().as_matrix())
        link_state.twist = rotation_6d @ (link_state.twist - ref_frame_twist)
        link_state.force_torque = rotation_6d @ link_state.force_torque

        return link_state
    
    def is_collide_with(self, model_name: str, tollerance: float = 0.001):
        collision_list = []
        if model_name in self.__robots:
            modelA_id = self.__robots[model_name].robot_id
        elif model_name in self.__objects:
            modelA_id = self.__objects[model_name]['id']
        
        for modelB_name in self.__objects:
            if modelA_id!= self.__objects[modelB_name]['id']:
                closest_points = self.__p.getClosestPoints(
                    modelA_id,
                    bodyB = self.__objects[modelB_name]['id'],
                    distance = tollerance
                )
                if len(closest_points)>0:
                    collision_list.append(modelB_name)
        for modelB_name in self.__robots:
            if modelA_id!= self.__robots[modelB_name].robot_id:
                closest_points = self.__p.getClosestPoints(
                    modelA_id,
                    bodyB = self.__robots[modelB_name].robot_id,
                    distance = tollerance
                )
                if len(closest_points)>0:
                    collision_list.append(modelB_name)

        return collision_list

    def sim_step(self):
        self.__p.stepSimulation()
        self.__sim_time += self.__time_step
        if self.__recording:
            self.__blender_recorder.add_keyframe()
        if self.__pybullet_gui_mode == pybullet.GUI:
            dt = max(self.__time_step/self.__time_scale - (self.__last_real_time - time.time()), 0)
            time.sleep(dt)
        self.__last_real_time = time.time()
    
    @property
    def time_step(self):
        return self.__time_step
    
    
    def reset(self):

        for r in self.__robots.keys():
            self.__robots[r].clear_id()

        self.__p.resetSimulation()
        self.__p.setGravity(0, 0, -9.82)
        self.__p.setTimeStep(self.__time_step)
        self.__p.setPhysicsEngineParameter(fixedTimeStep=self.__time_step, numSolverIterations=100, numSubSteps=4)
        self.__p.setRealTimeSimulation(False)
        
        for r in self.__robots.keys():
            self.__robots[r].reset()

        self.__sim_time = 0.0
        self.__last_real_time = time.time()
        
        for n in self.__objects:
            obj = dict(self.__objects[n])
            if obj["save"]:
                self.__append_object(
                    n,
                    obj["urdf_filename"],
                    obj["base_tf"],
                    obj["fixed"],
                    obj["save"],
                    obj["scale_size"],
                    obj["enable_ft"]
                )
        
        self.__blender_recorder.reset()
    
    def add_additional_search_path(self, path: str) -> None:
        self.__p.setAdditionalSearchPath(path)
        self.additional_paths.append(path)

    def get_robot(self, robot_name: str) -> PyBulletRobot:
        return self.__robots[robot_name]

    def register_objects_for_record(self):
        self.__blender_recorder.reset()
        if self.__blender_recorder is None:
            print('Blender is not active')
            return

        # Add objects observer
        for robot in self.__robots.values():
            self.__blender_recorder.register_object(robot.robot_id, robot.urdf_filename)
        for obj in self.__objects.values():
            self.__blender_recorder.register_object(obj["id"], obj["urdf_filename"])

    def save_scene_record(self, filename) -> bool:
        if self.__blender_recorder is None:
            print('Blender is not active')
            return False

        self.__blender_recorder.save(filename)
        return True
    
    def start_record(self):
        self.__recording = True
    def stop_record(self):
        self.__recording = False
    
    @property
    def robot_names(self) -> list[str]:
        return list(self.__robots.keys())

    @property
    def sim_time(self) -> float:
        return self.__sim_time

    @property
    def client(self) -> float:
        return self.__p
