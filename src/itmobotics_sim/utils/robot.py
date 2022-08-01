from __future__ import annotations
from enum import Enum, auto

import numpy as np
from abc import ABC, abstractmethod
from spatialmath import SE3, SO3

class RobotControllerType(Enum):
    JOINT_POSITIONS = 'joint_positions'
    JOINT_VELOCITIES = 'joint_velocities'
    JOINT_TORQUES = 'joint_torques'
    TF = 'tf'
    TWIST = 'twist'

class JointState():
    def __init__(self, num_joints: int):
        self.__num_joints = num_joints
        self.__joint_positions: np.ndarray = None
        self.__joint_velocities: np.ndarray = None
        self.__joint_torques: np.ndarray =  np.zeros(num_joints)

    def __str__(self):
        out = 'Position: {:s},\nVelocity: {:s},\nTorque: {:s}'.format(
            str(np.round(self.__joint_positions.tolist(),decimals=4)),
            str(np.round(self.__joint_velocities.tolist(),decimals=4)),
            str(np.round(self.__joint_torques.tolist(),decimals=4)),
            )
        return out

    def __eq__(self, __js: JointState) -> bool:
        return np.equal(__js.joint_positions, self.joint_positions).all() and np.equal(__js.joint_velocities, self.joint_velocities).all() and np.equal(__js.joint_torques, self.joint_torques).all()
    
    def from_position(jpose: np.ndarray) -> JointState:
        js = JointState(jpose.shape[0])
        js.joint_positions = jpose
        return js
    
    def from_velocity(jvel: np.ndarray) -> JointState:
        js = JointState(jvel.shape[0])
        js.joint_velocities = jvel
        return js
    
    def from_torque(jtorq: np.ndarray) -> JointState:
        js = JointState(jtorq.shape[0])
        js.joint_torques = jtorq
        return js


    @property
    def num_joints(self) -> int:
        return self.__num_joints

    @property
    def joint_positions(self) -> np.ndarray:
        return self.__joint_positions
    
    @property
    def joint_velocities(self) -> np.ndarray:
        return self.__joint_velocities
    
    @property
    def joint_torques(self) -> np.ndarray:
        return self.__joint_torques
    
    
    @joint_positions.setter
    def joint_positions(self, jpose: np.ndarray):
        assert self.__num_joints == jpose.shape[0],'Invaliid input vector size, expected {:d}, but given {:d}'.format(self.__num_joints, jpose.shape[0])
        self.__joint_positions = jpose

    @joint_velocities.setter
    def joint_velocities(self, jvel: np.ndarray):
        assert self.__num_joints == jvel.shape[0],'Invaliid input vector size, expected {:d}, but given {:d}'.format(self.__num_joints, jvel.shape[0])
        self.__joint_velocities = jvel
    
    @joint_torques.setter
    def joint_torques(self, jtorq: np.ndarray):
        assert self.__num_joints == jtorq.shape[0],'Invaliid input vector size, expected {:d}, but given {:d}'.format(self.__num_joints, jtorq.shape[0])
        self.__joint_torques = jtorq


class EEState():
    def __init__(self, ee_link: str, ref_frame: str = 'world'):
        self.__ee_link = ee_link
        self.__ref_frame = ref_frame
        self.__tf = SE3()
        self.__twist = np.zeros(6)
        self.__force_torque = np.zeros(6)

    def __str__(self):
        out = 'State of {:s} in {:s}:\n Transform: {:s},\nTwist: {:s},\nForce_torque: {:s}'.format(
            self.__ee_link,
            self.__ref_frame,
            str(np.round(self.__tf.A.tolist(),decimals=4)),
            str(np.round(self.__twist.tolist(),decimals=4)),
            str(np.round(self.__force_torque.tolist(),decimals=3)),
            )
        return out

    def __eq__(self, __es: EEState) -> bool:
        return np.allclose(self.__tf.A, __es.tf.A, atol=1e-5) and np.allclose(self.__twist, __es.twist, atol=1e-5) 

    def copy(self) -> EEState:
        return 

    def from_force_torque(force_torque: np.ndarray, ee_link:str) -> EEState:
        es = EEState(ee_link)
        es.force_torque = force_torque
        return es
    
    def from_twist(twist: np.ndarray, ee_link:str) -> EEState:
        es = EEState(ee_link)
        es.twist = twist
        return es
    
    def from_tf(tf: SE3, ee_link:str)-> EEState:
        es = EEState(ee_link)
        es.tf = tf
        return es

    @property
    def ee_link(self) -> str:
        return self.__ee_link
    
    @property
    def ref_frame(self) -> str:
        return self.__ref_frame

    @property
    def tf(self) -> SE3:
        return self.__tf
    
    @property
    def twist(self) -> np.ndarray:
        return self.__twist
    
    @property
    def force_torque(self) -> np.ndarray:
        return self.__force_torque
    
    @tf.setter
    def tf(self, tf: SE3):
        assert isinstance(tf, SE3), "Unknown type for tf: {:s}".format(type(tf))
        self.__tf = tf

    @twist.setter
    def twist(self, twist: np.ndarray):
        assert twist.shape[0] == 6,'Invaliid input vector size, expected {:d}, but given {:d}'.format(6, twist.shape[0])
        self.__twist = twist
    
    @force_torque.setter
    def force_torque(self, force_torque: np.ndarray):
        assert force_torque.shape[0] == 6,'Invalid input vector size, expected {:d}, but given {:d}'.format(6, force_torque.shape[0])
        self.__force_torque = force_torque

class Motion():
    def __init__(self, ee_link: str, num_joints: int):
        self.__ee_state = EEState(ee_link)
        self.__joint_state = JointState(num_joints)
    
    def __str__(self):
        return str(self.__ee_state) + str(self.__joint_state)

    @property
    def joint_state(self) -> JointState:
        return self.__joint_state
    
    @property
    def ee_state(self) -> EEState:
        return self.__ee_state
    
    @joint_state.setter
    def joint_state(self, js: JointState):
        self.__joint_state = js
    
    @ee_state.setter
    def ee_state(self, es: EEState):
        self.__ee_state = es
    
    def from_states(joint_state: JointState, ee_state: EEState) -> Motion:
        motion = Motion(ee_state.ee_link, joint_state.num_joints)
        motion.ee_state = ee_state
        motion.joint_state = joint_state
        return motion
    
    def from_joint_state(joint_state: JointState) -> Motion:
        motion = Motion(None, joint_state.num_joints)
        motion.joint_state = joint_state
        return motion
    
    def from_ee_state(ee_state: EEState) -> Motion:
        motion = Motion(ee_state.ee_link, None)
        motion.ee_state = ee_state
        return motion

class Robot(ABC):
    def __init__(self, urdf_filename: str, base_transform: SE3 = SE3()):
        self._urdf_filename = urdf_filename
        self._base_transform = base_transform
        self._joint_state = JointState(6)
        self._force_sensor_link = None
        self.__controllers_types ={
            RobotControllerType.JOINT_POSITIONS.value: self._send_jointcontrol_position,
            RobotControllerType.JOINT_VELOCITIES.value: self._send_jointcontrol_velocity,
            RobotControllerType.JOINT_TORQUES.value: self._send_jointcontrol_torque,
            RobotControllerType.TF.value: self._send_cartcontrol_position,
            RobotControllerType.TWIST.value: self._send_cartcontrol_velocity
        }

    @abstractmethod
    def _send_jointcontrol_position(self, position: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_jointcontrol_velocity(self, velocity: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_cartcontrol_position(self, tf: SE3()) -> bool:
        pass

    @abstractmethod
    def _send_cartcontrol_velocity(self, velocity: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_jointcontrol_torque(self, torque: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _update_joint_state(self, joint_state: JointState):
        pass

    @abstractmethod
    def _update_cartesian_state(self, tool_state: EEState):
        pass

    @abstractmethod
    def jacobian(self, joint_pose: np.ndarray, ee_link: str):
        pass

    @abstractmethod
    def apply_force_sensor(self, link: str):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_joint_state(self, jstate:JointState):
        pass
    
    @property
    def joint_state(self) -> JointState:
        self._update_joint_state(self._joint_state)
        return self._joint_state
    
    @property
    def num_joints(self) -> JointState:
        return self._joint_state.num_joints

    def ee_state(self, ee_link: str, ref_frame: str = 'world') -> EEState:
        tool_state = EEState(ee_link, ref_frame)
        self._update_joint_state(self._joint_state)
        self._update_cartesian_state(tool_state)
        return tool_state

    def set_control(self, target_motion: Motion, type_ctrl: RobotControllerType) -> bool:
        assert type_ctrl.value in self.__controllers_types, "Unknown joint controller type: {:s} ".format(type_ctrl)
        target = None
        try:
            target = getattr(target_motion.joint_state, type_ctrl.value)
        except AttributeError:
            try:
                target = getattr(target_motion.ee_state, type_ctrl.value)
            except AttributeError:
                pass
        if not target is None:
            return self.__controllers_types[type_ctrl.value](target)
        return False

