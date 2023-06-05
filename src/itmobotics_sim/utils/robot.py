from __future__ import annotations
import functools
import operator
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np
from spatialmath import SE3


class RobotControllerType(Enum):
    """Robot controller types

    Args:
        Enum (_type_): type of joint control
    """

    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    JOINT_TORQUES = "joint_torques"
    TF = "tf"
    TWIST = "twist"


class JointLimits:
    """Joint limits class

    Args:
        limit_positions (Tuple[np.ndarray]): limit joints positions
        limit_velocities (Tuple[np.ndarray]): limit joints velocities
        limit_torques (Tuple[np.ndarray]): limit joints torques
    """

    def __init__(
        self,
        limit_positions: Tuple[np.ndarray],
        limit_velocities: Tuple[np.ndarray],
        limit_torques: Tuple[np.ndarray],
    ):
        assert (
            limit_positions[0].shape[0] == limit_positions[1].shape[0]
        ), f"Diffrent shapes of lower an upper limits: {limit_positions[0].shape[0]:d} \
                                        and {limit_positions[1].shape[0]:d} respectively"
        self.__num_joints = limit_positions[0].shape[0]
        self.limit_positions = limit_positions

        if limit_velocities is None:
            self.limit_velocities = self.__limit_positions
        else:
            self.limit_velocities = limit_velocities

        if limit_torques is None:
            self.limit_torques = (
                1000 * np.zeros(self.__num_joints, dtype=float),
                1000 * np.zeros(self.__num_joints, dtype=float),
            )
        else:
            self.limit_torques = limit_torques

    def __str__(self):
        output_list = functools.reduce(
            operator.iconcat,
            [self.__limit_positions, self.__limit_velocities, self.__limit_torques],
            [],
        )
        out = (
            "Joint positions limits:\n   min: {:s}, max: {:s},\n"
            + "Joint velocities limits:\n   min: {:s}, max: {:s},\n"
            + "Joint torques limits: \n   min: {:s}, max: {:s}"
        ).format(*[str(v) for v in output_list])
        return out

    def clip_joint_state(self, joint_state: JointState):
        """clip joint state

        Clip positions, velocities, torques by limits

        Args:
            joint_state (JointState): joint_state to clipping
        """
        np.clip(joint_state.joint_positions, self.__limit_positions[0], self.__limit_positions[1])
        np.clip(joint_state.joint_velocities, self.__limit_velocities[0], self.__limit_velocities[1])
        np.clip(joint_state.joint_torques, self.__limit_torques[0], self.__limit_torques[1])

    @property
    def num_joints(self) -> int:
        """int: number of joints"""
        return self.__num_joints

    @property
    def limit_positions(self) -> Tuple[np.ndarray]:
        """np.ndarray: current limit_positions"""
        return self.__limit_positions

    @property
    def limit_velocities(self) -> Tuple[np.ndarray]:
        """np.ndarray: current limit_velocities"""
        return self.__limit_velocities

    @property
    def limit_torques(self) -> Tuple[np.ndarray]:
        """np.ndarray: current limit_torques"""
        return self.__limit_torques

    @limit_positions.setter
    def limit_positions(self, limit: Tuple[np.ndarray]):
        assert (
            limit[0].shape[0] == limit[1].shape[0]
        ), f"Diffrent shapes of lower an upper limits: {limit[0].shape[0]} and {limit[1].shape[0]} respectively"
        assert (
            self.__num_joints == limit[0].shape[0]
        ), f"Invalid input vector size, expected {self.__num_joints}, but given {limit[0].shape[0]}"
        self.__limit_positions = limit

    @limit_velocities.setter
    def limit_velocities(self, limit: Tuple[np.ndarray]):
        assert (
            limit[0].shape[0] == limit[1].shape[0]
        ), f"Diffrent shapes of lower an upper limits: {limit[0].shape[0]} and {limit[1].shape[0]} respectively"
        assert (
            self.__num_joints == limit[0].shape[0]
        ), f"Invalid input vector size, expected {self.__num_joints}, but given {limit[0].shape[0]}"
        self.__limit_velocities = limit

    @limit_torques.setter
    def limit_torques(self, limit: Tuple[np.ndarray]):
        assert (
            limit[0].shape[0] == limit[1].shape[0]
        ), "Diffrent shapes of lower an upper limits: {limit[0].shape[0]} and {limit[1].shape[0]} respectively"
        assert (
            self.__num_joints == limit[0].shape[0]
        ), "Invalid input vector size, expected {self.__num_joints}, but given {limit[0].shape[0]}"
        self.__limit_torques = limit


class JointState:
    """_summary_"""

    def __init__(self, num_joints: int):
        """_summary_

        Args:
            num_joints (int): _description_
        """
        self.__num_joints = num_joints
        self.__joint_positions: np.ndarray = None
        self.__joint_velocities: np.ndarray = np.zeros(num_joints)
        self.__joint_torques: np.ndarray = np.zeros(num_joints)

    def __str__(self):
        out = "Position: {:s},\nVelocity: {:s},\nTorque: {:s}".format(
            str(np.round(self.__joint_positions.tolist(), decimals=4)),
            str(np.round(self.__joint_velocities.tolist(), decimals=4)),
            str(np.round(self.__joint_torques.tolist(), decimals=4)),
        )
        return out

    def __eq__(self, __js: JointState) -> bool:
        return (
            np.equal(__js.joint_positions, self.joint_positions).all()
            and np.equal(__js.joint_velocities, self.joint_velocities).all()
            and np.equal(__js.joint_torques, self.joint_torques).all()
        )

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
        assert self.__num_joints == jpose.shape[0], "Invalid input vector size, expected {:d}, but given {:d}".format(
            self.__num_joints, jpose.shape[0]
        )
        self.__joint_positions = jpose

    @joint_velocities.setter
    def joint_velocities(self, jvel: np.ndarray):
        assert self.__num_joints == jvel.shape[0], "Invalid input vector size, expected {:d}, but given {:d}".format(
            self.__num_joints, jvel.shape[0]
        )
        self.__joint_velocities = jvel

    @joint_torques.setter
    def joint_torques(self, jtorq: np.ndarray):
        assert self.__num_joints == jtorq.shape[0], "Invalid input vector size, expected {:d}, but given {:d}".format(
            self.__num_joints, jtorq.shape[0]
        )
        self.__joint_torques = jtorq


class EEState:
    """End Effector State

    Args:
        ee_link (str): name of end effector link in urdf
        ref_frame (str, optional): name of base link in urdf. Defaults to 'global'.
    """

    def __init__(self, ee_link: str, ref_frame: str = 'global'):
        self.__ee_link = ee_link
        self.__ref_frame = ref_frame
        self.__tf = SE3()
        self.__twist = np.zeros(6)
        self.__force_torque = np.zeros(6)

    def __str__(self):
        out = f"State of {self.__ee_link} in {self.__ref_frame}: \
            \n Transform: {np.round(self.__tf.A.tolist(), decimals=4)}, \
            \n Twist: {np.round(self.__twist.tolist(), decimals=4)}, \
            \n Force_torque: {np.round(self.__force_torque.tolist(), decimals=3)}"
        return out

    def __eq__(self, __es: EEState) -> bool:
        return np.allclose(self.__tf.A, __es.tf.A, atol=1e-5) and np.allclose(self.__twist, __es.twist, atol=1e-5)

    def copy(self) -> EEState:
        return

    @staticmethod
    def from_force_torque(force_torque: np.ndarray, ee_link: str, ref_link: str = 'global') -> EEState:
        """End Effector State

        Args:
            force_torque (np.ndarray): force and torque
            ee_link (str): name of end effector link in urdf
            ref_link (str, optional): name of base link in urdf. Defaults to 'global'.

        Returns:
            EEState: End Effector State
        """
        es = EEState(ee_link, ref_link)
        es.force_torque = force_torque
        return es

    @staticmethod
    def from_twist(twist: np.ndarray, ee_link: str, ref_link: str = 'global') -> EEState:
        """_summary_

        Args:
            twist (np.ndarray): linear and angular speeds
            ee_link (str): name of end effector link in urdf
            ref_link (str, optional): name of base link in urdf. Defaults to 'global'.

        Returns:
            EEState: End Effector State
        """
        es = EEState(ee_link, ref_link)
        es.twist = twist
        return es

    @staticmethod
    def from_tf(tf: SE3, ee_link: str, ref_link: str = 'global') -> EEState:
        """_summary_

        Args:
            tf (SE3): transformation
            ee_link (str): name of end effector link in urdf
            ref_link (str, optional): name of base link in urdf. Defaults to 'global'.

        Returns:
            EEState: End Effector State
        """
        es = EEState(ee_link, ref_link)
        es.tf = tf
        return es

    def transform(self, tf: SE3):
        """_summary_

        Args:
            tf (SE3): _description_
        """
        self.tf = tf @ self.tf
        rotation_6d = np.kron(np.eye(2, dtype=int), tf.R)
        self.twist = rotation_6d @ self.twist
        self.force_torque = rotation_6d @ self.force_torque

    def inv(self):
        """inverse chain"""
        self.tf = self.tf.inv()
        self.twist = -self.twist
        # self.force_torque = self.force_torque
        ref = self.ref_frame
        self.__ref_frame = self.ee_link
        self.__ee_link = ref

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
        assert isinstance(tf, SE3), f"Unknown type for tf: {type(tf)}"
        self.__tf = tf

    @twist.setter
    def twist(self, twist: np.ndarray):
        assert twist.shape[0] == 6, f"Invalid input vector size, expected {6}, but given {twist.shape[0]}"
        self.__twist = twist

    @force_torque.setter
    def force_torque(self, force_torque: np.ndarray):
        assert force_torque.shape[0] == 6, f"Invalid input vector size, expected {6}, but given {force_torque.shape[0]}"
        self.__force_torque = force_torque


class Motion:
    """Motion

    Class description

    Args:
        ee_link (str): end link
        num_joints (int): num of joints
    """

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

    @staticmethod
    def from_states(joint_state: JointState, ee_state: EEState) -> Motion:
        """

        Args:
            joint_state (JointState): JointState
            ee_state (EEState): End Effector State

        Returns:
            Motion:
        """
        motion = Motion(ee_state.ee_link, joint_state.num_joints)
        motion.ee_state = ee_state
        motion.joint_state = joint_state
        return motion

    @staticmethod
    def from_joint_state(joint_state: JointState) -> Motion:
        """

        Args:
            joint_state (JointState): JointState

        Returns:
            Motion:
        """
        motion = Motion(None, joint_state.num_joints)
        motion.joint_state = joint_state
        return motion

    @staticmethod
    def from_ee_state(ee_state: EEState) -> Motion:
        """

        Args:
            ee_state (EEState): End Effector State

        Returns:
            Motion:
        """
        motion = Motion(ee_state.ee_link, None)
        motion.ee_state = ee_state
        return motion


class Robot(ABC):
    """Robot base class

    Args:
        urdf_filename (str): path to robot URDF
        base_transform (SE3, optional): base transform from world to robot
    """

    def __init__(self, urdf_filename: str, base_transform: SE3 = SE3()):
        self._urdf_filename = urdf_filename
        self._base_transform = base_transform
        self._joint_state: JointState = None
        self.__controllers_types = {
            RobotControllerType.JOINT_POSITIONS.value: self._send_jointcontrol_position,
            RobotControllerType.JOINT_VELOCITIES.value: self._send_jointcontrol_velocity,
            RobotControllerType.JOINT_TORQUES.value: self._send_jointcontrol_torque,
            RobotControllerType.TF.value: self._send_eecontrol_position,
            RobotControllerType.TWIST.value: self._send_eecontrol_velocity,
        }

    @abstractmethod
    def _send_jointcontrol_position(self, position: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_jointcontrol_velocity(self, velocity: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_jointcontrol_torque(self, torque: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _send_eecontrol_position(self, tf: SE3) -> bool:
        pass

    @abstractmethod
    def _send_eecontrol_velocity(self, velocity: np.ndarray) -> bool:
        pass

    @abstractmethod
    def _update_joint_state(self, joint_state: JointState):
        pass

    @abstractmethod
    def _update_ee_state(self, tool_state: EEState):
        pass

    @abstractmethod
    def jacobian(self, joint_pose: np.ndarray, ee_link: str):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_joint_state(self, jstate: JointState):
        pass

    @property
    @abstractmethod
    def joint_limits(self) -> JointLimits:
        pass

    @property
    def joint_state(self) -> JointState:
        self._update_joint_state(self._joint_state)
        return self._joint_state

    @property
    def num_joints(self) -> int:
        return self._joint_state.num_joints

    def ee_state(self, ee_link: str, ref_frame: str = 'global') -> EEState:
        """EE state of robot

        Args:
            ee_link (str): name of end effector link in urdf
            ref_link (str, optional): name of base link in urdf. Defaults to 'global'.

        Returns:
            EEState: _description_
        """
        tool_state = EEState(ee_link, ref_frame)
        self._update_joint_state(self._joint_state)
        self._update_ee_state(tool_state)
        return tool_state

    def set_control(self, target_motion: Motion, type_ctrl: RobotControllerType) -> bool:
        """

        Args:
            target_motion (Motion):
            type_ctrl (RobotControllerType):

        Returns:
            bool: return True if success
        """
        assert type_ctrl.value in self.__controllers_types, f"Unknown joint controller type: {type_ctrl}"
        target = None
        target = getattr(target_motion.joint_state, type_ctrl.value, None)
        if target is None:
            target = getattr(target_motion.ee_state, type_ctrl.value, None)

        if target is not None:
            return self.__controllers_types[type_ctrl.value](target)
        return False
