from __future__ import annotations

import numpy as np
import scipy
from abc import ABC, abstractmethod
from .robot import EEState, JointState, Robot, Motion
from spatialmath import SE3, SO3

import copy
from typing import Tuple


class VectorController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def u(self, err: np.ndarray):
        pass

    @abstractmethod
    def reset(self):
        pass

class MPIDController(VectorController):
    def __init__(self, P: np.ndarray, I: np.ndarray, D: np.ndarray, dt: float):
        super().__init__()
        self.__P = P
        self.__I = I
        self.__D = D
        self.__dt = dt
        self.__integral_value = np.zeros(self.__I.shape[0])
        self.__wind_up_max = np.ones(self.__I.shape[0])*0.1
        self.__last_err = np.zeros(self.__I.shape[0])
    
    def reset(self):
        self.__integral_value = np.zeros(self.__I.shape[0])
        self.__last_err = np.zeros(self.__I.shape[0])
    
    def u(self, err: np.ndarray):
        if err.shape[0] != self.__P.shape[0]:
            raise(RuntimeError("Invalid error shape"))
        nonlimit_integral = self.__I @ err*self.__dt + self.__integral_value
        abs_integral_value = np.minimum(self.__wind_up_max, np.abs(nonlimit_integral))
        self.__integral_value = np.multiply(abs_integral_value,  np.sign(nonlimit_integral) )
        # print(self.__integral_value)
        d_err = (err - self.__last_err)/(self.__dt+1e-3)

        u = self.__P @ err + self.__integral_value + self.__D @ d_err
        self.__last_err = err
        # print(u)
        return u
    
    @property
    def P(self):
        return self.__P
    
    @property
    def I(self):
        return self.__I

    @property
    def D(self):
        return self.__D

    @P.setter
    def P(self, P: np.ndarray):
        assert P.shape[0] == self.__P.shape[0] and P.shape[1] == self.__P.shape[1],'Invalid input matrix size, expected {:d}x{:d}, but given {:d}x{:d}'.format(self.__P.shape[0], self.__P.shape[1], P.shape[0], P.shape[1])
        self.__P = P

    @I.setter
    def I(self, I: np.ndarray):
        assert I.shape[0] == self.__I.shape[0] and I.shape[1] == self.__I.shape[1],'Invalid input matrix size, expected {:d}x{:d}, but given {:d}x{:d}'.format(self.__I.shape[0], self.__I.shape[1], I.shape[0], I.shape[1])
        self.__I = I

    @D.setter
    def D(self, D: np.ndarray):
        assert D.shape[0] == self.__D.shape[0] and D.shape[1] == self.__D.shape[1],'Invalid input matrix size, expected {:d}x{:d}, but given {:d}x{:d}'.format(self.__D.shape[0], self.__D.shape[1], D.shape[0], D.shape[1])
        self.__D = D        

class RobotController(ABC):
    def __init__(self, rob: Robot, robot_controller_type: str):
        self.robot = rob
        self.__robot_controller_type = robot_controller_type
        self.__child_controller = None

    def connect_controller(self, controller: RobotController):
        self.__child_controller = controller

    @abstractmethod
    def calc_control(self, target_motion: Motion)-> bool:
        pass

    def send_control_to_robot(self, target_motion: Motion) -> bool:
        assert isinstance(target_motion, Motion), "Invalid type of target state, expected {:s}, but given {:s}". format(str(Motion), str(type(target_motion)))
        ok = self.calc_control(target_motion)
        if not ok:
            return False
        if not self.__child_controller is None:
            return self.__child_controller.send_control_to_robot(target_motion)
        return self.robot.set_control(target_motion, self.__robot_controller_type)


class CartVelocityToJointVelocityController(RobotController):
    def __init__(self, robot: Robot):
        super().__init__(robot, 'joint_velocities')
    
    def calc_control(self, target_motion: Motion)-> bool:
        target_motion.joint_state.joint_velocities =  np.linalg.pinv(
            self.robot.jacobian(self.robot.joint_state.joint_positions,
            target_motion.ee_state.ee_link,
            target_motion.ee_state.ref_frame)
        ) @ target_motion.ee_state.twist
        return True

class JointTorquesController(RobotController):
    def __init__(self, robot: Robot):
        super().__init__(robot, 'joint_torques')
    
    def calc_control(self, target_motion: Motion)-> bool:
        return True


class CartPositionToCartVelocityController(RobotController):
    def __init__(self, robot):
        super().__init__(robot, 'twist')       
        self.__pid =  MPIDController(10*np.identity(6), 1e-4*np.identity(6), 1e-1*np.identity(6), 1e-3)

    def calc_control(self, target_motion: Motion)-> bool:
        assert isinstance(target_motion, Motion), "Invalid type of target state, expected {:s}, but given {:s}". format(str(Motion), str(type(target_motion)))
        
        current_state = self.robot.ee_state(target_motion.ee_state.ee_link)

        target_tf = target_motion.ee_state.tf
        current_tf = current_state.tf

        pose_err =  target_tf.t -  current_tf.t
        orient_error =  target_tf.R @ current_tf.R.T
        twist_err =  (SE3(*pose_err.tolist()) @ SE3(SO3(orient_error, check=False))).twist().A
        target_twist = self.__pid.u(twist_err)
        
        target_motion.ee_state.twist = target_twist
        return True

class CartForceHybrideToCartVelocityController(RobotController):
    def __init__(self, robot: Robot, selected_axis: np.ndarray, stiffnes: np.ndarray, ref_basis: str = 'world'):
        super().__init__(robot, 'twist')
        self.__pid =  MPIDController(10*np.identity(6), 1e-4*np.identity(6), 1e-1*np.identity(6), 1e-3)
        self.__ref_basis = ref_basis
        self.__stiffnes = stiffnes
        self.__T, self.__Y = CartForceHybrideToCartVelocityController.generate_square_selection_matrix(selected_axis)

    def calc_control(self, target_motion: Motion)-> bool:
        assert isinstance(target_motion, Motion), "Invalid type of target state, expected {:s}, but given {:s}". format(str(Motion), str(type(target_motion)))
        
        basis_frame = self.robot.ee_state(self.__ref_basis)
        control_basis = basis_frame.tf.R
        control_move_block = scipy.linalg.block_diag(control_basis, np.identity(3))
        control_force_block = scipy.linalg.block_diag(control_basis, control_basis)

        current_state = self.robot.ee_state(target_motion.ee_state.ee_link)

        target_tf = target_motion.ee_state.tf
        current_tf = current_state.tf

        pose_err =  target_tf.t - control_basis.T @ current_tf.t
        orient_error =  target_tf.R @ current_tf.R.T
        twist_err =  (SE3(*pose_err.tolist()) @ SE3(SO3(orient_error, check=False))).twist().A

        force_torque_err = target_motion.ee_state.force_torque - control_move_block.T @ current_state.force_torque

        target_move_twist = control_move_block @ self.__T @ self.__pid.u(twist_err)
        target_force_torque_twist = control_force_block @ self.__Y @ self.__stiffnes @ -force_torque_err
        
        target_motion.ee_state.twist = target_move_twist + target_force_torque_twist
        return True
    
    def generate_square_selection_matrix(allow_moves: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
        T_matrix = np.diag(allow_moves)
        Y_matrix = np.identity(T_matrix.shape[0]) - T_matrix
        return (T_matrix, Y_matrix)