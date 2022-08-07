from spatialmath import SE3, SO3, Twist3
import numpy as np

def SE32vec(tf: SE3) -> np.ndarray:
    result = SE3(SO3(tf)).twist().A 
    result[:3] = tf.t
    return result

def vec2SE3(vec: np.ndarray) -> SE3:
    only_rot_vec = np.copy(vec)
    only_rot_vec[:3] = 0.0
    result = SE3(vec[:3]) @ Twist3(only_rot_vec).SE3()
    return result

