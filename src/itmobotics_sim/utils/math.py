from spatialmath import SE3, SO3, Twist3
import numpy as np

def SE32vec(tf: SE3) -> np.ndarray:
    """ SE3 to vec

    Convert a 4x4 homogeneous matrix to vector representation 
    
    Args:
        tf (SE3): group SE3 from spatialmath library

    Returns:
        np.ndarray: array [x, y, z, alpha, beta, gamma]
    """
    result = SE3(SO3(tf)).twist().A
    result[:3] = tf.t
    return result

def vec2SE3(vec: np.ndarray) -> SE3:
    """ vec to SE3

    Convert vector [x, y, z, alpha, beta, gamma] to 4x4 homogeneous matrix
    belonging to the group SE(3)

    Args:
        vec (np.ndarray): (6,) np array [x, y, z, alpha, beta, gamma]

    Returns:
        SE3: group SE3 from spatialmath library
    """
    only_rot_vec = np.copy(vec)
    only_rot_vec[:3] = 0.0
    result = SE3(vec[:3]) @ Twist3(only_rot_vec).SE3()
    return result
