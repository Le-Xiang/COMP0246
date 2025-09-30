from geometry_msgs.msg import Quaternion
import numpy as np
from numpy.typing import NDArray
import math

def rotmat2q(T: NDArray) -> Quaternion:
    """
    Converts a 3x3 rotation matrix to a ROS Quaternion.
    
    Args:
    - T: A 3x3 rotation matrix.
    
    Returns:
    - A ROS Quaternion representing the same rotation.
    """
    
    if T.shape != (3, 3):
        raise ValueError("Input rotation matrix must be 3x3.")

    # Calculate the trace
    trace = np.trace(T)
    
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (T[2, 1] - T[1, 2]) / S
        y = (T[0, 2] - T[2, 0]) / S
        z = (T[1, 0] - T[0, 1]) / S
    else:
        if T[0, 0] > T[1, 1] and T[0, 0] > T[2, 2]:
            S = math.sqrt(1.0 + T[0, 0] - T[1, 1] - T[2, 2]) * 2
            w = (T[2, 1] - T[1, 2]) / S
            x = 0.25 * S
            y = (T[0, 1] + T[1, 0]) / S
            z = (T[0, 2] + T[2, 0]) / S
        elif T[1, 1] > T[2, 2]:
            S = math.sqrt(1.0 + T[1, 1] - T[0, 0] - T[2, 2]) * 2
            w = (T[0, 2] - T[2, 0]) / S
            x = (T[0, 1] + T[1, 0]) / S
            y = 0.25 * S
            z = (T[1, 2] + T[2, 1]) / S
        else:
            S = math.sqrt(1.0 + T[2, 2] - T[0, 0] - T[1, 1]) * 2
            w = (T[1, 0] - T[0, 1]) / S
            x = (T[0, 2] + T[2, 0]) / S
            y = (T[1, 2] + T[2, 1]) / S
            z = 0.25 * S
    
    # Create the quaternion message
    q = Quaternion()
    q.x = x
    q.y = y
    q.z = z
    q.w = w
    
    return q
