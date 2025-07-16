import numpy as np
from transforms3d.quaternions import mat2quat

# UR3 Denavit-Hartenberg parameters
# (alpha_i-1, a_i-1, d_i, theta_offset)
# Units: alpha [rad], a [m], d [m], theta_offset [rad]
DH_PARAMS = [
    (np.pi/2,    0.0,     0.1519, 0.0),    # Link 1
    (0.0,       -0.24365, 0.0,    0.0),    # Link 2
    (0.0,       -0.21325, 0.0,    0.0),    # Link 3
    (np.pi/2,    0.0,     0.11235,0.0),    # Link 4
    (-np.pi/2,   0.0,     0.08535,0.0),    # Link 5
    (0.0,        0.0,     0.0819, 0.0),    # Link 6 (TCP)
]

def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Compute individual DH transformation."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [   ct,   -st,    0,    a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa,  ca,  d*ca],
        [    0,     0,   0,     1],
    ])


def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """
    Compute UR3 end-effector pose from 6 joint angles.

    Args:
        joint_angles: array-like of length 6, joint angles in radians.

    Returns:
        ee_pose: np.ndarray of shape (7,) = [x, y, z, qx, qy, qz, qw]
    """
    assert len(joint_angles) == 6, "Expect 6 joint angles"
    # Base to TCP homogeneous transform
    T = np.eye(4)
    for idx, q in enumerate(joint_angles):
        alpha, a, d, offset = DH_PARAMS[idx]
        T_link = dh_transform(alpha, a, d, q + offset)
        T = T @ T_link

    # Extract position
    pos = T[:3, 3].copy()

    # Extract quaternion: mat2quat returns [w, x, y, z]
    quat_wxyz = mat2quat(T[:3, :3])
    # Reorder to [qx, qy, qz, qw]
    ee_quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    # Concatenate to 7D pose
    ee_pose = np.concatenate([pos, ee_quat])
    return ee_pose


if __name__ == '__main__':
    # Example usage
    test_joints = np.zeros(6)
    pose = forward_kinematics(test_joints)
    print("EE Pose [x,y,z,qx,qy,qz,qw]:", pose)
