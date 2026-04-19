import numpy as np

def rpy_to_mat(r, p, y):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

def rot_z(q):
    return np.array([[np.cos(q), -np.sin(q), 0],
                     [np.sin(q), np.cos(q), 0],
                     [0, 0, 1]])

# lf leg
T0 = np.eye(4)

# joint1
T1 = np.eye(4)
T1[:3, 3] = [0.1916, 0.069987, 0.061999]
T1[:3, :3] = rpy_to_mat(0, 1.5708, 0)

# joint1 rotates around Z
q1 = 0.0
T1_rot = np.eye(4)
T1_rot[:3, :3] = rot_z(q1)

# joint2
T2 = np.eye(4)
T2[:3, 3] = [0, 0.029313, 0.060013]
T2[:3, :3] = rpy_to_mat(-1.5708, 0, 0)

# joint2 rotates around Z
q2 = 0.8
T2_rot = np.eye(4)
T2_rot[:3, :3] = rot_z(q2)

# joint3
T3 = np.eye(4)
T3[:3, 3] = [0.23001, 0, 0.0935]
T3[:3, :3] = rpy_to_mat(0, 0, -1.570796325)

# joint3 rotates around Z
q3 = -1.5
T3_rot = np.eye(4)
T3_rot[:3, :3] = rot_z(q3)

# joint4
T4 = np.eye(4)
T4[:3, 3] = [0.225, 0, 0.028]
T4[:3, :3] = rpy_to_mat(0, 0, 0)

T_final = T1 @ T1_rot @ T2 @ T2_rot @ T3 @ T3_rot @ T4
print("LF foot pos in base frame:", T_final[:3, 3])

