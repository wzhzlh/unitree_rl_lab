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

def get_foot_pos(leg_name, q1, q2, q3):
    T1 = np.eye(4)
    if leg_name == 'lf':
        T1[:3, 3] = [0.1916, 0.069987, 0.061999]
        T1[:3, :3] = rpy_to_mat(0, 1.5708, 0)
    elif leg_name == 'rf':
        T1[:3, 3] = [0.1916, -0.070013, 0.061999]
        T1[:3, :3] = rpy_to_mat(0, 1.5708, 0)
    elif leg_name == 'lb':
        T1[:3, 3] = [-0.1916, 0.070013, 0.061999]
        T1[:3, :3] = rpy_to_mat(3.1416, 1.5708, 0)
    elif leg_name == 'rb':
        T1[:3, 3] = [-0.1916, -0.07, 0.062]
        T1[:3, :3] = rpy_to_mat(3.1416, 1.5708, 0)

    T1_rot = np.eye(4)
    T1_rot[:3, :3] = rot_z(q1)

    T2 = np.eye(4)
    if leg_name == 'lf':
        T2[:3, 3] = [0, 0.029313, 0.060013]
        T2[:3, :3] = rpy_to_mat(-1.5708, 0, 0)
    elif leg_name == 'rf':
        T2[:3, 3] = [0, -0.029287, 0.059987]
        T2[:3, :3] = rpy_to_mat(1.5708, 0, 0)
    elif leg_name == 'lb':
        T2[:3, 3] = [-0.00047387, -0.029283, 0.059987]
        T2[:3, :3] = rpy_to_mat(1.5708, 0, 0)
    elif leg_name == 'rb':
        T2[:3, 3] = [-0.001301, 0.029271, 0.06]
        T2[:3, :3] = rpy_to_mat(-1.5708, 0, 0)

    T2_rot = np.eye(4)
    T2_rot[:3, :3] = rot_z(q2)

    T3 = np.eye(4)
    if leg_name == 'lf':
        T3[:3, 3] = [0.23001, 0, 0.0935]
        T3[:3, :3] = rpy_to_mat(0, 0, -1.570796325)
    elif leg_name == 'rf':
        T3[:3, 3] = [0.22999, 1.0192e-05, 0.0935]
        T3[:3, :3] = rpy_to_mat(0, 0, 1.570796325)
    elif leg_name == 'lb':
        T3[:3, 3] = [0.23001, 1.0811e-05, 0.0935]
        T3[:3, :3] = rpy_to_mat(0, 0, -1.570796325)
    elif leg_name == 'rb':
        T3[:3, 3] = [0.23, 0, 0.0935]
        T3[:3, :3] = rpy_to_mat(0, 0, 1.570796325)

    T3_rot = np.eye(4)
    T3_rot[:3, :3] = rot_z(q3)

    T4 = np.eye(4)
    T4[:3, 3] = [0.225, 0, 0.028]
    T4[:3, :3] = rpy_to_mat(0, 0, 0)

    T_final = T1 @ T1_rot @ T2 @ T2_rot @ T3 @ T3_rot @ T4
    return T_final[:3, 3]

# Grid search for good posture
q2_range = np.linspace(-1.5, 1.5, 20)
q3_range = np.linspace(-2.5, 2.5, 20)

legs = ['lf', 'rf', 'lb', 'rb']
for leg in legs:
    print(f"--- {leg} ---")
    best_dist = 1000
    best_q = None
    best_pos = None
    
    target_z = -0.3
    if leg in ['lf', 'rf']:
        target_x = 0.2
    else:
        target_x = -0.2
        
    for q2 in q2_range:
        for q3 in q3_range:
            pos = get_foot_pos(leg, 0, q2, q3)
            # We want foot_x to be roughly target_x, and foot_z to be roughly target_z
            dist = (pos[0] - target_x)**2 + (pos[2] - target_z)**2
            if dist < best_dist:
                best_dist = dist
                best_q = (q2, q3)
                best_pos = pos
                
    print(f"Target: X={target_x}, Z={target_z}")
    print(f"Best q2={best_q[0]:.3f}, q3={best_q[1]:.3f} => pos X={best_pos[0]:.3f}, Y={best_pos[1]:.3f}, Z={best_pos[2]:.3f}")

