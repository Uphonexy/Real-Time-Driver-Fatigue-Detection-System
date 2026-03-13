import cv2
import numpy as np
from scipy.spatial import distance as dist

# 3D model points for head pose
object_pts = np.float32([
    [6.825897,  6.760612,  4.402142],
    [1.330353,  7.122144,  6.903745],
    [-1.330353, 7.122144,  6.903745],
    [-6.825897, 6.760612,  4.402142],
    [5.311432,  5.485328,  3.987654],
    [1.789930,  5.393625,  4.413414],
    [-1.789930, 5.393625,  4.413414],
    [-5.311432, 5.485328,  3.987654],
    [2.005628,  1.409845,  6.165652],
    [-2.005628, 1.409845,  6.165652],
    [2.774015,  -2.080775, 5.048531],
    [-2.774015, -2.080775, 5.048531],
    [0.000000,  -3.116408, 6.097667],
    [0.000000,  -7.415691, 4.070434]
])

reprojectsrc = np.float32([
    [10.0, 10.0, 10.0],
    [10.0, 10.0, -10.0],
    [10.0, -10.0, -10.0],
    [10.0, -10.0, 10.0],
    [-10.0, 10.0, 10.0],
    [-10.0, 10.0, -10.0],
    [-10.0, -10.0, -10.0],
    [-10.0, -10.0, 10.0]
])

line_pairs = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

# FIX 7: Scale the camera matrix based on actual frame dimensions
def get_head_pose(shape, frame_width, frame_height):
    """Estimate head pose using PnP and return reprojected points + Euler angles."""
    fx = 653.08 * (frame_width / 640.0)
    fy = fx
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    
    K = [fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0]

    D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

    image_pts = np.float32([
        shape[17], shape[21], shape[22], shape[26],
        shape[36], shape[39], shape[42], shape[45],
        shape[31], shape[35], shape[48], shape[54],
        shape[57], shape[8]
    ])

    _, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs
    )

    reprojectdst, _ = cv2.projectPoints(
        reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs
    )
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def eye_aspect_ratio(eye):
    """Compute the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mou):
    """Compute the Mouth Aspect Ratio (MAR)."""
    X  = dist.euclidean(mou[0], mou[6])
    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])
    Y  = (Y1 + Y2) / 2.0
    return Y / X
