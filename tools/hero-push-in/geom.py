import numpy as np
W, H = 2560, 1440
F = 3233.5
K = np.array([[F, 0, W/2], [0, F, H/2], [0, 0, 1.]])
Ki = np.linalg.inv(K)
QUAD = np.array([[1164.5, 483], [1932.5, 464.5], [1910.3, 990], [1138.7, 942.5]])  # TL TR BR BL in the last frame
TEX = (2560, 1440)  # screen texture size (the Agents section at 1440 px, DPR 16/9)

def homog(src, dst):
    import cv2
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)

def ph(Hm, pts):
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]

RECT = np.array([[0, 0], [TEX[0], 0], [TEX[0], TEX[1]], [0, TEX[1]]], float)
H_TEX = homog(RECT, QUAD)          # texture px -> last-frame px
M = Ki @ H_TEX
k = np.linalg.norm(M[:, 0]) * TEX[0]   # screen width = 1 world unit
P = lambda u, v: (M[:, 0] * u + M[:, 1] * v + M[:, 2]) / k   # texture px -> world point (camera 0 frame)
TL, TR, BR, BL = P(0, 0), P(TEX[0], 0), P(*TEX), P(0, TEX[1])
CENTER = P(TEX[0] / 2, TEX[1] / 2)
ex = TR - TL; ex /= np.linalg.norm(ex)
ey = BL - TL; ey -= ex * (ey @ ex); ey /= np.linalg.norm(ey)
NIN = np.cross(ex, ey)                      # screen normal pointing away from the camera
R1 = np.vstack([ex, ey, NIN])               # world -> camera rotation that faces the screen
D_END = F / W                                # distance where the screen width fills the frame
C1 = CENTER - NIN * D_END
PLANE_D = NIN @ CENTER                       # n.X = d for the screen plane

def rot_slerp(Ra, Rb, s):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    r = Rot.from_matrix(np.stack([Ra, Rb]))
    return Slerp([0, 1], r)([s]).as_matrix()[0]

def plane_h(R, C, n, d):
    """last-frame px -> view px for points on plane n.X = d, camera at C with world->cam rotation R"""
    return K @ R @ (np.eye(3) - np.outer(C, n) / d) @ Ki
