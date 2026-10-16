import cv2, numpy as np
from geom import *
im = cv2.imread('../last.png')
cut = cv2.imread('alpha_full.png', 0).astype(np.float32) / 255
L = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)

# monitor (screen + bezel) — the cutout is not trustworthy there
def expand(q, l, t, r, b):
    q = q.copy(); q[0] += (-l, -t); q[1] += (r, -t); q[2] += (r, b); q[3] += (-l, b); return q
mon = np.zeros((H, W), np.uint8)
cv2.fillPoly(mon, [expand(QUAD, 22, 22, 22, 62).astype(np.int32)], 255)

# shoulder edge where it crosses the screen, the chin and the desk (traced on the last frame)
edge = [(1140, 790), (1146.7, 830), (1183.3, 883.3), (1206.7, 926.7), (1216.7, 946.7), (1230, 980), (1240, 1013.3), (1250, 1046.7), (1262, 1090)]
shoulder = np.array([(900, 790)] + edge + [(900, 1090)], np.int32)
shm = np.zeros((H, W), np.uint8); cv2.fillPoly(shm, [shoulder], 255)

# chair back and the lap under the arm (the cutout leaves them out)
low = np.array([(465, 1060), (500, 1030), (530, 1020), (700, 1010), (860, 1025), (950, 1070), (1000, 1120),
                (1035, 1240), (1120, 1300), (1250, 1345), (1300, 1365), (1420, 1370), (1480, 1440), (465, 1440)], np.int32)
lowm = np.zeros((H, W), np.uint8); cv2.fillPoly(lowm, [low], 255)

fg = cut.copy()
fg[mon > 0] = 0
fg = np.maximum(fg, cv2.GaussianBlur(shm, (0, 0), 1.2).astype(np.float32) / 255 * (mon > 0))
fg = np.maximum(fg, cv2.GaussianBlur(lowm, (0, 0), 1.5).astype(np.float32) / 255)
# drop stray cutout specks away from the figure
keep = (fg > 0.5).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(keep)
big = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
reg = cv2.dilate((lab == big).astype(np.uint8), np.ones((25, 25), np.uint8))
fg *= reg
cv2.imwrite('fg.png', (fg * 255).astype(np.uint8))

# background plate: fill the figure from its surroundings
hole = cv2.dilate((fg > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8))
small = cv2.resize(im, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
scr = np.zeros((H, W), np.uint8); cv2.fillPoly(scr, [expand(QUAD, 3, 3, 3, 3).astype(np.int32)], 255)
hs = cv2.resize(np.maximum(hole, scr), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
fill = cv2.inpaint(small, hs, 20, cv2.INPAINT_TELEA)
fill = cv2.GaussianBlur(fill, (0, 0), 3)
fill = cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC)
plate = im.copy(); plate[hole > 0] = fill[hole > 0]

# rebuild the monitor chin and left bezel behind the shoulder: extend the nearest visible pixels along each edge
def extend(region, direction, step=2, limit=500):
    ys, xs = np.nonzero((region > 0) & (hole > 0))
    out_y = np.full(len(ys), -1); out_x = np.full(len(ys), -1)
    for t in range(0, limit, step):
        sy = np.clip(np.round(ys + direction[1] * t).astype(int), 0, H - 1)
        sx = np.clip(np.round(xs + direction[0] * t).astype(int), 0, W - 1)
        free = (out_y < 0) & (hole[sy, sx] == 0)
        sy2 = np.clip(np.round(ys + direction[1] * (t + 4)).astype(int), 0, H - 1)
        sx2 = np.clip(np.round(xs + direction[0] * (t + 4)).astype(int), 0, W - 1)
        out_y[free] = sy2[free]; out_x[free] = sx2[free]
    ok = out_y >= 0
    plate[ys[ok], xs[ok]] = im[out_y[ok], out_x[ok]]
chin = np.zeros((H, W), np.uint8)
cv2.fillPoly(chin, [np.array([QUAD[3] + (-14, -2), QUAD[2] + (22, -2), QUAD[2] + (22, 48), QUAD[3] + (-14, 48)], np.int32)], 255)
d = (QUAD[2] - QUAD[3]); d /= np.linalg.norm(d)
extend(chin, d)
side = np.zeros((H, W), np.uint8)
cv2.fillPoly(side, [np.array([QUAD[0] + (-14, 0), QUAD[0], QUAD[3], QUAD[3] + (-14, 0)], np.int32)], 255)
d2 = (QUAD[0] - QUAD[3]); d2 /= np.linalg.norm(d2)
extend(side, d2)
cv2.imwrite('plate.png', plate)

ov = im.copy(); m = fg > 0.5; ov[m] = (ov[m] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
cv2.imwrite('fg_ov.jpg', cv2.resize(ov, (1280, 720)))
cv2.imwrite('plate_crop.jpg', plate[600:1440, 300:1900])
