import sys, subprocess, cv2, numpy as np
from geom import *

FPS, SECONDS, SUB = 24, 4.0, 4
N = int(FPS * SECONDS)
orig = cv2.imread('../last.png').astype(np.float32)
plate = cv2.imread('plate.png').astype(np.float32)
fg = cv2.imread('fg.png', 0).astype(np.float32)[..., None] / 255
fg_rgb = orig * fg
tex0 = cv2.imread('tex.png').astype(np.float32)
mips = [tex0]
for _ in range(4):
    mips.append(cv2.resize(mips[-1], (mips[-1].shape[1] // 2, mips[-1].shape[0] // 2), interpolation=cv2.INTER_AREA))
ones = [np.ones(m.shape[:2], np.float32) for m in mips]

def smooth(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x)
def ease(t):                      # starts from rest (the previous clip ends on a still camera) and lands softly
    return float(smooth(0, 1, t)) ** 1.15

Hf = plane_h(R1, C1, NIN, PLANE_D)
OUT = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
HFIX = homog(ph(Hf, QUAD), OUT)
D0 = np.linalg.norm(CENTER)        # camera 0 sits at the origin
dist_end = D_END

def camera(e):
    dist = D0 * (dist_end / D0) ** e                      # apparent size grows evenly in log space
    frac = (D0 - dist) / (D0 - dist_end)
    C = C1 * frac
    R = rot_slerp(np.eye(3), R1, smooth(0, 1, e ** 0.9))
    Hb = plane_h(R, C, NIN, PLANE_D)
    Hx = np.eye(3) + (HFIX / HFIX[2, 2] - np.eye(3)) * e
    return Hx @ Hb

def warp(img, Hm, border=0):
    return cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT if border == 0 else cv2.BORDER_REPLICATE)

def jac_scale(Hm, p):
    eps = 1.0
    a = ph(Hm, np.array([p, p + (eps, 0), p + (0, eps)]))
    u, v = a[1] - a[0], a[2] - a[0]
    return np.sqrt(abs(u[0] * v[1] - u[1] * v[0]))

CENTER_PX = ph(H_TEX, np.array([[TEX[0] / 2, TEX[1] / 2]]))[0]
fg_h = cv2.resize(fg, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
fg_rgb_h = cv2.resize(fg_rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
HALF = np.diag([0.5, 0.5, 1.0])
FG_PROBES = np.array([[900, 450], [1200, 900], [1480, 1150], [700, 1300]], float)

def fg_h_of(e, Hb):
    pc = ph(Hb, CENTER_PX[None])[0]
    sig = 1 + 2.2 * smooth(0.0, 1.0, e) ** 1.7
    A = np.array([[sig, 0, pc[0] * (1 - sig)], [0, sig, pc[1] * (1 - sig)], [0, 0, 1]]) @ np.array([[1, 0, -260 * e ** 1.5], [0, 1, 140 * e ** 1.5], [0, 0, 1]])
    return A @ Hb

def tex_layer(Hb):
    Ht = Hb @ H_TEX
    s = jac_scale(Ht, np.array([TEX[0] / 2, TEX[1] / 2]))
    lv = float(np.clip(np.log2(1 / max(s, 1e-6)), 0, len(mips) - 1))
    l0 = int(np.floor(lv)); l1 = min(l0 + 1, len(mips) - 1); f = lv - l0
    def at(l):
        S = np.diag([2.0 ** l, 2.0 ** l, 1]); Hl = Ht @ S
        return warp(mips[l], Hl), warp(ones[l], Hl)[..., None]
    t0, m0 = at(l0)
    if f > 1e-3 and l1 != l0:
        t1, m1 = at(l1); return t0 * (1 - f) + t1 * f, m0 * (1 - f) + m1 * f
    return t0, m0

def shutter(i):
    """eased camera positions sampled across a 0.4-frame shutter, enough samples for <=1.5 px steps"""
    ts = [float(np.clip((i + d) / (N - 1), 0, 1)) for d in (-0.2, 0.2)]
    es = [ease(t) for t in ts]
    Ha, Hz = camera(es[0]), camera(es[1])
    pts = ph(H_TEX, np.array(RECT))
    db = np.abs(ph(Ha, pts) - ph(Hz, pts)).max()
    df = np.abs(ph(fg_h_of(es[0], Ha), FG_PROBES) - ph(fg_h_of(es[1], Hz), FG_PROBES)).max() / 2
    nb = int(np.clip(np.ceil(db / 1.5), 1, 40)); nf = int(np.clip(np.ceil(df / 1.5), 1, 40))
    lin = lambda n: [float(np.clip((i + (k + 0.5) / n * 0.4 - 0.2) / (N - 1), 0, 1)) for k in range(n)] if n > 1 else [i / (N - 1)]
    return lin(nb), lin(nf)

def frame(i):
    if i == 0:
        return np.clip(orig + 0.5, 0, 255).astype(np.uint8)       # exactly the last frame of the previous clip
    if i == N - 1:
        return np.clip(tex0 + 0.5, 0, 255).astype(np.uint8)       # exactly the UI
    tb, tf = shutter(i)
    t_mid = i / (N - 1); e_mid = ease(t_mid)
    # background + screen
    acc = 0; macc = 0
    for t in tb:
        e = ease(t); Hb = camera(e)
        bg = warp(plate, Hb, 1); tx, mk = tex_layer(Hb)
        w = smooth(0.0, 0.12, t)
        if w < 1:
            org = warp(orig, Hb, 1)
            bg = org * (1 - w) + bg * w; mk = mk * w
        acc = acc + bg * (1 - mk) + tx * mk; macc = macc + mk
    base = acc / len(tb); mk = macc / len(tb)
    # depth of field: the screen plane stays sharp, the room around it softens as we close in
    dof = 7 * smooth(0.15, 0.85, e_mid)
    if dof > 0.3:
        small = cv2.resize(base, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), dof / 2), (W, H), interpolation=cv2.INTER_LINEAR)
        mk_far = cv2.resize(cv2.GaussianBlur(cv2.resize(mk, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 15), (W, H))[..., None]
        far = np.clip(1 - mk_far * 1.6, 0, 1)
        base = base * (1 - far) + blur * far
    # foreground (man + chair): nearer the camera, so it grows and leaves faster, out of focus
    fb = 0.4 + 22 * smooth(0.08, 0.7, e_mid)
    Hq = np.diag([1 / 8, 1 / 8, 1.0]) @ fg_h_of(e_mid, camera(e_mid)) @ np.diag([4, 4, 1.0])
    if cv2.warpPerspective(cv2.resize(fg_h, (W // 4, H // 4)), Hq, (W // 8, H // 8)).max() < 0.002 and e_mid > 0.5:
        tf = []                       # the foreground has left the frame
    if not tf:
        return np.clip(base + 0.5, 0, 255).astype(np.uint8)
    full = fb < 3
    fa = 0; fc = 0
    for t in tf:
        e = ease(t); Hf_ = fg_h_of(e, camera(e))
        if full:
            fa = fa + warp(fg, Hf_)[..., None]; fc = fc + warp(fg_rgb, Hf_)
        else:
            Hh = HALF @ Hf_ @ np.linalg.inv(HALF)
            fa = fa + cv2.warpPerspective(fg_h, Hh, (W // 2, H // 2))[..., None]; fc = fc + cv2.warpPerspective(fg_rgb_h, Hh, (W // 2, H // 2))
    fa = fa / len(tf); fc = fc / len(tf)
    if full:
        if fb > 0.6:
            fa = cv2.GaussianBlur(fa, (0, 0), fb)[..., None]; fc = cv2.GaussianBlur(fc, (0, 0), fb)
    else:
        fa = cv2.resize(cv2.GaussianBlur(fa, (0, 0), fb / 2), (W, H))[..., None]; fc = cv2.resize(cv2.GaussianBlur(fc, (0, 0), fb / 2), (W, H))
    out = base * (1 - fa) + fc
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        for i in map(int, sys.argv[2:]):
            cv2.imwrite(f'test_{i:03d}.png', frame(i))
    else:
        out = sys.argv[2]
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        for i in range(N):
            p.stdin.write(frame(i).tobytes())
            if i % 12 == 0: print('frame', i, flush=True)
        p.stdin.close(); p.wait()
