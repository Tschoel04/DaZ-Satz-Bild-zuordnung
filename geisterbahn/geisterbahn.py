"""Geisterbahn-Video: Tür knattert und öffnet sich, Fahrt durch einen dunklen
Gang, 360°-Drehung nach ca. 7 s, Lichtflackern, Augen und Gespenster.

Alles wird prozedural erzeugt (numpy + ffmpeg), keine externen Medien.
Aufruf:  python3 geisterbahn.py [ausgabe.mp4]
"""
import math
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np

W_PX, H_PX, FPS = 1280, 720, 30
DUR = 28.0
ASPECT = W_PX / H_PX
FOCAL = 0.85

HALF_W, FLOOR, CEIL = 1.2, -1.0, 1.4      # Gang-Masse
DOOR_Z, DOORWAY_X, DOORWAY_Y = 2.0, 0.85, 0.95

T_RATTLE_END, T_OPEN_END = 1.35, 3.6
T_ROLL0, T_ROLL1 = 7.0, 10.5
T_DARK0, T_DARK1 = 19.5, 25.0
T_GHOST1 = (15.0, 17.2)
T_GHOST2 = (25.0, 26.6)
FLICKER_WINDOWS = [(5.0, 5.9), (12.3, 13.4), (17.6, 18.3)]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def h1(*v):
    s = 0.0
    for i, a in enumerate(v):
        s += a * (127.1 + 211.7 * i)
    return (math.sin(s) * 43758.5453) % 1.0


# ---------------------------------------------------------------- Zeitverlauf
def speed(t):
    if t < 3.0:
        return 0.0
    v = 1.6 * smooth((t - 3.0) / 1.2)
    if t > T_DARK0:
        v = 1.6 - 0.8 * smooth((t - T_DARK0) / 1.0)
    if t > T_DARK1 - 0.6:
        v *= 1 - smooth((t - (T_DARK1 - 0.6)) / 0.6)
    return v


_TS = np.arange(0, DUR + 1, 0.001)
_CZ = np.cumsum([speed(t) for t in _TS]) * 0.001


def cam_z(t):
    return float(np.interp(t, _TS, _CZ))


def roll(t):
    return 2 * math.pi * smooth((t - T_ROLL0) / (T_ROLL1 - T_ROLL0))


def door_angle(t):
    if t < T_RATTLE_END:
        return 0.03 * abs(math.sin(t * 70)) * h1(int(t * 30), 3)
    p = (t - T_RATTLE_END) / (T_OPEN_END - T_RATTLE_END)
    p = p + 0.04 * math.sin(p * 40) * (1 - p)          # ruckeliges Aufgehen
    return 1.75 * smooth(p)


def global_flicker(t):
    """1 = Licht normal, kleiner = Licht flackert/aus."""
    for a, b in FLICKER_WINDOWS:
        if a <= t <= b:
            return 1.0 if h1(int(t * 22), 7) > 0.55 else 0.04
    return 1.0


def lamp_level(i, t):
    if t > T_GHOST2[1]:
        return 0.0
    lv = 0.85 + 0.15 * math.sin(t * 31 + i * 1.7)
    if h1(i, int(t * 12)) < 0.08:
        lv = 0.15
    lv *= global_flicker(t)
    if T_DARK0 < t < T_GHOST2[1]:
        lv *= 0.0 if i != 6 else (1.0 if h1(int(t * 9), 2) > 0.7 else 0.0)
    return lv


def cart_light(t):
    if t > T_GHOST2[1]:
        return 0.0
    lv = 1.0
    if T_DARK0 < t:
        lv = 0.45 if h1(int(t * 15), 5) > 0.15 else 0.1
    return lv * max(global_flicker(t), 0.3)


LAMPS = [((0.0, 1.15, DOOR_Z - 0.6), (1.0, 0.7, 0.35), 1.3, False)]
_LCOL = [(1.0, 0.5, 0.15), (1.0, 0.5, 0.15), (0.35, 1.0, 0.3), (1.0, 0.5, 0.15), (1.0, 0.12, 0.08)]
for k in range(11):
    side = -1 if k % 2 == 0 else 1
    LAMPS.append(((side * (HALF_W - 0.08), 0.5, DOOR_Z + 2.5 + 4.5 * k), _LCOL[k % 5], 1.6, True))

EYES = [  # x, y, z, von, bis
    (-HALF_W + 0.03, 0.1, 18.0, 10.0, 14.5),
    (HALF_W - 0.03, -0.3, 24.0, 11.5, 15.5),
    (-HALF_W + 0.03, 0.6, 31.0, 19.0, 26.6),
    (HALF_W - 0.03, 0.2, 33.0, 20.0, 26.6),
    (-HALF_W + 0.03, -0.4, 35.5, 21.0, 26.6),
    (0.4, 0.9, 40.0, 21.5, 26.6),
    (HALF_W - 0.03, 0.7, 37.5, 22.5, 26.6),
]


# ---------------------------------------------------------------- Rendering
_jj, _ii = np.meshgrid(np.arange(W_PX), np.arange(H_PX))
U = ((_jj + 0.5) / W_PX * 2 - 1) * ASPECT * FOCAL
V = (1 - (_ii + 0.5) / H_PX * 2) * FOCAL
rng_grain = np.random.default_rng(1)
GRAIN = rng_grain.normal(0, 1, (8, H_PX, W_PX)).astype(np.float32)
VIGNETTE = np.clip(1.15 - 0.55 * ((_jj / W_PX - 0.5) ** 2 * 3.2 + (_ii / H_PX - 0.5) ** 2 * 3.2), 0, 1)


def fract(x):
    return x - np.floor(x)


def hash2(a, b):
    return fract(np.sin(a * 127.1 + b * 311.7) * 43758.5453)


def bricks(s, y):
    row = np.floor((y + 1) / 0.25)
    off = (row % 2) * 0.25
    col = np.floor((s + off) / 0.5)
    fs, fy = fract((s + off) / 0.5), fract((y + 1) / 0.25)
    mortar = (fs < 0.05) | (fy < 0.1)
    h = hash2(row, col)
    stain = 0.75 + 0.25 * np.sin(s * 1.3 + y * 2.1) * np.sin(s * 0.37 + 1.0)
    base = (0.55 + 0.6 * h) * stain
    c = np.stack([0.34 * base, 0.30 * base, 0.27 * base], -1)
    moss = np.clip(np.sin(s * 0.9) * np.sin(y * 3 + s) - 0.5, 0, 1)[..., None]
    c = c * (1 - moss) + moss * np.array([0.12, 0.25, 0.1])
    c[mortar] *= 0.35
    return c


def cobweb(s, y, period=7.0):
    ds = s - (np.round(s / period) * period + 1.0)
    dy = CEIL - y
    r = np.sqrt(ds * ds + dy * dy)
    ang = np.arctan2(dy, ds)
    spokes = np.abs(fract(ang / (math.pi / 7)) - 0.5) > 0.47
    rings = np.abs(fract(r / 0.09 + 0.3 * np.sin(ang * 7)) - 0.5) > 0.44
    return ((spokes | rings) & (r < 0.65)).astype(np.float32) * np.clip(1.2 - r * 1.6, 0, 1)


def render(n):
    t = n / FPS
    cz = cam_z(t)
    moving = min(speed(t) / 1.6, 1.0)
    cx = 0.05 * math.sin(t * 0.7) * moving
    cy = 0.035 * math.sin(t * 2 * math.pi * 1.4) * moving + 0.01 * (h1(n, 1) - 0.5) * moving
    rho = roll(t) + 0.02 * math.sin(t * 1.9) * moving
    psi = 0.05 * math.sin(t * 0.45) * moving
    cr, sr, cp, sp = math.cos(rho), math.sin(rho), math.cos(psi), math.sin(psi)

    dx0 = U * cr - V * sr
    dy = U * sr + V * cr
    dx = dx0 * cp + sp
    dz = -dx0 * sp + cp
    eps = 1e-6
    BIG = 1e9

    def tplane(pos, cam, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (pos - cam) / d
        return np.where((tt > 1e-4) & np.isfinite(tt), tt, BIG)

    t_r = tplane(HALF_W, cx, np.where(dx > eps, dx, np.nan))
    t_l = tplane(-HALF_W, cx, np.where(dx < -eps, dx, np.nan))
    t_f = tplane(FLOOR, cy, np.where(dy < -eps, dy, np.nan))
    t_c = tplane(CEIL, cy, np.where(dy > eps, dy, np.nan))
    tt = np.stack([t_r, t_l, t_f, t_c])
    ids = np.argmin(tt, 0)
    T = np.min(tt, 0)

    # Wand mit Türöffnung bei z = DOOR_Z
    t_dw = tplane(DOOR_Z, cz, np.where(np.abs(dz) > eps, dz, np.nan))
    hx, hy = cx + t_dw * dx, cy + t_dw * dy
    in_door = (np.abs(hx) < DOORWAY_X) & (hy < DOORWAY_Y)
    t_dw = np.where(in_door, BIG, t_dw)
    m = t_dw < T
    T = np.where(m, t_dw, T)
    ids = np.where(m, 4, ids)

    # Türflügel
    a = door_angle(t)
    panel_u = np.zeros_like(T)
    pnx = np.zeros_like(T)
    pnz = np.zeros_like(T)
    for side, hxp in ((1, -DOORWAY_X), (-1, DOORWAY_X)):
        ang = a + (0.02 * math.sin(t * 50 + side) if t < T_RATTLE_END else 0)
        Dx, Dz = side * math.cos(ang), math.sin(ang)
        det = -dx * Dz + Dx * dz
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = (-(hxp - cx) * Dz + Dx * (DOOR_Z - cz)) / det
            up = (dx * (DOOR_Z - cz) - dz * (hxp - cx)) / det
        yp = cy + tp * dy
        ok = (tp > 1e-4) & (up >= 0) & (up <= DOORWAY_X) & (yp > FLOOR) & (yp < DOORWAY_Y) & np.isfinite(tp)
        m = ok & (tp < T)
        T = np.where(m, tp, T)
        ids = np.where(m, 5, ids)
        panel_u = np.where(m, up, panel_u)
        pnx = np.where(m, -Dz, pnx)
        pnz = np.where(m, Dx, pnz)

    T = np.minimum(T, 200.0)
    px, py, pz = cx + T * dx, cy + T * dy, cz + T * dz

    # Texturen
    col = np.zeros((H_PX, W_PX, 3), np.float32)
    spec = np.zeros((H_PX, W_PX), np.float32)
    nx = np.zeros_like(T); ny = np.zeros_like(T); nz = np.zeros_like(T)

    wall = (ids == 0) | (ids == 1)
    if wall.any():
        c = bricks(pz[wall], py[wall])
        web = cobweb(pz[wall] + (ids[wall] == 1) * 3.0, py[wall])[..., None]
        col[wall] = c * (1 - web * 0.8) + web * np.array([0.75, 0.75, 0.7])
        nx[wall] = np.where(ids[wall] == 0, -1, 1)

    fl = ids == 2
    if fl.any():
        X, Z = px[fl], pz[fl]
        g = 0.5 + 0.5 * hash2(np.floor(X * 8), np.floor(Z * 8))
        c = np.stack([0.16 * g, 0.14 * g, 0.12 * g], -1)
        sleeper = (fract(Z / 0.6) < 0.35) & (np.abs(X) < 0.55)
        grain = 0.8 + 0.2 * np.sin(Z * 90 + np.sin(X * 13) * 2)
        c[sleeper] = np.stack([0.35 * grain, 0.2 * grain, 0.1 * grain], -1)[sleeper]
        rail = np.abs(np.abs(X) - 0.3) < 0.028
        c[rail] = np.array([0.55, 0.55, 0.6])
        col[fl] = c
        s = np.zeros_like(X); s[rail] = 1.0
        spec[fl] = s
        ny[fl] = 1

    ce = ids == 3
    if ce.any():
        X, Z = px[ce], pz[ce]
        beam = fract(Z / 1.5) < 0.2
        c = np.full((X.size, 3), 0.07, np.float32)
        c[beam] = np.array([0.25, 0.15, 0.08])
        web = cobweb(Z, CEIL - 0.25 * np.abs(np.abs(X) - HALF_W) * 4)[..., None]
        c = c * (1 - web * 0.7) + web * np.array([0.6, 0.6, 0.55])
        col[ce] = c
        ny[ce] = -1

    dw = ids == 4
    if dw.any():
        col[dw] = bricks(px[dw] * 1.0 + 0.3, py[dw]) * 0.9
        nz[dw] = -1

    dp = ids == 5
    if dp.any():
        u, Y = panel_u[dp], py[dp]
        plank = fract(u / 0.17)
        grain = 0.75 + 0.25 * np.sin(Y * 40 + np.sin(u * 30) * 3) * (0.5 + 0.5 * hash2(np.floor(u / 0.17), 3.0))
        c = np.stack([0.32 * grain, 0.18 * grain, 0.09 * grain], -1)
        c[(plank < 0.06)] *= 0.3
        band = (np.abs(Y + 0.55) < 0.05) | (np.abs(Y - 0.55) < 0.05)
        c[band] = np.array([0.2, 0.17, 0.15])
        rivet = band & (np.abs(fract(u / 0.17) - 0.5) < 0.12)
        c[rivet] = np.array([0.45, 0.4, 0.35])
        ring = np.abs(np.hypot(u - 0.72, Y - 0.0) - 0.07) < 0.015
        c[ring] = np.array([0.5, 0.42, 0.3])
        col[dp] = c
        nx[dp] = pnx[dp]; nz[dp] = pnz[dp]
        spec[dp] = (band | ring).astype(np.float32) * 0.5

    # Beleuchtung
    light = np.full((H_PX, W_PX, 3), 0.012, np.float32)
    beyond = pz > DOOR_Z - 0.02
    for i, (lp, lc, strength, behind) in enumerate(LAMPS):
        lv = lamp_level(i, t)
        if lv <= 0.001 or lp[2] - cz > 40 or lp[2] - cz < -12:
            continue
        lx, ly, lz = lp[0] - px, lp[1] - py, lp[2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        d = np.sqrt(d2) + 1e-6
        lam = np.abs(nx * lx + ny * ly + nz * lz) / d
        f = lv * strength * (0.25 + 0.75 * lam) / (1 + 0.9 * d2)
        if behind:
            f = f * beyond
        light += f[..., None] * np.array(lc, np.float32)
    cl = cart_light(t)
    if cl > 0:
        lx, ly, lz = cx - px, cy + 0.3 - py, cz - pz
        d2 = lx * lx + ly * ly + lz * lz
        lam = np.abs(nx * lx + ny * ly + nz * lz) / (np.sqrt(d2) + 1e-6)
        f = cl * 0.45 * (0.3 + 0.7 * lam) / (1 + 0.35 * d2)
        light += f[..., None] * np.array([0.55, 0.6, 0.8], np.float32)

    img = col * light * (1 + spec[..., None] * 1.5)
    dist = T * np.sqrt(dx * dx + dy * dy + dz * dz)
    img *= np.exp(-0.06 * dist)[..., None]

    # Projektion Welt -> Bild (für Lampen, Augen, Gespenster)
    def project(P):
        qx, qy, qz = P[0] - cx, P[1] - cy, P[2] - cz
        x1 = qx * cp - qz * sp
        z1 = qx * sp + qz * cp
        x2 = x1 * cr + qy * sr
        y2 = -x1 * sr + qy * cr
        if z1 < 0.05:
            return None
        u, v = x2 / z1, y2 / z1
        j = (u / (ASPECT * FOCAL) + 1) / 2 * W_PX
        i_ = (1 - v / FOCAL) / 2 * H_PX
        return j, i_, z1

    def blob(j, i_, rad, colr, inten, depth=None):
        r0 = int(rad * 4) + 2
        j0, j1 = max(int(j) - r0, 0), min(int(j) + r0, W_PX)
        i0, i1 = max(int(i_) - r0, 0), min(int(i_) + r0, H_PX)
        if j0 >= j1 or i0 >= i1:
            return
        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        g = np.exp(-((jj - j) ** 2 + (ii - i_) ** 2) / (rad * rad))
        if depth is not None:
            g = g * (T[i0:i1, j0:j1] > depth - 0.3)
        img[i0:i1, j0:j1] += (g * inten)[..., None] * np.array(colr, np.float32)

    for i, (lp, lc, strength, behind) in enumerate(LAMPS):
        lv = lamp_level(i, t)
        if lv <= 0.001:
            continue
        pr = project(lp)
        if pr is None or pr[2] > 45:
            continue
        j, i_, z1 = pr
        k = H_PX / (2 * FOCAL) / z1
        if behind and cz < DOOR_Z and not (0 <= int(i_) < H_PX and 0 <= int(j) < W_PX and T[int(i_), int(j)] > z1 - 0.3):
            continue
        blob(j, i_, max(0.18 * k, 2), lc, lv * 0.9, z1)
        blob(j, i_, max(0.05 * k, 1), (1, 1, 0.9), lv * 1.5, z1)

    for (ex, ey, ez, ta, tb) in EYES:
        if not (ta <= t <= tb):
            continue
        if h1(int(t * 2.5), ez) < 0.12:      # blinzeln
            continue
        fade = smooth((t - ta) / 0.8)
        inward = -0.04 if ex > 0 else 0.04
        for off in (-0.07, 0.07):
            pr = project((ex + inward, ey, ez + off) if abs(ex) > 1 else (ex + off, ey, ez))
            if pr is None:
                continue
            j, i_, z1 = pr
            k = H_PX / (2 * FOCAL) / z1
            blob(j, i_, max(0.035 * k, 1.2), (1.0, 0.1, 0.05), 2.5 * fade, z1)
            blob(j, i_, max(0.12 * k, 2), (1.0, 0.05, 0.0), 0.35 * fade, z1)

    def ghost(P, size, alpha):
        pr = project(P)
        if pr is None:
            return
        j, i_, z1 = pr
        s = size * H_PX / (2 * FOCAL) / z1
        r0 = int(s * 1.6) + 2
        j0, j1 = max(int(j) - r0, 0), min(int(j) + r0, W_PX)
        i0, i1 = max(int(i_) - r0, 0), min(int(i_) + int(s * 1.4) + 2, H_PX)
        if j0 >= j1 or i0 >= i1:
            return
        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        a_, b_ = (jj - j) / s, (ii - i_) / s
        head = np.hypot(a_, b_ + 0.25) < 0.55
        hem = 0.75 + 0.1 * np.sin(a_ * 18 + t * 12)
        body = (np.abs(a_) < 0.55 - 0.1 * b_ * (b_ < 0)) & (b_ > -0.25) & (b_ < hem)
        shape = (head | body).astype(np.float32)
        eye = (np.hypot((np.abs(a_) - 0.2) * 1.2, b_ + 0.35) < 0.11)
        mouth = np.hypot(a_ * 1.4, (b_ - 0.0) * 0.9) < 0.13
        a_map = shape * alpha
        gcol = np.where((eye | mouth)[..., None], np.array([0.02, 0.0, 0.03]),
                        np.array([0.85, 0.9, 1.0]) * (0.75 + 0.25 * np.cos(b_ * 2))[..., None])
        region = img[i0:i1, j0:j1]
        region[:] = region * (1 - a_map[..., None]) + gcol * a_map[..., None]
        glow = np.exp(-(a_ ** 2 + (b_ - 0.2) ** 2) / 0.8) * 0.25 * alpha
        region += glow[..., None] * np.array([0.5, 0.6, 1.0])

    g0, g1 = T_GHOST1
    if g0 <= t <= g1:
        p = (t - g0) / (g1 - g0)
        gx = 1.6 - 3.2 * p
        alpha = 0.8 * min(1, p * 5, (1 - p) * 5)
        ghost((gx, 0.25 + 0.15 * math.sin(p * 9), cz + 3.5), 0.55, alpha)
    g0, g1 = T_GHOST2
    if g0 <= t <= g1:
        p = (t - g0) / (g1 - g0)
        zrel = 14 * (1 - p) ** 2.2 + 0.25
        ghost((0.15 * math.sin(p * 13), 0.15, cz + zrel), 0.5, min(1, p * 4) * 0.9)

    # Nachbearbeitung
    img = 1 - np.exp(-img * 1.4)
    img = np.power(np.clip(img, 0, 1), 0.85)
    img *= VIGNETTE[..., None]
    img += GRAIN[n % 8][..., None] * 0.018
    if t > T_GHOST2[1]:
        img *= 0
    if t < 0.6:
        img *= t / 0.6
    return (np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- Audio
SR = 44100


def onepole(x, a):
    y = np.empty_like(x)
    acc = 0.0
    if np.isscalar(a):
        for i in range(len(x)):
            acc += a * (x[i] - acc)
            y[i] = acc
    else:
        for i in range(len(x)):
            acc += a[i] * (x[i] - acc)
            y[i] = acc
    return y


def fftconv(x, k):
    n = len(x) + len(k)
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(k, N), N)[: len(x)]


def env(tt, a, b, att=0.05, rel=0.2):
    return np.clip((tt - a) / att, 0, 1) * np.clip((b - tt) / rel, 0, 1)


def make_audio(path):
    rng = np.random.default_rng(7)
    n = int(DUR * SR)
    tt = np.arange(n) / SR
    L = np.zeros(n); R = np.zeros(n)

    def add(sig, start=0, pan=0.0, gain=1.0):
        s = int(start * SR)
        e = min(n, s + len(sig))
        sig = sig[: e - s] * gain
        if np.isscalar(pan):
            pl, pr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        else:
            pan = pan[: e - s]
            pl, pr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        L[s:e] += sig * pl
        R[s:e] += sig * pr

    end_mask = (tt < T_GHOST2[1]).astype(float)

    # Grundbrummen + Wind
    drone = (np.sin(2 * np.pi * 41.2 * tt) + 0.7 * np.sin(2 * np.pi * 55 * tt)
             + 0.5 * np.sin(2 * np.pi * 58.3 * tt) + 0.25 * np.sin(2 * np.pi * 87.3 * tt))
    drone *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * tt) ** 2
    wind = onepole(rng.normal(0, 1, n), 0.01) * 6
    wind *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * tt + 1)
    amb = (drone * 0.12 + wind * 0.15) * np.clip(tt / 1.0, 0, 1) * end_mask
    add(amb, 0, 0.0)
    add(onepole(rng.normal(0, 1, n), 0.012) * 5 * 0.12 * end_mask, 0, 0.6)

    # Knattern der Tür
    t0 = 0.15
    while t0 < T_RATTLE_END:
        ln = int(0.05 * SR)
        k = np.arange(ln) / SR
        f = 160 + 200 * rng.random()
        knock = np.sin(2 * np.pi * f * k) * np.exp(-k * 90) + rng.normal(0, 1, ln) * np.exp(-k * 400) * 0.6
        clank = np.sin(2 * np.pi * (2400 + 1500 * rng.random()) * k) * np.exp(-k * 160) * 0.25
        add((knock + clank) * (0.5 + 0.5 * rng.random()), t0, rng.uniform(-0.3, 0.3), 0.55)
        t0 += 0.025 + 0.04 * rng.random()

    # Knarren beim Öffnen (Stick-Slip-Impulse durch Resonator)
    ln = int((T_OPEN_END - T_RATTLE_END + 0.1) * SR)
    k = np.arange(ln) / SR
    pulses = np.zeros(ln)
    ph = 0.0
    for i in range(ln):
        rate = 35 + 90 * abs(math.sin(k[i] * 2.3)) + 30 * math.sin(k[i] * 11)
        ph += rate / SR
        if ph >= 1:
            ph -= 1
            pulses[i] = 0.6 + 0.4 * rng.random()
    kk = np.arange(int(0.03 * SR)) / SR
    res = (np.sin(2 * np.pi * 420 * kk) + 0.7 * np.sin(2 * np.pi * 870 * kk) + 0.4 * np.sin(2 * np.pi * 1530 * kk)) * np.exp(-kk * 160)
    creak = fftconv(pulses, res)
    creak *= np.clip(k / 0.15, 0, 1) * np.clip((k[-1] - k) / 0.2, 0, 1)
    add(creak / np.abs(creak).max(), T_RATTLE_END, 0.0, 0.6)

    # Tür schlägt gegen die Wand
    ln = int(1.2 * SR); k = np.arange(ln) / SR
    thud = np.sin(2 * np.pi * 62 * k) * np.exp(-k * 7) + onepole(rng.normal(0, 1, ln), 0.05) * np.exp(-k * 12) * 3
    add(thud, T_OPEN_END - 0.05, -0.4, 0.5)
    add(thud, T_OPEN_END + 0.05, 0.4, 0.5)

    # Wagen rollt: Rumpeln + Schienenstösse
    spd = np.array([speed(x) for x in tt[::441]])
    spd = np.interp(tt, tt[::441], spd) / 1.6
    rumble = onepole(rng.normal(0, 1, n), 0.004) * 12
    add(rumble * spd * 0.35, 0, 0.0)
    czs = np.interp(tt, _TS, _CZ)
    clacks = np.nonzero(np.diff(np.floor(czs / 0.6)) > 0)[0]
    ln = int(0.08 * SR); k = np.arange(ln) / SR
    clack = (np.sin(2 * np.pi * 900 * k) * 0.4 + rng.normal(0, 1, ln) * 0.6) * np.exp(-k * 120)
    for i, c in enumerate(clacks):
        g = 0.18 * spd[c]
        add(clack, c / SR, 0.0, g)
        add(clack, c / SR + 0.11, 0.0, g * 0.7)

    # Elektrisches Brummen/Knistern beim Flackern
    gate = np.array([0.0 if global_flicker(x) > 0.5 else 1.0 for x in tt[::100]])
    gate = np.repeat(gate, 100)[:n]
    flk = np.zeros(n)
    for a, b in FLICKER_WINDOWS:
        flk += env(tt, a, b, 0.02, 0.05)
    buzz = np.sign(np.sin(2 * np.pi * 100 * tt)) * 0.3 + np.sin(2 * np.pi * 200 * tt) * 0.3
    crackle = rng.normal(0, 1, n) * (rng.random(n) < 0.004) * 3
    add((buzz * (1 - gate) * 0.25 + crackle + gate * rng.normal(0, 1, n) * 0.08) * flk, 0, 0.2, 0.5)
    # Lampenknistern in der Dunkelphase
    add(crackle * env(tt, T_DARK0, T_GHOST2[1]) * 0.4, 0, -0.5, 1.0)

    # 360°-Drehung: Wusch mit Stereo-Kreisen
    m = env(tt, T_ROLL0 - 0.3, T_ROLL1 + 0.3, 0.6, 0.8)
    rolls = np.array([roll(x) for x in tt[::100]])
    rolls = np.interp(tt, tt[::100], rolls)
    rvel = np.gradient(rolls) * SR
    cutoff = np.clip(0.01 + rvel * 0.012, 0.005, 0.25)
    whoosh = onepole(rng.normal(0, 1, n) * m, cutoff) * 3
    tone = np.sin(2 * np.pi * np.cumsum(120 + 60 * rvel) / SR) * m * 0.25
    add((whoosh + tone) * 0.6, 0, np.sin(rolls), 1.0)

    # Gespenst 1: Heulen von rechts nach links
    g0, g1 = T_GHOST1
    ln = int((g1 - g0) * SR); k = np.arange(ln) / SR
    p = k / (g1 - g0)
    f = 520 - 200 * p + 25 * np.sin(2 * np.pi * 5.5 * k)
    ph = 2 * np.pi * np.cumsum(f) / SR
    moan = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * np.sin(np.pi * p) ** 1.5
    add(moan, g0, 1 - 2 * p, 0.35)

    # Herzschlag in der Dunkelphase
    t0, bpm = T_DARK0 + 0.3, 70.0
    ln = int(0.25 * SR); k = np.arange(ln) / SR
    beat = np.sin(2 * np.pi * 48 * k) * np.exp(-k * 22) + np.sin(2 * np.pi * 90 * k) * np.exp(-k * 40) * 0.4
    while t0 < T_GHOST2[1] - 0.2:
        add(beat, t0, 0.0, 0.7)
        add(beat, t0 + 0.22, 0.0, 0.45)
        bpm = min(bpm + 7, 130)
        t0 += 60 / bpm

    # Knurren aus der Dunkelheit
    ln = int(2.2 * SR); k = np.arange(ln) / SR
    growl = onepole(rng.normal(0, 1, ln), 0.03) * (0.6 + 0.4 * np.sin(2 * np.pi * 28 * k)) * np.sin(np.pi * k / k[-1]) * 6
    add(growl, 21.2, -0.6, 0.35)

    # Gespenst 2 rast heran -> Schrei -> Schlag, dann Stille
    g0, g1 = T_GHOST2
    ln = int((g1 - g0) * SR); k = np.arange(ln) / SR
    p = k / (g1 - g0)
    f = 300 + 1100 * p ** 2 + 40 * np.sin(2 * np.pi * 7 * k)
    ph = 2 * np.pi * np.cumsum(f) / SR
    shriek = (np.sin(ph) + 0.5 * np.sin(2.01 * ph) + 0.3 * np.sin(3.02 * ph)) * p ** 1.5
    shriek += rng.normal(0, 1, ln) * p ** 3 * 0.4
    add(shriek, g0, 0.0, 0.5)
    ln = int(2.0 * SR); k = np.arange(ln) / SR
    boom = (np.sin(2 * np.pi * 45 * k) * 1.2 + np.sin(2 * np.pi * 233 * k) * 0.3 + np.sin(2 * np.pi * 311 * k) * 0.3) * np.exp(-k * 2.5)
    boom += rng.normal(0, 1, ln) * np.exp(-k * 18) * 0.6
    add(boom, g1, 0.0, 0.8)

    # Hall
    ir_len = int(1.6 * SR)
    ir = rng.normal(0, 1, ir_len) * np.exp(-np.arange(ir_len) / SR * 4.0)
    ir[: int(0.02 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum())
    L2 = L + fftconv(L, ir) * 0.35
    R2 = R + fftconv(R, np.roll(ir, 331)) * 0.35
    out = np.stack([L2, R2], -1)
    out /= np.abs(out).max() / 0.9
    fade = np.clip((DUR - tt) / 0.8, 0, 1)
    out *= fade[:, None]
    data = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "geisterbahn.mp4"
    wav = out.rsplit(".", 1)[0] + "_ton.wav"
    make_audio(wav)
    frames = int(DUR * FPS)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-",
        "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
        "-af", "acompressor=threshold=-24dB:ratio=3:attack=5:release=200,loudnorm=I=-16:TP=-1.5,alimiter=limit=0.85",
        "-ar", "44100", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)
    with Pool() as pool:
        for i, fr in enumerate(pool.imap(render, range(frames), chunksize=4)):
            ff.stdin.write(fr)
            if i % 60 == 0:
                print(f"Bild {i}/{frames}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("fertig:", out)


if __name__ == "__main__":
    main()
