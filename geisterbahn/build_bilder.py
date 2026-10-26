"""Baut eine Version von geisterbahn-live.html mit eingebauten Fotos.

Die Fotos selbst gehören nicht ins (öffentliche) Repo. Sie liegen lokal in
einem Ordner mit diesen Namen (Endung egal):

    spinne.*    Spinne auf weissem Hintergrund
    vampir.*    Vampir-Gemälde im verzierten Rahmen, weisser Hintergrund
    zombie.*    Zombie freigestellt (PNG/WebP mit Transparenz)
    clown.*     Porträt, kommt in den Rahmen des Vampir-Bildes
    maedchen.*  Mädchen vor blauem Hintergrund

Aufruf:  python3 build_bilder.py ORDNER [ausgabe.html]
Ergebnis: geisterbahn-live-bilder.html (alles in einer Datei, läuft offline)
"""
import base64
import glob
import io
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_WINDOW = (63, 51, 306, 376)   # Innenfläche des Vampir-Rahmens (nach dem Zuschneiden)
ATLAS_SIZE = 2048


def find(folder, name):
    hits = glob.glob(os.path.join(folder, name + ".*"))
    if not hits:
        sys.exit(f"Bild fehlt: {name}.* in {folder}")
    return Image.open(hits[0])


def key_white(im, thr):
    """Weissen Hintergrund, der vom Rand aus erreichbar ist, durchsichtig machen."""
    a = np.asarray(im.convert("RGB")).astype(int)
    h, w, _ = a.shape
    white = np.abs(a - 255).max(-1) < thr
    m = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if white[y, x] and not m[y, x]:
                m[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if white[y, x] and not m[y, x]:
                m[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < h and 0 <= xx < w and white[yy, xx] and not m[yy, xx]:
                m[yy, xx] = True; q.append((yy, xx))
    alpha = Image.fromarray(((~m) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    out = im.convert("RGBA")
    out.putalpha(alpha)
    return out.crop(out.getbbox())


def key_girl(im):
    """Blauen Hintergrund weg, Gesicht behalten, unten weich ausblenden."""
    a = np.asarray(im.convert("RGB")).astype(float)
    h, w, _ = a.shape
    br = np.clip(a[..., 2] - a[..., 0], 0, 255).astype(np.uint8)
    br = np.asarray(Image.fromarray(br).filter(ImageFilter.GaussianBlur(2))).astype(float)
    key = 1 - np.clip((br - 92) / 22, 0, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    core = np.clip(1.6 - np.hypot((xx - w * 0.5) / (w * 0.25), (yy - h * 0.52) / (h * 0.2)), 0, 1)
    al = np.maximum(key, core)
    al *= np.clip((h * 0.97 - yy) / (h * 0.22), 0, 1)
    al *= np.clip(xx / (w * 0.12), 0, 1) * np.clip((w - xx) / (w * 0.06), 0, 1)
    al = Image.fromarray((al * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5))
    out = im.convert("RGBA")
    out.putalpha(al)
    return out


def framed(frame, pic):
    """Bild in den Vampir-Rahmen setzen (Rahmen bleibt, Innenfläche wird ersetzt)."""
    x0, y0, x1, y1 = FRAME_WINDOW
    ww, wh = x1 - x0, y1 - y0
    pic = pic.convert("RGBA")
    s = max(ww / pic.width, wh / pic.height)
    pic = pic.resize((round(pic.width * s), round(pic.height * s)), Image.LANCZOS)
    left, top = (pic.width - ww) // 2, max(0, int((pic.height - wh) * 0.2))
    pic = pic.crop((left, top, left + ww, top + wh))
    dark = Image.new("RGBA", pic.size, (12, 10, 10, 255))
    dark.alpha_composite(pic)
    out = frame.copy()
    out.paste(dark, (x0, y0))
    return out


def main():
    folder = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "geisterbahn-live-bilder.html")

    spider = key_white(find(folder, "spinne"), 35)
    vampire = key_white(find(folder, "vampir"), 30)
    z = find(folder, "zombie").convert("RGBA")
    zombie = z.crop(z.getbbox())
    clown = find(folder, "clown")
    girl = key_girl(find(folder, "maedchen"))
    zhead = zombie.crop((int(zombie.width * 0.5), 0, zombie.width, int(zombie.height * 0.3)))
    gface = girl.crop((0, int(girl.height * 0.12), girl.width, int(girl.height * 0.8)))
    portraits = [vampire, framed(vampire, clown), framed(vampire, zhead), framed(vampire, gface)]

    atlas = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))
    rects = {}

    def put(name, im, x, y, h=None, w=None):
        if h:
            im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
        elif w:
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        atlas.alpha_composite(im, (x, y))
        r = [x / ATLAS_SIZE, y / ATLAS_SIZE, im.width / ATLAS_SIZE, im.height / ATLAS_SIZE, im.width / im.height, im.height / im.width]
        if name:
            rects[name] = r
        return r, im

    _, zi = put("zombie", zombie, 4, 4, h=1000)
    _, gi = put("girl", girl, zi.width + 12, 4, h=1000)
    put("spider", spider, zi.width + gi.width + 20, 4, w=ATLAS_SIZE - zi.width - gi.width - 24)
    rects["portraits"] = []
    x = 4
    for p in portraits:
        r, pi = put(None, p, x, 1012, h=520)
        rects["portraits"].append(r[:4])
        x += pi.width + 8

    buf = io.BytesIO()
    atlas.save(buf, "WEBP", quality=88, method=6)
    data = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()

    html = open(os.path.join(HERE, "geisterbahn-live.html"), encoding="utf-8").read()
    html = html.replace("const ATLAS_SRC = null; // @@ATLAS@@", "const ATLAS_SRC = " + json.dumps(data) + ";")
    html = html.replace("const ATLAS = null; // @@RECTS@@", "const ATLAS = " + json.dumps(rects) + ";")
    open(dst, "w", encoding="utf-8").write(html)
    atlas.save(os.path.join(os.path.dirname(dst), "atlas_vorschau.png"))
    print(f"fertig: {dst} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
