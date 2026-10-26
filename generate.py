#!/usr/bin/env python3
"""
Générateur de courses de billes (vidéos verticales 1080x1920) pour TikTok / Reels / Shorts.

Pipeline : pymunk (physique déterministe) -> Pillow (rendu) -> ffmpeg (H.264 + audio AAC)

Pour chaque jour, le script produit un dossier :
    out/AAAA-MM-JJ/video.mp4      la vidéo prête à uploader
    out/AAAA-MM-JJ/caption.txt    légende + hashtags + commentaire épinglé suggéré
    out/AAAA-MM-JJ/meta.json      seed, thème, gagnant, durée
et un fichier out/planning.csv qui résume le lot.

Exemples :
    python generate.py                         # 7 vidéos, à partir de demain
    python generate.py --count 1 --fast        # test rapide
    python generate.py --start 2026-11-02 --count 7 --series "#MaCourse"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import shutil
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymunk
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------
# Paramètres généraux
# ----------------------------------------------------------------------------
W, H, FPS = 1080, 1920, 30
SUBSTEPS = 4              # sous-pas physiques par frame (stabilité)
GRAVITY = 1800.0          # px/s^2
BALL_R = 30
N_BALLS = 8
WALL = 10
START_HOLD = 1.0          # secondes d'immobilité au départ (le hook s'affiche)
END_HOLD = 3.5            # secondes après l'arrivée du gagnant
MIN_DUR, MAX_DUR = 28.0, 48.0   # durée totale visée (s), réglable avec --min-dur / --max-dur
SECTIONS = (12, 15)       # nombre de sections du circuit (plus = course plus longue)
HOOK_SECONDS = 2.2
SR = 44100                # fréquence d'échantillonnage audio

BALL_COLORS = [
    (255, 70, 70), (70, 140, 255), (80, 220, 120), (255, 215, 50),
    (255, 110, 200), (255, 150, 50), (170, 110, 255), (60, 225, 230),
]
BALL_NAMES = {
    "fr": ["Rouge", "Bleu", "Vert", "Jaune", "Rose", "Orange", "Violet", "Cyan"],
    "en": ["Red", "Blue", "Green", "Yellow", "Pink", "Orange", "Purple", "Cyan"],
}

# Un thème par jour de la semaine : évite la répétition visuelle.
THEMES = [
    dict(name="neon",   top=(12, 8, 40),   bottom=(45, 10, 70),   track=(0, 235, 255),   peg=(255, 60, 200)),
    dict(name="ocean",  top=(5, 25, 60),   bottom=(0, 70, 100),   track=(120, 230, 255), peg=(255, 255, 255)),
    dict(name="sunset", top=(60, 15, 60),  bottom=(150, 50, 40),  track=(255, 200, 90),  peg=(255, 255, 255)),
    dict(name="forest", top=(8, 35, 25),   bottom=(20, 70, 45),   track=(190, 255, 150), peg=(255, 230, 120)),
    dict(name="candy",  top=(70, 30, 110), bottom=(30, 70, 140),  track=(255, 170, 230), peg=(255, 255, 170)),
    dict(name="mono",   top=(15, 15, 18),  bottom=(45, 45, 55),   track=(235, 235, 240), peg=(150, 150, 160)),
    dict(name="gold",   top=(25, 18, 8),   bottom=(70, 45, 10),   track=(255, 205, 80),  peg=(255, 245, 200)),
]

STR = {
    "fr": dict(title="COURSE DE BILLES", hook=("QUELLE BILLE", "VA GAGNER ?"),
               winner="VAINQUEUR", follow="Nouvelle course demain !", finish="ARRIVÉE"),
    "en": dict(title="MARBLE RACE", hook=("WHICH MARBLE", "WILL WIN?"),
               winner="WINNER", follow="New race tomorrow!", finish="FINISH"),
}

CAPTIONS = {
    "fr": [
        "Quelle bille va gagner ? Dis-moi ton pronostic 👇",
        "Team {a} ou team {b} ? Réponse à la fin 👀",
        "Course n°{ep} : tu paries sur laquelle ?",
        "Regarde jusqu'à la fin, ça se joue au dernier moment !",
        "{a} contre {b}, qui va l'emporter ? 🏁",
        "Tu l'avais vu venir ? Commente ta bille préférée 🔮",
    ],
    "en": [
        "Which marble will win? Drop your prediction below 👇",
        "Team {a} or team {b}? Watch until the end 👀",
        "Race #{ep}: which one are you betting on?",
        "Watch till the end, it comes down to the wire!",
        "{a} vs {b}, who takes it? 🏁",
        "Did you see that coming? Comment your favorite marble 🔮",
    ],
}
HASHTAGS = {
    "fr": dict(big=["#fyp", "#pourtoi", "#foryou"],
               niche=["#marblerace", "#marbles", "#satisfying", "#courseDeBilles",
                      "#oddlysatisfying", "#animation"]),
    "en": dict(big=["#fyp", "#foryou", "#viral"],
               niche=["#marblerace", "#marbles", "#satisfying", "#marblerun",
                      "#oddlysatisfying", "#animation"]),
}
PINNED = {
    "fr": "Réponse : la bille {w} a gagné ! Tu avais parié sur laquelle ?",
    "en": "Answer: {w} won! Which one did you pick?",
}
PINNED_LABEL = {"fr": "commentaire épinglé suggéré", "en": "suggested pinned comment"}


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # très vieux Pillow
        return ImageFont.load_default()


def point_seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


# ----------------------------------------------------------------------------
# 1) Circuit
# ----------------------------------------------------------------------------
def build_track(rng: random.Random) -> dict:
    """Enchaîne des sections tirées au hasard : rampes, plinko, entonnoir, hélices rotatives."""
    segments, pegs, spinners = [], [], []
    state = {"side": rng.randint(0, 1)}   # côté d'attache de la prochaine rampe
    y = 760.0

    def zigzag(y):
        k = rng.randint(2, 3)
        spacing = rng.randint(340, 400)
        gap = BALL_R * 2 * rng.uniform(2.7, 3.5)
        length = W - gap
        local = []
        for i in range(k):
            y0 = y + i * spacing
            drop = rng.uniform(150, 260)
            if state["side"] == 0:
                a, b = (0.0, y0), (length, y0 + drop)
            else:
                a, b = (float(W), y0), (W - length, y0 + drop)
            state["side"] ^= 1
            local.append((a, b))
        segments.extend(local)
        for i in range(k - 1):
            for _ in range(rng.randint(1, 3)):
                for _try in range(15):
                    r = rng.uniform(14, 24)
                    p = (rng.uniform(140, W - 140), y + i * spacing + spacing * rng.uniform(0.3, 0.75))
                    if all(point_seg_dist(p, *s) > r + BALL_R * 2.2 for s in local) and \
                       all(math.hypot(p[0] - q[0], p[1] - q[1]) > 160 for q in pegs):
                        pegs.append((p[0], p[1], r))
                        break
        return y + (k - 1) * spacing + 260 + 160

    def plinko(y):
        rows = rng.randint(5, 8)
        dy = rng.randint(115, 135)
        dx = rng.randint(140, 170)
        r = rng.uniform(12, 17)
        n = int((W - 210) // dx) + 1
        start = (W - (n - 1) * dx) / 2
        for row in range(rows):
            xs = [start + i * dx for i in range(n)] if row % 2 == 0 \
                else [start + dx / 2 + i * dx for i in range(n - 1)]
            for x in xs:
                if rng.random() > 0.12:
                    pegs.append((x, y + 80 + row * dy, r))
        return y + 80 + rows * dy + 230

    def funnel(y):
        h = rng.uniform(260, 340)
        g = BALL_R * 2 * rng.uniform(2.5, 3.2)
        segments.append(((0.0, y), (W / 2 - g / 2, y + h)))
        segments.append(((float(W), y), (W / 2 + g / 2, y + h)))
        return y + h + 330

    def spin_section(y):
        rows = rng.randint(2, 3)
        for row in range(rows):
            cy = y + 130 + row * 270
            omega = rng.choice([-1, 1]) * rng.uniform(1.6, 3.0)
            if row % 2 == 0:
                for cx in (W * 0.28, W * 0.72):
                    spinners.append((cx, cy, 160.0, omega * rng.choice([-1, 1])))
            else:
                spinners.append((W / 2, cy, 210.0, omega))
        return y + 130 + (rows - 1) * 270 + 380

    kinds = [zigzag, plinko, funnel, spin_section]
    prev = None
    for _ in range(rng.randint(*SECTIONS)):
        fn = rng.choice([k for k in kinds if k is not prev])
        y = fn(y)
        prev = fn

    finish_y = y + 200
    floor_y = finish_y + 360
    return dict(segments=segments, pegs=pegs, spinners=spinners, finish_y=finish_y,
                floor_y=floor_y, world_h=max(H, int(floor_y + 120)))


# ----------------------------------------------------------------------------
# 2) Simulation (headless, déterministe)
# ----------------------------------------------------------------------------
def simulate(seed: int):
    rng = random.Random(seed)
    tr = build_track(rng)
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.iterations = 30
    st = space.static_body

    def add_static(shape, el=0.45, fr=0.6):
        shape.elasticity, shape.friction = el, fr
        space.add(shape)

    for a, b in tr["segments"]:
        add_static(pymunk.Segment(st, a, b, 7))
    for x, y, r in tr["pegs"]:
        add_static(pymunk.Circle(st, r, offset=(x, y)), el=0.7)
    add_static(pymunk.Segment(st, (0, -1500), (0, tr["floor_y"]), WALL))
    add_static(pymunk.Segment(st, (W, -1500), (W, tr["floor_y"]), WALL))
    add_static(pymunk.Segment(st, (0, tr["floor_y"]), (W, tr["floor_y"]), 10))

    spin_bodies = []   # hélices : corps cinématiques qui tournent à vitesse constante
    for cx, cy, L, omega in tr["spinners"]:
        sb = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        sb.position = (cx, cy)
        sb.angular_velocity = omega
        shp = pymunk.Segment(sb, (-L, 0), (L, 0), 9)
        shp.elasticity, shp.friction = 0.5, 0.5
        space.add(sb, shp)
        spin_bodies.append(sb)

    bodies = []
    for k in range(N_BALLS):
        col, row = k % 4, k // 4
        body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, BALL_R))
        body.position = (W / 2 + (col - 1.5) * 100 + rng.uniform(-10, 10),
                         230 + row * 90 + rng.uniform(-5, 5))
        shape = pymunk.Circle(body, BALL_R)
        shape.elasticity = rng.uniform(0.45, 0.7)
        shape.friction = 0.5
        space.add(body, shape)
        bodies.append(body)

    dt_ = 1.0 / (FPS * SUBSTEPS)
    frames, impacts, spin_frames = [], [], []
    last_hit = [-1.0] * N_BALLS
    finish_t = [None] * N_BALLS
    prev_v = [tuple(b.velocity) for b in bodies]
    win = None  # (index de la bille, frame)
    hold_f = int(START_HOLD * FPS)
    max_f = int(MAX_DUR * FPS)

    f = 0
    while f <= max_f:
        t = f / FPS
        if f >= hold_f:
            for s in range(SUBSTEPS):
                space.step(dt_)
                ts = t + (s + 1) * dt_
                for i, b in enumerate(bodies):
                    vx, vy = b.velocity
                    px, py = prev_v[i]
                    dv = math.hypot(vx - px, vy - (py + GRAVITY * dt_))
                    prev_v[i] = (vx, vy)
                    # un choc = variation de vitesse que la gravité seule n'explique pas
                    if dv > 160 and ts - last_hit[i] > 0.06:
                        impacts.append((ts, min(1.0, dv / 1000.0), i))
                        last_hit[i] = ts
        frames.append([(b.position.x, b.position.y, b.angle) for b in bodies])
        spin_frames.append([sb.angle for sb in spin_bodies])
        for i, b in enumerate(bodies):
            if finish_t[i] is None and b.position.y > tr["finish_y"]:
                finish_t[i] = t
                if win is None:
                    win = (i, f)
        if win is not None and f - win[1] >= END_HOLD * FPS:
            break
        f += 1
    else:
        return None  # course trop longue / bloquée

    duration = len(frames) / FPS
    if not (MIN_DUR <= duration <= MAX_DUR):
        return None
    return dict(frames=np.array(frames, dtype=np.float32), impacts=impacts, win=win,
                spin=np.array(spin_frames, dtype=np.float32).reshape(len(frames), len(spin_bodies)),
                finish_t=finish_t, track=tr, duration=duration)


def make_race(seed: int):
    """Essaie plusieurs variantes d'un même seed jusqu'à obtenir une course de bonne durée."""
    for attempt in range(40):
        used = seed * 100 + attempt
        race = simulate(used)
        if race:
            race["seed_used"] = used
            return race
    raise RuntimeError(f"Aucune course valide pour le seed {seed}")


# ----------------------------------------------------------------------------
# 3) Audio synthétisé (aucun droit d'auteur)
# ----------------------------------------------------------------------------
def synth_audio(race: dict, path: Path, seed: int) -> None:
    n = int((race["duration"] + 0.3) * SR)
    out = np.zeros(n, np.float32)
    nrng = np.random.default_rng(seed)
    penta = [0, 2, 4, 7, 9, 12, 14, 16]

    def tone(freq, length, decay, amp):
        t = np.arange(int(length * SR)) / SR
        env = np.exp(-t * decay)
        atk = int(0.003 * SR)
        env[:atk] *= np.linspace(0, 1, atk)
        sig = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t)
               + 0.12 * np.sin(6 * np.pi * freq * t))
        return amp * env * sig

    def add(sig, t0):
        i0 = int(t0 * SR)
        if i0 < n:
            i1 = min(n, i0 + len(sig))
            out[i0:i1] += sig[: i1 - i0]

    for ts, strength, i in race["impacts"]:
        add(tone(261.63 * 2 ** (penta[i] / 12), 0.22, 22, 0.10 + 0.30 * strength), ts)
        click = nrng.standard_normal(int(0.012 * SR)) * np.exp(-np.arange(int(0.012 * SR)) / (0.002 * SR))
        add((0.04 + 0.10 * strength) * click, ts)

    wt = race["win"][1] / FPS
    for k, semi in enumerate([0, 4, 7, 12, 16]):
        add(tone(523.25 * 2 ** (semi / 12), 0.9, 4.5, 0.25), wt + k * 0.11)

    out = np.tanh(out * 1.3)
    out *= 0.9 / max(float(np.abs(out).max()), 1e-6)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype("<i2").tobytes())


# ----------------------------------------------------------------------------
# 4) Rendu
# ----------------------------------------------------------------------------
def render_world(tr: dict, theme: dict, seed: int, finish_label: str = "FINISH") -> Image.Image:
    """Image géante du circuit (rendue une fois, puis on 'filme' une fenêtre qui descend)."""
    WH = tr["world_h"]
    rng = random.Random(seed ^ 0xBEEF)

    top = np.array(theme["top"], np.float32)
    bot = np.array(theme["bottom"], np.float32)
    k = np.linspace(0, 1, WH, dtype=np.float32)[:, None, None]
    grad = np.broadcast_to(top * (1 - k) + bot * k, (WH, W, 3)).astype(np.uint8)
    bg = Image.fromarray(np.ascontiguousarray(grad)).convert("RGBA")

    bokeh = Image.new("RGBA", (W, WH), (0, 0, 0, 0))
    d = ImageDraw.Draw(bokeh)
    for _ in range(max(12, WH // 200)):
        r, x, y = rng.randint(60, 160), rng.randint(0, W), rng.randint(0, WH)
        d.ellipse((x - r, y - r, x + r, y + r), fill=theme["track"] + (26,))
    bg = Image.alpha_composite(bg, bokeh.filter(ImageFilter.GaussianBlur(18)))

    S = 2  # supersampling pour des bords lisses
    tc = theme["track"]
    fy = tr["finish_y"]
    BAND, PAD = 2400, 24   # rendu par bandes horizontales : évite une image géante en mémoire

    for y0 in range(0, WH, BAND):
        y1 = min(WH, y0 + BAND)
        oy = y0 - PAD                      # ordonnée du haut de la bande (marge comprise)
        bh = (y1 - y0) + 2 * PAD
        tl = Image.new("RGBA", (W * S, bh * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(tl)

        def thick_line(a, b, width, color):
            d.line([(a[0] * S, (a[1] - oy) * S), (b[0] * S, (b[1] - oy) * S)], fill=color, width=int(width * S))
            r = width * S / 2
            for p in (a, b):
                d.ellipse((p[0] * S - r, (p[1] - oy) * S - r, p[0] * S + r, (p[1] - oy) * S + r), fill=color)

        for a, b in tr["segments"]:
            thick_line(a, b, 34, tc + (55,))
        for a, b in tr["segments"]:
            thick_line(a, b, 14, tc + (255,))
        for x, y, r in tr["pegs"]:
            if oy - 40 < y < oy + bh + 40:
                yy = (y - oy)
                d.ellipse(((x - r - 8) * S, (yy - r - 8) * S, (x + r + 8) * S, (yy + r + 8) * S),
                          fill=theme["peg"] + (60,))
                d.ellipse(((x - r) * S, (yy - r) * S, (x + r) * S, (yy + r) * S), fill=theme["peg"] + (255,))
        d.rectangle((0, 0, WALL * S, bh * S), fill=tc + (255,))
        d.rectangle(((W - WALL) * S, 0, W * S, bh * S), fill=tc + (255,))
        thick_line((0, tr["floor_y"]), (W, tr["floor_y"]), 20, tc + (255,))

        if oy - 200 < fy < oy + bh + 200:
            for row in range(2):
                for col in range(W // 40 + 1):
                    c = (255, 255, 255, 255) if (row + col) % 2 == 0 else (10, 10, 10, 255)
                    d.rectangle((col * 40 * S, (fy - 20 + row * 20 - oy) * S,
                                 (col + 1) * 40 * S, (fy + row * 20 - oy) * S), fill=c)
            d.text((W * S / 2, (fy + 110 - oy) * S), finish_label, font=font(70 * S), fill=(255, 255, 255, 235),
                   anchor="mm", stroke_width=4 * S, stroke_fill=(0, 0, 0, 200))

        tl = tl.resize((W, bh), Image.LANCZOS).crop((0, PAD, W, PAD + (y1 - y0)))
        bg.paste(Image.alpha_composite(bg.crop((0, y0, W, y1)), tl), (0, y0))

    return bg.convert("RGB")


def make_sprites():
    S = 4
    size = BALL_R * 2 + 6
    big = size * S
    c, R = big / 2, BALL_R * S
    bases = []
    for col in BALL_COLORS:
        dark = tuple(int(v * 0.5) for v in col)
        im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse((c - R, c - R, c + R, c + R), fill=dark + (255,))
        d.ellipse((c - R + 3 * S, c - R + 3 * S, c + R - 3 * S, c + R - 3 * S), fill=col + (255,))
        sx, rr = c + R * 0.5, R * 0.22  # tache qui matérialise la rotation
        d.ellipse((sx - rr, c - rr, sx + rr, c + rr), fill=dark + (255,))
        bases.append(im.resize((size, size), Image.LANCZOS))
    hl = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse((c - R * 0.58, c - R * 0.62, c - R * 0.08, c - R * 0.22), fill=(255, 255, 255, 105))
    hl = hl.resize((size, size), Image.LANCZOS)
    rs = (BALL_R + 16) * 2
    ring = Image.new("RGBA", (rs * S, rs * S), (0, 0, 0, 0))
    rr = (BALL_R + 10) * S
    ImageDraw.Draw(ring).ellipse((rs * S / 2 - rr, rs * S / 2 - rr, rs * S / 2 + rr, rs * S / 2 + rr),
                                 outline=(255, 255, 255, 235), width=5 * S)
    return bases, hl, ring.resize((rs, rs), Image.LANCZOS)


def make_spinner_sprite(L: float, theme: dict) -> Image.Image:
    S = 3
    size = int(2 * L + 40)
    big = size * S
    c = big / 2
    col = theme["peg"]
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def bar(width, color):
        d.line([(c - L * S, c), (c + L * S, c)], fill=color, width=int(width * S))
        r = width * S / 2
        for sx in (c - L * S, c + L * S):
            d.ellipse((sx - r, c - r, sx + r, c + r), fill=color)

    bar(36, col + (55,))      # halo
    bar(18, col + (255,))     # barre (même épaisseur que la collision : rayon 9)
    r = 20 * S
    d.ellipse((c - r, c - r, c + r, c + r), fill=tuple(int(v * 0.45) for v in col) + (255,))
    r = 11 * S
    d.ellipse((c - r, c - r, c + r, c + r), fill=col + (255,))
    return im.resize((size, size), Image.LANCZOS)


def make_title_overlay(ep: int, strs: dict) -> Image.Image:
    ov = Image.new("RGBA", (W, 400), (0, 0, 0, 0))
    a = np.linspace(170, 0, 400).astype(np.uint8)
    shade = np.zeros((400, W, 4), np.uint8)
    shade[..., 3] = a[:, None]
    ov = Image.fromarray(shade, "RGBA")
    d = ImageDraw.Draw(ov)
    d.text((W / 2, 215), strs["title"], font=font(76), fill=(255, 255, 255, 255), anchor="mm",
           stroke_width=5, stroke_fill=(0, 0, 0, 255))
    d.text((W / 2, 292), f"#{ep}", font=font(54), fill=(255, 220, 80, 255), anchor="mm",
           stroke_width=4, stroke_fill=(0, 0, 0, 255))
    return ov


def make_hook_overlay(strs: dict) -> Image.Image:
    ov = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for y, line in zip((120, 260), strs["hook"]):
        d.text((W / 2, y), line, font=font(112), fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=9, stroke_fill=(0, 0, 0, 255))
    return ov


def make_winner_overlay(idx: int, names: list, strs: dict) -> Image.Image:
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 125))
    d = ImageDraw.Draw(ov)
    col = BALL_COLORS[idx]
    d.text((W / 2, 600), strs["winner"], font=font(112), fill=(255, 215, 60, 255), anchor="mm",
           stroke_width=8, stroke_fill=(0, 0, 0, 255))
    cx, cy, r = W / 2, 860, 125
    d.ellipse((cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8), fill=tuple(int(v * 0.5) for v in col) + (255,))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col + (255,))
    d.ellipse((cx - r * 0.6, cy - r * 0.65, cx - r * 0.1, cy - r * 0.25), fill=(255, 255, 255, 110))
    d.text((W / 2, 1090), names[idx].upper(), font=font(128), fill=col + (255,), anchor="mm",
           stroke_width=9, stroke_fill=(0, 0, 0, 255))
    d.text((W / 2, 1260), strs["follow"], font=font(48), fill=(255, 255, 255, 255), anchor="mm",
           stroke_width=4, stroke_fill=(0, 0, 0, 255))
    return ov


def scale_alpha(img: Image.Image, k: float) -> Image.Image:
    return img.getchannel("A").point(lambda v: int(v * k))


def ranking(pos, finish_t, t):
    def key(i):
        ft = finish_t[i]
        if ft is not None and ft <= t:
            return (0, ft)
        return (1, -float(pos[i, 1]))
    return sorted(range(N_BALLS), key=key)


def render_video(race, theme, ep, seed, names, strs, wav, out_mp4, fast):
    tr, frames = race["track"], race["frames"]
    WH = tr["world_h"]
    world = render_world(tr, theme, seed, strs["finish"])
    bases, hl, ring = make_sprites()
    spin_sprites = {L: make_spinner_sprite(L, theme) for _cx, _cy, L, _om in tr["spinners"]}
    title = make_title_overlay(ep, strs)
    hook = make_hook_overlay(strs)
    win_idx, win_f = race["win"]
    winner = make_winner_overlay(win_idx, names, strs)

    crng = random.Random(seed ^ 0xC0FFEE)
    confetti_cols = BALL_COLORS + [(255, 255, 255), (255, 215, 60)]
    confetti = [(crng.uniform(0, W), -crng.uniform(0, 900), crng.uniform(420, 820),
                 crng.uniform(0, 6.28), crng.choice(confetti_cols)) for _ in range(150)]

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav),
           "-c:v", "libx264", "-preset", "ultrafast" if fast else "medium",
           "-crf", "26" if fast else "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_mp4)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cam = 0.0
    total = len(frames)
    pw, rh = 330, 52
    for f in range(total):
        pos, t = frames[f], f / FPS
        ys = pos[:, 1]
        lead = int(np.argmax(ys))
        target = 0.6 * float(ys[lead]) + 0.4 * float(ys.mean()) - H * 0.45
        target = max(target, float(ys[lead]) - H * 0.72)
        target = min(max(target, 0.0), WH - H)
        cam += (target - cam) * 0.10
        cy = int(round(cam))
        fr = world.crop((0, cy, W, cy + H))

        # hélices (sous les billes)
        for k, (scx, scy, sL, _om) in enumerate(tr["spinners"]):
            ssy = scy - cy
            if -sL - 40 < ssy < H + sL + 40:
                tile = spin_sprites[sL].rotate(-math.degrees(float(race["spin"][f][k])),
                                               resample=Image.BICUBIC)
                ts_ = tile.size[0]
                fr.paste(tile, (int(scx - ts_ / 2), int(ssy - ts_ / 2)), tile)

        # billes (le leader en dernier, avec un anneau)
        def draw_ball(i):
            x, y, ang = pos[i]
            sy = y - cy
            if not (-60 < sy < H + 60):
                return
            if i == lead:
                rs = ring.size[0]
                fr.paste(ring, (int(x - rs / 2), int(sy - rs / 2)), ring)
            tile = bases[i].rotate(-math.degrees(float(ang)), resample=Image.BICUBIC)
            tile = Image.alpha_composite(tile, hl)
            s = tile.size[0]
            fr.paste(tile, (int(x - s / 2), int(sy - s / 2)), tile)

        for i in range(N_BALLS):
            if i != lead:
                draw_ball(i)
        draw_ball(lead)

        # titre
        fr.paste(title, (0, 0), title)

        # classement en direct (masqué quand l'écran gagnant prend le relais)
        if f < win_f:
            rank = ranking(pos, race["finish_t"], t)
            panel = Image.new("RGBA", (pw, 20 + rh * N_BALLS), (0, 0, 0, 0))
            pd = ImageDraw.Draw(panel)
            pd.rounded_rectangle((0, 0, pw - 1, panel.size[1] - 1), 24, fill=(0, 0, 0, 140))
            for r, i in enumerate(rank):
                y = 10 + r * rh + rh // 2
                pd.ellipse((20, y - 14, 48, y + 14), fill=BALL_COLORS[i] + (255,))
                pd.text((62, y), f"{r + 1}. {names[i]}", font=font(34), anchor="lm",
                        fill=(255, 228, 90, 255) if r == 0 else (255, 255, 255, 255))
            fr.paste(panel, (30, 330), panel)

        # hook des premières secondes
        if t < HOOK_SECONDS:
            k = 1.0 if t < HOOK_SECONDS - 0.4 else max(0.0, (HOOK_SECONDS - t) / 0.4)
            fr.paste(hook, (0, 740), scale_alpha(hook, k))

        # écran gagnant + confettis
        if f >= win_f:
            tw = (f - win_f) / FPS
            fr.paste(winner, (0, 0), scale_alpha(winner, min(1.0, tw / 0.5)))
            d = ImageDraw.Draw(fr)
            for x0, y0, vy, ph, col in confetti:
                x = x0 + 40 * math.sin(3 * tw + ph)
                y = y0 + vy * tw
                if 0 <= y < H:
                    d.rectangle((x, y, x + 14, y + 9), fill=col)

        ff.stdin.write(fr.tobytes())
        if f % 60 == 0:
            print(f"\r  rendu {f}/{total}", end="", flush=True)

    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg a échoué")
    print(f"\r  rendu {total}/{total} ✔")


# ----------------------------------------------------------------------------
# 5) Légende
# ----------------------------------------------------------------------------
def make_caption(rng, lang, ep, names, series, win_idx):
    a, b = rng.sample(names, 2)
    text = rng.choice(CAPTIONS[lang]).format(a=a, b=b, ep=ep)
    pool = HASHTAGS[lang]
    tags = [series, rng.choice(pool["big"])] + rng.sample(pool["niche"], 2)
    pinned = PINNED[lang].format(w=names[win_idx])
    return f"{text}\n\n{' '.join(tags)}\n\n--- {PINNED_LABEL[lang]} ---\n{pinned}\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global MIN_DUR, MAX_DUR
    ap = argparse.ArgumentParser(description="Génère des vidéos de courses de billes pour TikTok.")
    ap.add_argument("--count", type=int, default=7, help="nombre de vidéos (défaut 7)")
    ap.add_argument("--start", help="date de la 1re vidéo AAAA-MM-JJ (défaut : demain)")
    ap.add_argument("--out", default="out", help="dossier de sortie")
    ap.add_argument("--epoch", default="2026-10-27", help="date de l'épisode n°1")
    ap.add_argument("--series", default="#MarbleDaily", help="hashtag propre à ta chaîne")
    ap.add_argument("--lang", choices=("fr", "en"), default="en", help="langue des textes et légendes (défaut : en)")
    ap.add_argument("--min-dur", type=float, default=MIN_DUR, help="durée minimale de la vidéo en secondes")
    ap.add_argument("--max-dur", type=float, default=MAX_DUR, help="durée maximale de la vidéo en secondes")
    ap.add_argument("--fast", action="store_true", help="encodage rapide (qualité réduite, pour tester)")
    ap.add_argument("--force", action="store_true", help="régénère même si la vidéo existe")
    args = ap.parse_args()
    MIN_DUR, MAX_DUR = args.min_dur, args.max_dur

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg est introuvable. Installe-le (apt install ffmpeg / brew install ffmpeg).")

    start = dt.date.fromisoformat(args.start) if args.start else dt.date.today() + dt.timedelta(days=1)
    epoch = dt.date.fromisoformat(args.epoch)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    names = BALL_NAMES[args.lang]
    strs = STR[args.lang]
    rows = []

    for n in range(args.count):
        day = start + dt.timedelta(days=n)
        ep = (day - epoch).days + 1
        seed = int(day.strftime("%Y%m%d"))
        theme = THEMES[day.toordinal() % len(THEMES)]
        folder = out_root / day.isoformat()
        folder.mkdir(exist_ok=True)
        mp4 = folder / "video.mp4"
        print(f"[{day}] épisode #{ep} — thème {theme['name']}")

        if mp4.exists() and not args.force:
            print("  déjà généré, on passe (--force pour refaire)")
        else:
            race = make_race(seed)
            wav = folder / "audio.wav"
            synth_audio(race, wav, seed)
            render_video(race, theme, ep, race["seed_used"], names, strs, wav, mp4, args.fast)
            wav.unlink(missing_ok=True)
            win_idx = race["win"][0]
            caption = make_caption(random.Random(seed), args.lang, ep, names, args.series, win_idx)
            (folder / "caption.txt").write_text(caption, encoding="utf-8")
            (folder / "meta.json").write_text(json.dumps(dict(
                date=day.isoformat(), episode=ep, seed=seed, seed_used=race["seed_used"],
                theme=theme["name"], winner=names[win_idx], duration_s=round(race["duration"], 1)),
                indent=2, ensure_ascii=False), encoding="utf-8")

        cap_file = folder / "caption.txt"
        first_line = cap_file.read_text(encoding="utf-8").splitlines()[0] if cap_file.exists() else ""
        rows.append([day.isoformat(), str(mp4), first_line])

    with open(out_root / "planning.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", "video", "legende"])
        wr.writerows(rows)
    print(f"\nTerminé. Planning : {out_root / 'planning.csv'}")


if __name__ == "__main__":
    main()
