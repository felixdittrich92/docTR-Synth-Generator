# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Procedural surfaces and paper damage: fabric, wood, creases, crumpled sheets.

Documents in the wild are photographed lying on tablecloths, wooden tables and
car seats, and the paper itself is folded, creased and crumpled. A detector
trained without them fires on the fine, high-contrast structure of a weave or a
crease - the dominant false positives on receipt photos such as CORD. Generated
here instead of downloaded, so the variety is unlimited and nothing in them is
text.
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = ["fabric", "wood", "crease_shading", "apply_creases", "surface", "crumpled_sheet"]


def _smooth_noise(h: int, w: int, scale: float) -> np.ndarray:
    """Low-frequency noise in [0, 1] (bilinear upsampled random grid)."""
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    grid = Image.fromarray((np.random.random((gh, gw)) * 255).astype(np.uint8))
    return np.asarray(grid.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def _palette() -> np.ndarray:
    return np.array(
        random.choice([
            (226, 214, 190), (180, 150, 110), (90, 70, 60), (60, 70, 95), (150, 40, 45), (40, 90, 70),
            (200, 200, 205), (230, 200, 160), (120, 120, 130), (35, 35, 40), (190, 170, 210), (210, 120, 60),
        ]),
        dtype=np.float32,
    )  # fmt: skip


def _weave(u: np.ndarray, v: np.ndarray, h: int, w: int) -> np.ndarray:
    """Thread structure in about [0.25, 0.75]: plain weave, twill/denim, knit or corduroy."""
    # Thread spacing as it lands in a ~1600 px frame: fine shirting to chunky knit,
    # so the weave stays visible after the frame is shrunk to a model's input.
    period = random.uniform(2.5, 12.0)
    kind = random.random()
    if kind < 0.4:  # plain weave: threads alternate over/under
        out = 0.5 + 0.25 * np.sin(2 * math.pi * u / period) * np.sign(np.sin(2 * math.pi * v / period))
    elif kind < 0.65:  # twill (denim, gabardine): the over/under steps diagonally
        out = 0.5 + 0.25 * np.sin(2 * math.pi * (u + v) / period) * (0.6 + 0.4 * np.sin(2 * math.pi * v / period))
    elif kind < 0.85:  # knit (jersey, jumper): rows of small V-shaped loops
        row = v / period
        out = 0.5 + 0.25 * np.sin(2 * math.pi * (np.abs(((u / period) % 2) - 1) + row))
    else:  # corduroy: soft parallel ribs
        rib = random.uniform(5, 14)
        out = 0.5 + 0.25 * np.abs(np.sin(math.pi * u / rib)) ** 0.5 - 0.1
    return out + 0.12 * (_smooth_noise(h, w, period * 3) - 0.5)  # slubs: uneven yarn


def _flecks(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Scattered contrasting nubs (tweed, boucle, flecked tablecloth, car-seat fabric).

    Small isolated blobs are what a detector most easily mistakes for tiny text.
    """
    count = int(h * w * random.uniform(0.0008, 0.006))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x, y in zip(np.random.randint(0, w, count), np.random.randint(0, h, count)):
        r = random.uniform(1.0, 3.5)
        draw.ellipse((x - r, y - r * random.uniform(0.6, 1.4), x + r, y + r), fill=random.randint(150, 255))
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)[..., None] / 255.0
    return img * (1 - m) + _palette()[None, None, :] * m


def fabric(size: tuple[int, int]) -> Image.Image:
    """A textile: woven, twill, knit or corduroy; plain, checked, striped or printed; draped in folds."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = random.uniform(-0.3, 0.3)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    weave = _weave(u, v, h, w)
    base = _palette()
    img = base[None, None, :] * (0.8 + 0.35 * weave[..., None])
    kind = random.random()
    if kind < 0.25:  # checks / plaid
        p = random.uniform(30, 120)
        bands = ((np.sin(2 * math.pi * u / p) > 0.4) | (np.sin(2 * math.pi * v / p) > 0.4)).astype(np.float32)
        img = img * (1 - 0.35 * bands[..., None]) + _palette()[None, None, :] * 0.35 * bands[..., None]
    elif kind < 0.42:  # stripes
        p = random.uniform(20, 90)
        bands = (np.sin(2 * math.pi * u / p) > 0.2).astype(np.float32)
        img = img * (1 - 0.4 * bands[..., None]) + _palette()[None, None, :] * 0.4 * bands[..., None]
    elif kind < 0.65:  # a printed motif: big soft blobs, like a leaf print on a tablecloth
        blobs = _smooth_noise(h, w, random.uniform(60, 180))
        mask = np.clip((blobs - random.uniform(0.55, 0.7)) * 8, 0, 1)
        img = img * (1 - mask[..., None]) + _palette()[None, None, :] * mask[..., None] * (0.8 + 0.3 * weave[..., None])
    elif kind < 0.75:  # polka dots
        p = random.uniform(18, 60)
        dots = (np.hypot((u % p) - p / 2, (v % p) - p / 2) < p * random.uniform(0.15, 0.3)).astype(np.float32)
        img = img * (1 - 0.6 * dots[..., None]) + _palette()[None, None, :] * 0.6 * dots[..., None]
    if random.random() < 0.3:
        img = _flecks(img, h, w)
    shade = 0.85 + 0.3 * _smooth_noise(h, w, max(w, h) / 3)  # the cloth sags
    if random.random() < 0.6:  # and is wrinkled: sharp folds where it was creased
        shade = shade * crease_shading(size, folds=random.randint(2, 7), crumple=random.uniform(0.0, 0.8))
    return Image.fromarray(np.clip(img * shade[..., None], 0, 255).astype(np.uint8))


def wood(size: tuple[int, int]) -> Image.Image:
    """Wood grain: growth rings warped by low-frequency noise, in a random wood tone."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = _smooth_noise(h, w, random.uniform(80, 200)) * random.uniform(20, 60)
    rings = np.sin(2 * math.pi * (yy + warp + 0.05 * xx) / random.uniform(6, 18))
    fine = _smooth_noise(h, w, 2.0)
    tone = np.array(
        random.choice([(150, 100, 60), (190, 140, 90), (110, 70, 45), (205, 175, 130), (80, 55, 40)]), np.float32
    )
    img = tone[None, None, :] * (0.82 + 0.1 * rings[..., None] + 0.12 * fine[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _blur_float(arr: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a float array (PIL blurs 8-bit images only)."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return arr
    img = Image.fromarray(((arr - lo) / (hi - lo) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0 * (hi - lo) + lo


def _facets(h: int, w: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Slopes (gx, gy) of a crumpled sheet: facets around random seeds, each tilted its own way.

    Half of each slope is the facet's own (a crisp edge where two facets meet),
    half is blurred across neighbours - paper bends between creases, it does
    not snap into perfectly flat plates.
    """
    sw, sh = _limited((w, h), 320)
    seeds = np.random.random((count, 2)) * (sh, sw)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    nearest = np.argmin((yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2, axis=-1)
    tilt = np.random.normal(0, 1, (count, 2)).astype(np.float32)
    cell = math.sqrt(sh * sw / count)
    out = []
    for k in range(2):
        hard = tilt[nearest, k]
        mixed = 0.5 * hard + 0.5 * _blur_float(hard, cell * 0.3)
        out.append(np.asarray(Image.fromarray(mixed).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32))
    return out[0], out[1]


def crease_shading(
    size: tuple[int, int], folds: int | None = None, crumple: float = 0.0, strength: float = 1.0
) -> np.ndarray:
    """Multiplicative shading of folded/crumpled paper, lit from one side.

    Each fold is a straight ridge or valley; ``crumple`` breaks the sheet into
    flat facets tilted at random plus short wrinkles. The light direction turns
    a ridge into a bright and a dark flank; ``strength`` scales the relief
    (1: a gently folded letter, 2-3: a receipt pulled out of a pocket).
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    height = np.zeros((h, w), np.float32)
    for _ in range(folds if folds is not None else random.randint(1, 4)):
        ang = random.uniform(0, math.pi)
        nx, ny = math.cos(ang), math.sin(ang)
        off = random.uniform(0.2, 0.8) * (abs(nx) * w + abs(ny) * h)
        d = xx * nx + yy * ny - off
        height += random.choice((-1, 1)) * random.uniform(4, 12) * np.exp(-np.abs(d) / random.uniform(2, 8))
    if crumple > 0:
        for _ in range(int(40 * crumple)):
            cx, cy = random.uniform(0, w), random.uniform(0, h)
            ang = random.uniform(0, math.pi)
            length = random.uniform(0.05, 0.25) * max(w, h)
            d = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
            along = np.abs(-(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang))
            height += (
                random.choice((-1, 1))
                * random.uniform(2, 7)
                * np.exp(-np.abs(d) / 3)
                * np.clip(1 - along / length, 0, 1)
            )
        height += _smooth_noise(h, w, max(w, h) / 6) * 20 * crumple
    grad_y, grad_x = np.gradient(height)
    gx: np.ndarray = grad_x  # widened below by the facet slopes
    gy: np.ndarray = grad_y
    if crumple > 0:
        fx, fy = _facets(h, w, int(random.uniform(25, 70) * crumple) + 6)
        gx, gy = gx + 0.25 * crumple * fx, gy + 0.25 * crumple * fy
    light = random.uniform(0, 2 * math.pi)
    shade = 1.0 + 0.045 * strength * (gx * math.cos(light) + gy * math.sin(light))
    return np.clip(shade, 1.0 - 0.22 * strength, 1.0 + 0.1 * strength).astype(np.float32)


def _limited(size: tuple[int, int], limit: int) -> tuple[int, int]:
    w, h = size
    s = min(1.0, limit / max(w, h))
    return max(1, int(w * s)), max(1, int(h * s))


def apply_creases(
    image: Image.Image, crumple: float = 0.0, folds: int | None = None, strength: float = 1.0
) -> Image.Image:
    """Fold or crumple ``image`` (RGB or RGBA; alpha kept). Ink and paper are shaded alike."""
    small = _limited(image.size, 1400)  # shading is smooth: compute it small, upsample
    shade = crease_shading(small, folds, crumple, strength)
    if small != image.size:
        shade = np.asarray(Image.fromarray(shade).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32)
    arr = np.asarray(image).astype(np.float32)
    arr[..., :3] *= shade[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), image.mode)


def crumpled_sheet(size: tuple[int, int]) -> Image.Image:
    """A blank sheet of crumpled paper filling the frame (white, cream, kraft or coloured)."""
    size = _limited(size, 1600)
    tint = np.array(
        random.choice([
            (245, 244, 240), (240, 236, 222), (236, 238, 242), (200, 170, 125),  # white, cream, grey, kraft
            (250, 240, 150), (215, 235, 215), (220, 232, 245), (245, 215, 220),  # yellow, green, blue, pink
        ]),
        np.float32,
    ) - random.randint(0, 15)  # fmt: skip
    w, h = size
    base = Image.fromarray(np.clip(np.broadcast_to(tint, (h, w, 3)), 0, 255).astype(np.uint8).copy())
    return apply_creases(
        base, crumple=random.uniform(0.6, 1.5), folds=random.randint(2, 6), strength=random.uniform(1.5, 2.8)
    )


def surface(size: tuple[int, int], weights: tuple[float, float, float] = (0.5, 0.35, 0.15)) -> Image.Image:
    """Something a document is photographed lying on: cloth, wood or crumpled paper (``weights``)."""
    small = _limited(size, 1600)
    make = random.choices([fabric, wood, crumpled_sheet], weights=weights)[0]
    img = make(small)
    if small != size:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 1.2)))  # slightly out of focus
