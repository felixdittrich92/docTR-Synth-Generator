# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Generated page designs: magazine, brochure, flyer and poster backgrounds.

Printed matter is rarely black on white: magazines set text on colour fields,
gradients, tinted sidebars, halftone screens and geometric artwork. A handful of
background photos cannot cover that range; these designs are generated, so the
variety is unlimited - and, unlike a downloaded image, they never carry text of
their own that would end up unlabelled.
"""

import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = ["design_background", "design_palette"]

_MAX_SIDE = 1600  # designs are smooth: generated at most this large, then resized


def design_palette() -> list[tuple[int, int, int]]:
    """Four to six colours of one scheme: analogous, complementary, triadic or monochrome.

    Lightness/saturation follow a style - pastel, bold, dark or muted - as a
    designer keeps one mood per spread.
    """
    hue = random.random()
    scheme = random.choice(["analogous", "complementary", "triadic", "mono", "split"])
    schemes: dict[str, list[float]] = {
        "analogous": [0.0, 0.06, -0.06, 0.12],
        "complementary": [0.0, 0.5, 0.03, 0.47],
        "triadic": [0.0, 1 / 3, 2 / 3],
        "mono": [0.0, 0.0, 0.0, 0.0],
        "split": [0.0, 0.42, 0.58],
    }
    offsets = schemes[scheme]
    style = random.choice(["pastel", "bold", "dark", "muted", "pastel"])
    sat, val = {
        "pastel": ((0.12, 0.35), (0.88, 1.0)),
        "bold": ((0.55, 0.95), (0.6, 0.95)),
        "dark": ((0.3, 0.8), (0.12, 0.35)),
        "muted": ((0.15, 0.4), (0.45, 0.8)),
    }[style]
    colours = []
    for off in offsets + [random.choice(offsets)]:
        r, g, b = colorsys.hsv_to_rgb((hue + off) % 1.0, random.uniform(*sat), random.uniform(*val))
        colours.append((int(r * 255), int(g * 255), int(b * 255)))
    if random.random() < 0.6:  # paper white / ink black accents
        colours.append(random.choice([(250, 248, 243), (245, 245, 245), (28, 28, 32)]))
    return colours


def _gradient(w: int, h: int, a, b, radial: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if radial:
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        t = np.hypot(xx - cx, yy - cy) / (max(w, h) * random.uniform(0.6, 1.2))
    else:
        ang = random.uniform(0, 2 * math.pi)
        t = (xx * math.cos(ang) + yy * math.sin(ang)) / (abs(w * math.cos(ang)) + abs(h * math.sin(ang)) + 1e-6)
        t = t - t.min()
    t = np.clip(t, 0, 1)[..., None]
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def _duotone(w: int, h: int, a, b) -> np.ndarray:
    """Marbled / watercolour wash: smooth noise mapped between two colours."""
    cell = random.randint(60, 220)  # square cells: an uneven grid smears into streaks
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    grid = Image.fromarray((np.random.random((gh, gw)) * 255).astype(np.uint8))
    wash = grid.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(cell / 4))
    t = np.asarray(wash, np.float32)[..., None] / 255.0
    t = (t - t.min()) / max(1e-6, float(t.max() - t.min()))
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def _blocks(img: Image.Image, palette) -> None:
    """Colour panels: a header band, a sidebar, a split page, boxed areas."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 4)):
        colour = random.choice(palette)
        kind = random.random()
        if kind < 0.3:  # horizontal band
            y0 = random.randint(0, h)
            draw.rectangle((0, y0, w, y0 + random.randint(h // 12, h // 3)), fill=colour)
        elif kind < 0.55:  # sidebar
            x0 = random.choice([0, w - random.randint(w // 8, w // 3)])
            draw.rectangle((x0, 0, x0 + random.randint(w // 8, w // 3), h), fill=colour)
        elif kind < 0.75:  # diagonal split
            x = random.randint(0, w)
            draw.polygon([(x, 0), (w, 0), (w, h), (x - random.randint(-w // 2, w // 2), h)], fill=colour)
        else:  # boxed area
            x0, y0 = random.randint(0, w * 2 // 3), random.randint(0, h * 2 // 3)
            draw.rectangle(
                (x0, y0, x0 + random.randint(w // 6, w // 2), y0 + random.randint(h // 8, h // 3)), fill=colour
            )


def _shapes(img: Image.Image, palette) -> None:
    """Geometric artwork: discs, rings, triangles and stripes, partly translucent."""
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(random.randint(2, 10)):
        r, g, b = random.choice(palette)
        fill = (r, g, b, random.choice([255, 255, 200, 140, 90]))
        cx, cy = random.randint(-w // 6, w + w // 6), random.randint(-h // 6, h + h // 6)
        s = random.randint(min(w, h) // 12, max(w, h) // 2)
        kind = random.random()
        if kind < 0.35:
            draw.ellipse((cx - s, cy - s, cx + s, cy + s), fill=fill)
        elif kind < 0.55:
            draw.ellipse((cx - s, cy - s, cx + s, cy + s), outline=fill, width=max(2, s // random.randint(4, 12)))
        elif kind < 0.75:
            pts = [(cx + s * math.cos(a), cy + s * math.sin(a)) for a in np.random.uniform(0, 2 * math.pi, 3)]
            draw.polygon(pts, fill=fill)
        else:  # a run of parallel stripes
            step = random.randint(12, 60)
            ang = random.uniform(0, math.pi)
            dx, dy = math.cos(ang) * max(w, h) * 2, math.sin(ang) * max(w, h) * 2
            for k in range(random.randint(3, 12)):
                ox, oy = cx + k * step * -math.sin(ang), cy + k * step * math.cos(ang)
                draw.line((ox - dx, oy - dy, ox + dx, oy + dy), fill=fill, width=max(2, step // 3))
    img.paste(layer, (0, 0), layer)


def _halftone(img: Image.Image, palette) -> None:
    """A print halftone screen whose dot size follows a gradient (fades out across a panel)."""
    w, h = img.size
    pitch = random.randint(6, 18)
    colour = random.choice(palette)
    draw = ImageDraw.Draw(img)
    ang = random.uniform(0, 2 * math.pi)
    for y in range(0, h + pitch, pitch):
        for x in range((y // pitch % 2) * pitch // 2, w + pitch, pitch):
            t = ((x * math.cos(ang) + y * math.sin(ang)) / max(w, h) + 1) / 2
            r = pitch * 0.5 * max(0.0, min(1.0, t * 1.4 - 0.2))
            if r > 0.6:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)


def _bokeh(img: Image.Image, palette) -> None:
    """Out-of-focus light blobs, the stand-in photograph of lifestyle magazines."""
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(random.randint(8, 30)):
        r, g, b = random.choice(palette)
        s = random.randint(min(w, h) // 30, min(w, h) // 6)
        x, y = random.randint(0, w), random.randint(0, h)
        draw.ellipse((x - s, y - s, x + s, y + s), fill=(r, g, b, random.randint(50, 150)))
    layer = layer.filter(ImageFilter.GaussianBlur(min(w, h) / random.uniform(60, 150)))
    img.paste(layer, (0, 0), layer)


def _pattern(img: Image.Image, palette) -> None:
    """A light repeating pattern: grid, waves, chevrons or dots (end papers, packaging)."""
    w, h = img.size
    colour = random.choice(palette)
    draw = ImageDraw.Draw(img)
    step = random.randint(18, 70)
    kind = random.random()
    if kind < 0.3:
        for x in range(0, w, step):
            draw.line((x, 0, x, h), fill=colour, width=1)
        for y in range(0, h, step):
            draw.line((0, y, w, y), fill=colour, width=1)
    elif kind < 0.6:
        amp = step / random.uniform(2, 5)
        for y0 in range(0, h + step, step):
            pts = [(x, y0 + amp * math.sin(x / step * 2 * math.pi)) for x in range(0, w + 8, 8)]
            draw.line(pts, fill=colour, width=max(1, step // 12))
    elif kind < 0.8:
        for y0 in range(0, h + step, step):
            pts = [(x, y0 + (step / 2 if (x // step) % 2 else 0)) for x in range(0, w + step, step)]
            draw.line(pts, fill=colour, width=max(1, step // 10))
    else:
        r = max(1, step // random.randint(6, 12))
        for y in range(0, h, step):
            for x in range((y // step % 2) * step // 2, w, step):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)


def design_background(size: tuple[int, int]) -> Image.Image:
    """A generated magazine/brochure/poster page design of ``size`` (RGB, no text)."""
    w, h = size
    s = min(1.0, _MAX_SIDE / max(w, h))
    sw, sh = max(8, int(w * s)), max(8, int(h * s))
    palette = design_palette()
    a, b = random.sample(palette, 2)
    base = random.random()
    if base < 0.3:
        arr = np.broadcast_to(np.asarray(a, np.float32), (sh, sw, 3))
    elif base < 0.6:
        arr = _gradient(sw, sh, a, b, radial=random.random() < 0.35)
    else:
        arr = _duotone(sw, sh, a, b)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    overlays = [_blocks, _shapes, _halftone, _bokeh, _pattern]
    for overlay in random.sample(overlays, random.choices([0, 1, 2], weights=[0.2, 0.55, 0.25])[0]):
        overlay(img, palette)
    if random.random() < 0.5:  # paper and print: a faint grain
        noise = np.random.normal(0, random.uniform(1.5, 5.0), (sh, sw, 1)).astype(np.float32)
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.8)))
    return img.resize(size, Image.Resampling.BICUBIC) if (sw, sh) != size else img
