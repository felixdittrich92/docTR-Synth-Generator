# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Turn a rendered page into a photographed capture.

A rendered page on its own is a flatbed scan: axis-aligned, edge to edge, evenly
lit. A phone capture is a *sheet of paper lying in a scene* - warped by
perspective, resting on a surface, casting a shadow, lit unevenly and often
slightly shaken. This module applies that transform to the page and carries the
word polygons through the same homography so the labels stay exact.
"""

import math
import random

import numpy as np
from PIL import Image, ImageFilter

from .config import GenerationConfig
from .legibility import DegradationBudget

__all__ = ["apply_capture", "should_capture"]

Polygon = list[list[float]]


def should_capture(config: GenerationConfig) -> bool:
    """Return whether this page should be rendered as a camera capture."""
    return random.random() < config.capture.prob


def _perspective_coeffs(src, dst) -> np.ndarray:
    """Solve the homography mapping ``dst`` -> ``src`` (what PIL's transform wants).

    ``Image.transform(..., PERSPECTIVE, coeffs)`` iterates over *output* pixels
    and samples the input, so it needs the inverse mapping. Calling this with
    the arguments swapped yields the forward mapping used for the polygons.
    """
    rows, rhs = [], []
    for (sx, sy), (dx, dy) in zip(src, dst):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        rhs += [sx, sy]
    return np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))


def _project(coeffs: np.ndarray, x: float, y: float) -> list[float]:
    """Apply a homography to a single point."""
    a, b, c, d, e, f, g, h = coeffs
    denom = g * x + h * y + 1.0
    if abs(denom) < 1e-9:
        denom = 1e-9
    return [(a * x + b * y + c) / denom, (d * x + e * y + f) / denom]


def _surface_background(size: tuple[int, int]) -> Image.Image:
    """A generated surface for the sheet to rest on.

    Deliberately generated rather than a photo: a desk photo would carry its own
    printed text into the frame as unlabelled false negatives - the same trap
    ``plain_background_prob`` guards against for page backgrounds.
    """
    width, height = size
    tone = random.randint(55, 205)
    arr = np.full((height, width, 3), float(tone), dtype=np.float32)
    # Per-channel tint so surfaces read as wood/fabric/desk rather than grey.
    arr += np.random.uniform(-18, 18, size=3)[None, None, :]
    gx = np.linspace(random.uniform(-25, 25), random.uniform(-25, 25), width)
    gy = np.linspace(random.uniform(-25, 25), random.uniform(-25, 25), height)
    arr += gx[None, :, None] + gy[:, None, None]
    arr += np.random.normal(0.0, 4.0, (height, width, 1))
    if random.random() < 0.4:  # faint directional grain (wood, brushed metal)
        period = random.uniform(6.0, 30.0)
        grain = np.sin(np.arange(height if random.random() < 0.5 else width) / period) * random.uniform(2.0, 7.0)
        arr += grain[:, None, None] if grain.shape[0] == height else grain[None, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _illuminate(config: GenerationConfig, image: Image.Image) -> Image.Image:
    """Multiplicative lighting: falloff, vignette and an optional glare blob."""
    cfg = config.capture
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    # Lighting is low-frequency by definition, so the field is built on a coarse
    # grid and upsampled - a full-resolution mgrid would cost far more for an
    # identical result.
    gh, gw = max(8, height // 8), max(8, width // 8)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    xx /= max(1, gw - 1)
    yy /= max(1, gh - 1)
    field = np.ones((gh, gw), dtype=np.float32)

    if random.random() < cfg.illumination_prob:
        # One or two soft light sources somewhere in (or just outside) the frame.
        for _ in range(random.randint(1, 2)):
            cx, cy = random.uniform(-0.2, 1.2), random.uniform(-0.2, 1.2)
            sigma = random.uniform(0.35, 0.9)
            strength = random.uniform(0.4, 1.0) * cfg.illumination_strength
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            field *= 1.0 - strength + strength * np.exp(-dist2 / (2.0 * sigma**2))

    if random.random() < cfg.vignette_prob:
        dist2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
        field *= 1.0 - cfg.vignette_strength * random.uniform(0.5, 1.0) * (dist2 / 0.5)

    # Lighting must not flatten the page: a field that darkens without bound
    # drags ink and paper together until noise and JPEG finish the job.
    field = np.clip(field, 0.55, 1.25)
    full = np.asarray(
        Image.fromarray(field, mode="F").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    )
    arr *= full[:, :, None]

    if random.random() < cfg.glare_prob:
        cx, cy = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        sx, sy = random.uniform(0.05, 0.22), random.uniform(0.05, 0.22)
        blob = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
        blob *= cfg.glare_strength * random.uniform(0.5, 1.0)
        blob_full = np.asarray(
            Image.fromarray(blob.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )[:, :, None]
        # Lift toward white instead of multiplying and clipping: multiplying
        # sends paper past 255 while ink stays put, so the highlight silently
        # erases the text it covers.
        arr = arr + blob_full * (255.0 - arr)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _motion_blur(image: Image.Image, length: int, angle: float) -> Image.Image:
    """Directional blur, as from camera shake (not the isotropic lens blur)."""
    arr = np.asarray(image, dtype=np.float32)
    dx, dy = math.cos(angle), math.sin(angle)
    acc = np.zeros_like(arr)
    half = length // 2
    offsets = range(-half, half + 1)
    for step in offsets:
        acc += np.roll(np.roll(arr, int(round(step * dy)), axis=0), int(round(step * dx)), axis=1)
    acc /= len(list(offsets))
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), mode="RGB")


def apply_capture(
    config: GenerationConfig,
    page: Image.Image,
    polygons: list[Polygon],
    budget: DegradationBudget | None = None,
) -> tuple[Image.Image, list[Polygon]]:
    """Photograph ``page``: warp it onto a surface, light it and shake the camera.

    The word polygons are pushed through the same homography, so the labels stay
    pixel-exact no matter how strong the perspective is.
    """
    cfg = config.capture
    width, height = page.size

    # 1. Where the sheet lands in the frame: centred, rotated, corners jittered.
    scale = random.uniform(*cfg.page_scale_range)
    canvas_w, canvas_h = int(width / scale), int(height / scale)
    ox, oy = (canvas_w - width) / 2.0, (canvas_h - height) / 2.0
    quad = [[ox, oy], [ox + width, oy], [ox + width, oy + height], [ox, oy + height]]

    angle = math.radians(random.uniform(*cfg.rotation_range))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    quad = [[cx + (x - cx) * cos_a - (y - cy) * sin_a, cy + (x - cx) * sin_a + (y - cy) * cos_a] for x, y in quad]
    jitter = cfg.perspective * min(width, height)
    quad = [[x + random.uniform(-jitter, jitter), y + random.uniform(-jitter, jitter)] for x, y in quad]

    # Grow the frame rather than clipping, so the sheet is always fully visible.
    pad = max(
        0.0,
        -min(p[0] for p in quad),
        -min(p[1] for p in quad),
        max(p[0] for p in quad) - canvas_w,
        max(p[1] for p in quad) - canvas_h,
    )
    if pad > 0:
        pad = math.ceil(pad) + 2
        canvas_w += 2 * pad
        canvas_h += 2 * pad
        quad = [[x + pad, y + pad] for x, y in quad]

    corners = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
    inverse = _perspective_coeffs(corners, quad)  # canvas -> page, for PIL
    forward = _perspective_coeffs(quad, corners)  # page -> canvas, for the labels

    warped = page.convert("RGBA").transform(
        (canvas_w, canvas_h),
        Image.Transform.PERSPECTIVE,
        tuple(inverse),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    # 2. Surface, drop shadow, sheet.
    scene = _surface_background((canvas_w, canvas_h)).convert("RGBA")
    if random.random() < cfg.shadow_prob:
        offset = int(min(canvas_w, canvas_h) * cfg.shadow_offset_frac * random.uniform(0.5, 1.5))
        blur = max(1.0, min(canvas_w, canvas_h) * cfg.shadow_blur_frac * random.uniform(0.6, 1.6))
        mask = warped.getchannel("A").filter(ImageFilter.GaussianBlur(blur))
        shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shadow.paste((0, 0, 0, int(150 * random.uniform(0.6, 1.0))), (offset, offset), mask)
        scene.alpha_composite(shadow)
    scene.alpha_composite(warped)
    scene = scene.convert("RGB")

    # 3. Lighting and camera shake.
    scene = _illuminate(config, scene)
    if random.random() < cfg.motion_blur_prob:
        length = random.randint(*cfg.motion_blur_length_range)
        if budget is not None:
            # A smear longer than the stroke width closes the counters; the cap
            # scales with the smallest type on the page.
            length = min(length, budget.max_motion_length)
        if length >= 2:
            scene = _motion_blur(scene, length, random.uniform(0, math.pi))

    moved = [[_project(forward, x, y) for x, y in poly] for poly in polygons]
    return scene, moved
