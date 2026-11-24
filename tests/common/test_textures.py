import random

import numpy as np
import pytest

from generator.components.textures import apply_creases, crumpled_sheet, fabric, surface, wood


@pytest.mark.parametrize("seed", range(6))
def test_textures_render_at_the_requested_size(seed):
    random.seed(seed)
    np.random.seed(seed)
    for make in (fabric, wood, crumpled_sheet):
        img = make((160, 120))
        assert img.size == (160, 120) and img.mode == "RGB"
        assert np.asarray(img, dtype=np.float32).std() > 1  # structured, not flat


def test_surface_weights_pick_the_kind():
    random.seed(0)
    np.random.seed(0)
    # Only crumpled paper: light and nearly neutral.
    arr = np.asarray(surface((200, 150), weights=(0, 0, 1)), dtype=np.float32)
    assert arr.mean() > 120


def test_creases_shade_but_keep_alpha():
    from PIL import Image

    page = Image.new("RGBA", (300, 200), (240, 240, 240, 128))
    out = apply_creases(page, crumple=1.0, folds=3)
    arr = np.asarray(out)
    assert out.mode == "RGBA" and (arr[..., 3] == 128).all()
    assert arr[..., 0].std() > 1
