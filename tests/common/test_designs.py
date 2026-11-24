import random

import numpy as np
import pytest

from generator.components.background_manager import BackgroundManager
from generator.components.designs import design_background, design_palette


@pytest.mark.parametrize("seed", range(8))
def test_designs_fill_the_requested_size(seed):
    random.seed(seed)
    np.random.seed(seed)
    img = design_background((240, 320))
    assert img.size == (240, 320) and img.mode == "RGB"
    assert len(design_palette()) >= 4


def test_designs_vary():
    means = []
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        means.append(np.asarray(design_background((64, 64)), dtype=np.float32).mean(axis=(0, 1)))
    assert np.std(means, axis=0).max() > 20  # different colour schemes, not one look


def test_background_manager_mixes_in_designs_without_any_images():
    random.seed(0)
    np.random.seed(0)
    bm = BackgroundManager(None, procedural_prob=1.0)
    crop = bm.get_background_crop((120, 40))
    page = bm.get_page_background((300, 400))
    assert crop.size == (120, 40) and page.size == (300, 400)
    assert len({tuple(p) for p in np.asarray(crop).reshape(-1, 3)[::50]}) > 1  # not a flat blank
    assert len(bm._designs) == 1  # crops come from the pool
    plain = BackgroundManager(None)  # default: images only, blank without them
    assert np.asarray(plain.get_background_crop((20, 20))).min() == 255
