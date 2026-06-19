import os
import random
import shutil
import tempfile
import warnings
from pathlib import Path

from PIL import Image

from generator.components import BackgroundManager


def test_load_with_valid_image(temp_image_dir):
    bm = BackgroundManager(temp_image_dir)
    crop = bm.get_background_crop((10, 10))
    assert isinstance(crop, Image.Image)
    assert crop.size == (10, 10)


def test_no_directory_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bm = BackgroundManager(None)
        assert len(w) == 1
        assert "No background images found" in str(w[0].message)
    crop = bm.get_background_crop((5, 5))
    assert crop.size == (5, 5)
    assert crop.getpixel((0, 0)) == (255, 255, 255)


def test_empty_directory_warns():
    empty_dir = tempfile.mkdtemp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = BackgroundManager(empty_dir)
        assert len(w) == 1
        assert "No background images found" in str(w[0].message)
    shutil.rmtree(empty_dir)


def test_invalid_image_file():
    temp_dir = tempfile.mkdtemp()
    bad_img_path = os.path.join(temp_dir, "bg.png")
    Path(bad_img_path).write_bytes(b"not an image")
    bm = BackgroundManager(temp_dir)
    # Force choice to select the bad image file
    random.seed(0)
    crop = bm.get_background_crop((10, 10))
    assert crop.size == (10, 10)  # falls back to white after attempts
    shutil.rmtree(temp_dir)


def test_image_smaller_than_crop_returns_blank(temp_image_dir):
    # Replace valid image with a very small one
    img = Image.new("RGB", (5, 5), (0, 0, 0))
    img.save(os.path.join(temp_image_dir, "small.png"))
    bm = BackgroundManager(temp_image_dir)
    crop = bm.get_background_crop((50, 50))  # Larger than image
    assert crop.size == (50, 50)


def test_background_cache_reuse(temp_image_dir):
    # A background is decoded once and then served from the in-memory cache.
    bm = BackgroundManager(temp_image_dir, cache_size=4)
    path = bm.bg_images[0]
    first = bm._get_image(path)
    second = bm._get_image(path)
    assert first is second
    assert path in bm._cache


def test_background_cache_is_bounded(temp_image_dir):
    # Adding more distinct images than the cache size must not grow it unbounded.
    for i in range(6):
        Image.new("RGB", (60, 60), (i * 10, i * 10, i * 10)).save(os.path.join(temp_image_dir, f"extra_{i}.png"))
    bm = BackgroundManager(temp_image_dir, cache_size=3)
    for path in bm.bg_images:
        bm._get_image(path)
    assert len(bm._cache) <= 3


def test_background_max_dimension_downscales():
    temp_dir = tempfile.mkdtemp()
    Image.new("RGB", (800, 400), (120, 120, 120)).save(os.path.join(temp_dir, "big.png"))
    bm = BackgroundManager(temp_dir, cache_size=2, max_dimension=200)
    img = bm._get_image(bm.bg_images[0])
    assert max(img.size) <= 200
    shutil.rmtree(temp_dir)
