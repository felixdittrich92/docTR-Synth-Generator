import os
import shutil
import tempfile

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    return Image.new("RGB", (10, 10), color="red")


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with a sample image."""
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "bg.png")
    img = Image.new("RGB", (50, 50), (100, 100, 100))
    img.save(img_path)
    yield temp_dir
    shutil.rmtree(temp_dir)
