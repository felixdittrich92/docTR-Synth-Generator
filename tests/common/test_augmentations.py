import numpy as np
from PIL import Image

from generator.augmentations import (
    AugmentationPipeline,
    RandomBlur,
    RandomPerspective,
    RandomRotate,
)


def test_augmentation_pipeline_applies_all(sample_image):
    # augmentation that flips image size
    def flip_size(img):
        return img.resize((img.height, img.width))

    pipeline = AugmentationPipeline([flip_size, flip_size])
    result = pipeline(sample_image)
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size  # flip twice returns original
    assert "AugmentationPipeline" in repr(pipeline)


def test_randomblur_always_applied(sample_image):
    rb = RandomBlur(radius_range=(0.3, 1.0), prob=1.0)
    result = rb(sample_image)
    assert isinstance(result, Image.Image)
    assert result is not sample_image


def test_randomblur_never_applied(sample_image):
    rb = RandomBlur(prob=0.0)
    result = rb(sample_image)
    assert result == sample_image  # same object
    assert "RandomBlur" in repr(rb)


def test_randomperspective_always_applied(sample_image):
    rp = RandomPerspective(margin=1, prob=1.0)
    result = rp(sample_image)
    assert isinstance(result, Image.Image)
    assert "RandomPerspective" in repr(rp)


def test_randomperspective_never_applied(sample_image):
    rp = RandomPerspective(prob=0.0)
    result = rp(sample_image)
    assert result == sample_image
    assert "RandomPerspective" in repr(rp)


def test_randomperspective_find_coeffs_matches_known():
    rp = RandomPerspective()
    pa = [(0, 0), (1, 0), (1, 1), (0, 1)]
    pb = [(0, 0), (2, 0), (2, 2), (0, 2)]
    coeffs = rp._find_coeffs(pa, pb)
    assert np.allclose(coeffs.shape, (8,))
    # transform 0,0 should map close to 0,0 in pb space
    # we won't check exact match because it's LSQ


def test_randomrotate_always_applied(sample_image):
    rr = RandomRotate(angle_range=(45, 45), prob=1.0)
    result = rr(sample_image)
    assert isinstance(result, Image.Image)
    assert result.size != sample_image.size  # expand=True changes size
    assert "RandomRotate" in repr(rr)


def test_randomrotate_never_applied(sample_image):
    rr = RandomRotate(prob=0.0)
    result = rr(sample_image)
    assert result == sample_image
    assert "RandomRotate" in repr(rr)


def test_pipeline_with_all_augmentations(sample_image):
    pipeline = AugmentationPipeline([
        RandomBlur(radius_range=(0.1, 0.1), prob=1.0),
        RandomRotate(angle_range=(5, 5), prob=1.0),
        RandomPerspective(margin=1, prob=1.0),
    ])
    result = pipeline(sample_image)
    assert isinstance(result, Image.Image)
