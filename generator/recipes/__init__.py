"""Ready-made, large-scale dataset builds on top of the generator.

* :mod:`generator.recipes.recognition` - word crops with a guaranteed-complete vocab.
* :mod:`generator.recipes.detection` - document pages with per-word polygons.

Both are command line programs (``python -m generator.recipes.recognition --help``).
"""

from .languages import LANGUAGES, PRESETS, resolve_languages
