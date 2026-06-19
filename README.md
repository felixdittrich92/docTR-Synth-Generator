[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Build Status](https://github.com/felixdittrich92/docTR-Synth-Generator/workflows/builds/badge.svg)
[![codecov](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator/graph/badge.svg?token=31MDR20JGI)](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator)
[![CodeFactor](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator/badge)](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator)
[![Pypi](https://img.shields.io/badge/pypi-v0.0.1-blue.svg)](https://pypi.org/project/docTR-Synth-Generator/)

# docTR-Synth-Generator
A tool to generate synthetic OCR text recognition datasets - made for docTR

# WORK IN PROGRESS

## Features

- **Zero-config**: generate a dataset with nothing but an output directory - real
  words, matching fonts *and* background images are downloaded automatically.
- **Multilingual by language code**: `languages=["de", "ru", "ar", ...]` resolves
  both the words *and* the fonts for each script (~85 languages), with correct
  complex-script shaping and right-to-left layout for Arabic/Hebrew.
- **No more dropped words**: any character a local font cannot render triggers an
  on-demand download of a font that can, instead of silently skipping the word.
- **Realistic output**: supersampled anti-aliasing, background-aware ink colour
  and contrast (dark-on-light and light-on-dark), faux-bold/outlines, and
  scanner/camera-style degradations (JPEG artifacts, sensor noise, blur).
- **Controllable balancing**: explicit per-language allocation, a stratified
  train/val split, optional character-coverage guarantees, and a balance report.
- **Recognition *and* detection**: produce word/line crops for recognition, or
  full document-like pages with per-word polygons for detection - both in the
  formats docTR's training references expect.
- **Fast & memory-bounded**: font objects and decoded backgrounds are cached, with
  a configurable cache size.

## Quickstart (zero configuration)

You no longer need to provide a wordlist or a font directory. With nothing but an
output directory and a count, the generator downloads real words for the
requested language(s) and automatically fetches matching open-source fonts:

```python
from generator import GenerationConfig, SyntheticDatasetGenerator

config = GenerationConfig(output_dir="output_dataset", num_images=1000)  # English by default
SyntheticDatasetGenerator(config).generate_dataset()
```

Multilingual is a one-liner - a language code selects both its words *and* its
script, so the correct fonts are pulled in for you:

```python
config = GenerationConfig(
    output_dir="output_dataset",
    num_images=10000,
    languages=["en", "de", "ru", "el", "ar"],  # words + fonts resolved automatically
    bg_image_dir="resources/background_images",  # optional; blank backgrounds otherwise
)
SyntheticDatasetGenerator(config).generate_dataset()
```

> The first run downloads word lists and fonts from public mirrors and caches
> them (`corpus_cache_dir` / `font_cache_dir`). Subsequent runs are offline. To
> run fully offline from the start, supply your own `wordlist_path` and
> `font_dir`.

## Bring your own resources (classic usage)

Supplying a `wordlist_path` and/or `font_dir` still works and takes precedence
over the automatic downloads:

```python
config = GenerationConfig(
    wordlist_path="resources/corpus/latin_ext_balanced_words.txt",
    font_dir="resources/font",  # e.g. the extracted fonts_v1 release
    bg_image_dir="resources/background_images",  # bundled with the repo
    output_dir="output_dataset",
    num_images=1000,
    val_percent=0.2,
    num_workers=6,
    # If a word contains characters none of your local fonts cover, download a
    # matching font instead of dropping the word (default: True):
    auto_download_fonts=True,
)
SyntheticDatasetGenerator(config).generate_dataset()
```

## Automatic fonts

When no local font covers every character of a word, a matching open-source font
(from the [Noto](https://fonts.google.com/noto) family, which spans the whole
Unicode range) is downloaded, verified for coverage and cached. This prevents
words from being silently skipped - the main cause of biased, latin-only
datasets. Disable with `auto_download_fonts=False`.

## Automatic words

When no `wordlist_path` is given, real frequency-ranked words for `languages`
are downloaded (from the open
[FrequencyWords](https://github.com/hermitdave/FrequencyWords) project, ~85
languages) and cleaned (script filtering, length bounds, punctuation removal).
Two realism helpers are applied by default and can be tuned or disabled:

- `casing_variant_prob` (0.3): adds Title/UPPERCASE variants so the model sees
  capital letters (frequency lists are almost all lowercase).
- `numeric_token_ratio` (0.05): mixes in realistic numbers, dates, prices and
  codes - the kind of content real documents are full of.

## Automatic backgrounds

When no `bg_image_dir` is given, a curated set of background images is downloaded
and cached automatically (instead of producing blank backgrounds). Supplying your
own `bg_image_dir` takes precedence and skips the download entirely - exactly like
fonts and word lists. Disable with `auto_download_backgrounds=False`, point
`background_cache_dir` somewhere persistent, or pass a `background_manifest_url`
(a newline-separated list of filenames/URLs) to use a different collection.

## Dataset balancing

For multilingual runs the language mix is explicit and controllable instead of
being dominated by whichever language has the most words:

```python
config = GenerationConfig(
    output_dir="output_dataset",
    num_images=30000,
    languages=["en", "de", "ru"],
    language_balance="balanced",  # "balanced" (default) or "proportional"
    # language_weights={"en": 0.6, "de": 0.3, "ru": 0.1},  # or set explicit weights
    min_char_coverage=20,  # ensure every character appears >= N times (0 = off)
)
```

The split is *stratified*: train and val share the same language mix and exact
words do not leak from train into val. A balance report is printed before
generation (per-language train/val counts, train/val overlap, distinct/rare
characters, word-length statistics); silence it with
`print_balance_report=False`.

## Detection datasets

Set `task="detection"` to generate document-like **pages** with a 4-point
polygon for every word, ready for
[docTR detection training](https://github.com/mindee/doctr/tree/main/references/detection):

```python
config = GenerationConfig(
    task="detection",
    output_dir="detection_dataset",
    num_images=5000,  # = number of pages
    languages=["en", "de"],  # words + fonts resolved automatically
    bg_image_dir="resources/background_images",
    output_jpeg=True,
)
SyntheticDatasetGenerator(config).generate_dataset()
```

Each split is written as `images/` plus a `labels.json` in the exact docTR
format (absolute pixel coordinates):

```json
{
  "00000.jpg": {
    "img_dimensions": [1462, 1056],
    "img_hash": "<sha256 of the image>",
    "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
  }
}
```

It reuses the same fonts, ink styling, contrast, backgrounds and degradations as
the recognition path, and lays words out in paragraph blocks with margins, line
wrapping, occasional headings/indents, numbers and dates, and an optional small
global page rotation (the polygons rotate with the page, giving rotated boxes
usable with docTR's `use_polygons=True`). Tune layout with the `det_*` config
fields (`det_page_*_range`, `det_font_size_range`, `det_words_per_page_range`,
`det_max_blocks`, `det_rotation_*`, ...).

> **Backgrounds for detection:** only the words *you* place are labelled, so any
> text already printed in a background photo becomes an unlabelled false
> negative. `det_plain_background_prob` (0.4) mixes in clean generated paper;
> set it to `1.0` for all-paper pages, or point `bg_image_dir` at **text-free**
> textures (plain paper, surfaces, fabrics) only.

Non-Latin scripts work out of the box: words and fonts are resolved per language,
complex scripts are shaped correctly (Arabic joining, Indic conjuncts), and
right-to-left languages (Arabic, Hebrew, ...) are laid out right-to-left so pages
read naturally. For example `languages=["ar"]`, `["he"]`, `["zh"]` or `["hi"]`.

## Realism

Rendered crops are meant to match real captured documents rather than clean
synthetic glyphs. The pipeline applies, all configurable:

- Supersampled rendering with high-quality downsampling for photographic
  anti-aliasing (`supersample`).
- Background-aware ink: dark-on-light **and** light-on-dark text, a controllable
  (often deliberately low) contrast range, neutral or coloured ink, variable
  opacity, faux-bold and outlines.
- Glyph-space augmentations before compositing (rotation, perspective, ink
  erosion) and image-space degradations after (Gaussian sensor noise, JPEG
  compression artifacts, blur, brightness/contrast jitter) - matching how a real
  capture degrades the whole frame.
- Optional JPEG output (`output_jpeg=True`) to match real document captures.

## Performance & memory

Font objects and decoded background images are cached, giving a large throughput
improvement over re-loading them per sample. Memory stays bounded and tunable:

- `bg_cache_size` (16): number of decoded backgrounds held in memory per worker.
  Lower it on memory-constrained machines or with many workers; raise it for more
  background variety. `bg_max_dimension` (2000) downscales very large backgrounds
  on load so the cache stays light regardless of source resolution.
- Caches are per worker process, so peak memory scales roughly with
  `num_workers`.

## Configuration reference

All behaviour is controlled through `GenerationConfig`; see the dataclass
docstring in `generator/components/config.py` for every field and its default.

## Resources

- **fonts_v1**: A collection of fonts used for text rendering can be downloaded from [Fonts_v1](https://github.com/felixdittrich92/docTR-Synth-Generator/releases/download/v0.0.1/fonts_v1.zip).
- **background_images_v1**: A collection of background images used for text rendering can be downloaded from [Background_Images_v1](https://github.com/felixdittrich92/docTR-Synth-Generator/releases/download/v0.0.1/background_images_v1.zip).

## Citation

If you wish to cite please refer to the base project citation, feel free to use this [BibTeX](http://www.bibtex.org/) references:

```bibtex
@misc{docTR-Synth-Generator,
    title={docTR-Synth-Generator: A tool to generate synthetic OCR text recognition datasets - made for docTR},
    author={{Dittrich, Felix}},
    year={2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/felixdittrich92/docTR-Synth-Generator}}
}
```

The automatic word lists are derived from the
[FrequencyWords](https://github.com/hermitdave/FrequencyWords) project
(OpenSubtitles-based) and fonts from [Google Fonts / Noto](https://fonts.google.com/noto);
please respect their respective licenses when redistributing generated datasets.

## Development & tests

The test suite is fully offline - it builds a tiny in-memory font with
`fontTools` and monkeypatches the network downloads, so no fonts or corpora are
fetched while testing. Run it with:

```bash
make test      # pytest + coverage (fails under 70%)
make quality   # ruff + mypy
make style     # auto-format and fix
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create.

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Add your Changes
4. Run the tests and quality checks (`make test` and `make style` and `make quality`)
5. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the Branch (`git push origin feature/AmazingFeature`)

## License

Distributed under the Apache 2.0 License. See [`LICENSE`](https://github.com/felixdittrich92/OnnxTR?tab=Apache-2.0-1-ov-file#readme) for more information.
