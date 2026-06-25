[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Build Status](https://github.com/felixdittrich92/docTR-Synth-Generator/workflows/builds/badge.svg)
[![codecov](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator/graph/badge.svg?token=31MDR20JGI)](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator)
[![CodeFactor](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator/badge)](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator)
[![Pypi](https://img.shields.io/badge/pypi-v0.4.0a0-blue.svg)](https://pypi.org/project/docTR-Synth-Generator/)

# docTR-Synth-Generator
A tool to generate synthetic OCR datasets - made for docTR

![Examples: detection pages and recognition crops](https://github.com/felixdittrich92/docTR-Synth-Generator/raw/main/docs/examples_grid.png)

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

## Quickstart

One call - the words, fonts and backgrounds it needs are downloaded and cached
automatically. English by default:

```python
from generator import generate_dataset

generate_dataset("output_dataset", num_images=1000)
```

Multilingual is just a list of ISO 639-1 codes; each code selects its words *and*
its script, so matching fonts are pulled in and complex scripts are shaped
correctly:

```python
generate_dataset("output_dataset", num_images=10000, languages=["en", "de", "ru", "el", "ar"])
```

Detection pages instead of recognition crops:

```python
generate_dataset("pages", num_images=5000, task="detection", languages=["en", "de"])
```

...or straight from the command line (installs as `doctr-synth`, also runnable as `python -m generator`):

```bash
doctr-synth output_dataset -n 10000 -l en de ru
doctr-synth pages -t detection -l en de --layout newspaper
```

> The first run downloads word lists, fonts and backgrounds from public mirrors
> and caches them; later runs are offline. To stay fully offline from the start,
> supply your own `wordlist_path` and `font_dir` (see below).

`generate_dataset(...)` is a thin wrapper over the `GenerationConfig` +
`SyntheticDatasetGenerator` pair you can still use directly for full control. The
config is organised into focused sub-configs (`core`, `resources`, `corpus`,
`balance`, `coverage`, `recognition`, `realism`, `detection`) - build it nested,
or use `GenerationConfig.flat(...)` to pass flat keyword names. Any of those
keywords can also be passed straight to `generate_dataset(...)`:

```python
from generator import GenerationConfig, CoreConfig, DetectionConfig

# nested - group related options together
cfg = GenerationConfig(
    core=CoreConfig(num_images=10_000, task="detection", languages=["en", "de"]),
    detection=DetectionConfig(layout="newspaper"),
)

# ...or flat, routed into the sub-configs for you
cfg = GenerationConfig.flat(num_images=10_000, task="detection", det_layout="newspaper")

# ...or just the one-liner
generate_dataset("output_dataset", num_images=10_000, languages=["en", "de"], output_jpeg=True, num_workers=8)
```

### Bring your own resources

`wordlist_path` and/or `font_dir` take precedence over the automatic downloads
(e.g. the bundled `resources/` or the `fonts_v1` release):

```python
generate_dataset(
    "output_dataset",
    num_images=1000,
    wordlist_path="resources/corpus/latin_ext_balanced_words.txt",
    font_dir="resources/font",
    bg_image_dir="resources/background_images",
)
```

## Automatic resources

On the first run anything you don't provide is fetched from public mirrors and
cached (later runs are offline). Providing your own `font_dir` / `wordlist_path`
/ `bg_image_dir` takes precedence and skips the matching download.

- **Fonts** - when no local font covers every character of a word, a matching
  open-source [Noto](https://fonts.google.com/noto) font is downloaded, verified
  and cached, so words are never silently skipped (the usual cause of biased,
  latin-only datasets). Disable with `auto_download_fonts=False`.
- **Words** - with no `wordlist_path`, real frequency-ranked words are downloaded
  and cleaned (script filter, length bounds, punctuation removal). They come from
  [FrequencyWords](https://github.com/hermitdave/FrequencyWords) first, falling
  back to [most-common-words-multilingual](https://github.com/frekwencja/most-common-words-multilingual)
  for the many languages it lacks (Odia, Khmer, Burmese and most Indic/SE-Asian
  scripts), so far fewer languages need synthesised words. Any character still
  missing from the corpus is then synthesised. Two realism helpers are on by
  default: `casing_variant_prob` (0.3) adds Title/UPPERCASE variants, and
  `numeric_token_ratio` (0.05) mixes in numbers, dates and prices.
- **Backgrounds** - with no `bg_image_dir`, a curated background set is downloaded
  instead of blank pages. Disable with `auto_download_backgrounds=False`, or pass
  a `background_manifest_url` for your own collection.

## Dataset balancing

For multilingual runs the language mix is explicit and controllable instead of
being dominated by whichever language has the most words:

```python
config = GenerationConfig.flat(
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

## Supported languages

Pass ISO 639-1 codes in `languages=[...]`. Each code resolves three things
automatically: real words from a public frequency list (when one exists), a
script-matching open-source font (downloaded on demand), and the docTR vocab
used for coverage and for the recognition vocab restriction. Complex scripts are
shaped correctly - Arabic, Hebrew and Urdu run right-to-left, and Indic, Thai and
Myanmar clusters (consonant conjuncts, matras, medials, vowel signs and viramas)
are built as valid grapheme clusters, including in synthesised coverage tokens.

| Script | Languages (ISO 639-1 code) |
| --- | --- |
| Latin | Afrikaans (`af`), Azerbaijani (`az`), Catalan (`ca`), Czech (`cs`), Danish (`da`), Dutch (`nl`), English (`en`), Estonian (`et`), Basque (`eu`), Finnish (`fi`), French (`fr`), German (`de`), Hungarian (`hu`), Icelandic (`is`), Indonesian (`id`), Irish (`ga`), Italian (`it`), Latvian (`lv`), Lithuanian (`lt`), Maltese (`mt`), Norwegian (`no`/`nb`), Polish (`pl`), Portuguese (`pt`), Romanian (`ro`), Slovak (`sk`), Slovene (`sl`), Spanish (`es`), Albanian (`sq`), Swedish (`sv`), Croatian (`hr`), Turkish (`tr`), Vietnamese (`vi`) |
| Cyrillic | Belarusian (`be`), Bulgarian (`bg`), Macedonian (`mk`), Russian (`ru`), Ukrainian (`uk`) |
| Greek | Greek (`el`) |
| Perso-Arabic | Arabic (`ar`), Persian (`fa`), Urdu (`ur`) |
| Hebrew | Hebrew (`he`) |
| Armenian | Armenian (`hy`) |
| Georgian | Georgian (`ka`) |
| Devanagari | Hindi (`hi`), Marathi (`mr`) |
| Bengali | Bengali (`bn`) |
| Gujarati | Gujarati (`gu`) |
| Tamil | Tamil (`ta`) |
| Telugu | Telugu (`te`) |
| Kannada | Kannada (`kn`) |
| Malayalam | Malayalam (`ml`) |
| Oriya | Odia (`or`) |
| Sinhala | Sinhala (`si`) |
| Thai | Thai (`th`) |
| Myanmar | Burmese (`my`) |
| CJK | Japanese (`ja`), Korean (`ko`) - corpus-driven only (no fixed small vocab, so vocab-coverage synthesis is skipped) |

A few notes:

- A handful of languages have a vocab and a font but no public frequency list
  (e.g. Burmese `my`, Odia `or`). They render correctly and their full character
  set is still guaranteed through synthesised coverage tokens - you just won't
  get a real-word corpus unless you supply your own via `wordlist_path`.
- For **recognition**, any of the 214 keys in ``VOCABS`` (e.g. `"german"`,
  `"arabic"`, `"hindi"`) can be used as ``target_vocab`` / the ``vocab`` argument,
  and several may be combined for a multilingual model
  (`["german", "urdu", "odia"]`); every generated label is then restricted to
  that exact character set so a docTR model trained on the matching vocab never
  sees an un-encodable character (see the docTR training section).

## Vocabulary coverage (recognition)

A recognition model is trained against a fixed character set (docTR's `VOCABS`).
Real frequency corpora rarely contain *every* character of that set - rare
accented capitals (`ẞ`), currency signs, some punctuation - so a model trained
only on downloaded words never sees them. With `ensure_vocab_coverage=True`
(the default), each language is mapped to its docTR vocab and extra word-like
tokens are synthesised so **every renderable vocab character appears in both the
train and val splits**:

```python
config = GenerationConfig.flat(
    output_dir="dataset",
    num_images=50000,
    languages=["de"],  # mapped to the "german" vocab automatically
    ensure_vocab_coverage=True,  # default
    vocab_coverage_min_count=3,  # each vocab char appears in >= N train samples
)
```

- `target_vocab` overrides the per-language mapping - pass a `VOCABS` key
  (e.g. `"german"`) or a literal string of characters to cover. It also enables
  coverage when you supply your own `wordlist_path`.
- Coverage is enforced **after** the train/val split, so a rare character can
  never land in only one split. This makes `num_images` a *floor*: a small,
  bounded number of coverage samples (proportional to the vocab size, not the
  dataset) is appended on top.
- Languages with no fixed small vocab (CJK) are skipped automatically, and very
  large scripts (thousands of CJK ideographs / Hangul syllables) are left to the
  real corpus rather than synthesised.
- Every synthesised token stays within a single script (a Hebrew character is
  only ever placed in a Hebrew token, etc.), so each renders with one font. When
  several languages are generated together, coverage is computed over the union
  of their vocabs but tokens are never mixed across scripts.
- Coverage prefers repeating **real corpus words** that contain a rare
  character, so diacritic combinations are linguistically attested. Synthesis
  is only a fallback for characters absent from the corpus (rare punctuation,
  currency, capitals in a lower-cased corpus, or marks like Hebrew niqqud that
  real text omits) - and even then a combining mark is inserted into a real
  same-script word after a base letter, never rendered alone on a dotted circle.
- A character that **no available font can render** (e.g. `฿` inside a
  Latin-script vocab) is the one case that cannot be covered - that is a font
  limitation, reported in the log, not a logic gap.

## Detection datasets

Set `task="detection"` to generate document-like **pages** with a 4-point
polygon for every word, ready for
[docTR detection training](https://github.com/mindee/doctr/tree/main/references/detection):

```python
config = GenerationConfig.flat(
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
the recognition path. Pages are filled top-to-bottom by the available vertical
space (word count varies naturally with font size), and words are recycled as
needed so a page always fills regardless of how many candidate words it is given.

### Real-world layouts

To better mimic real documents, the layout is chosen per page via `det_layout`:

- `"paragraph"` - multi-block running text with headings and indents.
- `"newspaper"` - a full-width masthead with a double rule and a dateline, then
  several narrow columns separated by vertical rules, each with article
  headlines, bylines and small, tightly-leaded body text (~500-1100 words on an
  A4-ish page). Tune density with `det_newspaper_columns_range` (default
  `(3, 6)`, clamped to the page width), `det_newspaper_font_size_range` (default
  `(9, 15)`) and `det_newspaper_line_spacing_range` (default `(1.05, 1.2)`).
- `"form"` - a title with a header rule, then `Label:` / value rows with either
  underlines or boxed fields, shaded section-header bars, and occasional
  checkbox rows.
- `"id_card"` - a card with a coloured issuing-authority header band (emblem +
  light title text), a photo placeholder, labelled field rows, a signature line
  and MRZ-style lines. Mirrors fully for right-to-left scripts.
- `"mixed"` (default) - a weighted blend of the above; tune via
  `det_layout_weights` (e.g. `{"paragraph": 0.4, "newspaper": 0.25, "form": 0.2,
  "id_card": 0.15}`).

Forms and ID cards always render on clean generated paper. All layouts emit the
same per-word polygons, and the optional small global page rotation
(`det_rotation_*`) rotates the polygons with the page for use with docTR's
`use_polygons=True`. Other layout knobs: `det_page_*_range`, `det_font_size_range`,
`det_max_blocks`, `det_margin_ratio`, `det_heading_prob`.

> **Backgrounds for detection:** only the words *you* place are labelled, so any
> text already printed in a background photo becomes an unlabelled false
> negative. `det_plain_background_prob` (0.4) mixes in clean generated paper;
> set it to `1.0` for all-paper pages, or point `bg_image_dir` at **text-free**
> textures (plain paper, surfaces, fabrics) only.

Non-Latin scripts work out of the box: words and fonts are resolved per language,
complex scripts are shaped correctly (Arabic joining, Indic conjuncts), and
right-to-left languages (Arabic, Hebrew, ...) are laid out right-to-left so pages
read naturally. For example `languages=["ar"]`, `["he"]`, `["zh"]` or `["hi"]`.

## Plug into docTR training (on-the-fly, in-RAM)

You can skip writing a dataset to disk entirely and feed freshly synthesised
samples straight into docTR's training scripts. `generator/doctr_dataset.py`
provides PyTorch `Dataset` wrappers that generate one sample per
`__getitem__`, matching docTR's dataset contract - `(image_tensor, target)` per
sample plus a static `collate_fn` - so they drop into the existing `DataLoader`
in
[`references/detection/train.py`](https://github.com/mindee/doctr/blob/main/references/detection/train.py)
and
[`references/recognition/train.py`](https://github.com/mindee/doctr/blob/main/references/recognition/train.py).

Targets are identical to docTR's own datasets, so the model transforms and loss
treat them the same: recognition yields the label string; detection yields
`{CLASS_NAME: geoms}` with absolute-pixel polygons `(N, 4, 2)` when
`use_polygons=True` else straight boxes `(N, 4)` as `[xmin, ymin, xmax, ymax]`.

**Detection** - in `references/detection/train.py`, replace the
`DetectionDataset(...)` construction (keep the `DataLoader` lines):

```python
from generator.components import GenerationConfig
from generator.doctr_dataset import build_detection_datasets, synth_worker_init_fn

# num_images is the total dataset size; val_percent carves out the val set.
cfg = GenerationConfig.flat(task="detection", num_images=50_000, val_percent=0.2)
train_set, val_set = build_detection_datasets(
    cfg,
    languages=["german", "english", "french", "italian", "spanish", "malay"],  # VOCABS keys or ISO codes
    use_polygons=args.rotation,  # straight boxes unless --rotation
    sample_transforms=sample_transforms,  # the script's image+target transforms
    img_transforms=img_transforms,  # the script's image-only transforms
)
# len(train_set) == 40_000, len(val_set) == 10_000
```

**Recognition** - in `references/recognition/train.py`, replace the
`RecognitionDataset(...)` construction:

```python
from generator.components import GenerationConfig
from generator.doctr_dataset import build_recognition_datasets, synth_worker_init_fn

cfg = GenerationConfig.flat(task="recognition", num_images=10_000, val_percent=0.2)
train_set, val_set = build_recognition_datasets(
    cfg,
    vocab=args.vocab,  # e.g. ["multilingual"] - a VOCABS key / list = the model's vocab
    img_transforms=img_transforms,  # the script's existing resize/aug
)
# len(train_set) == 8_000, len(val_set) == 2_000
```

Pass `vocab` the **same vocab you train the model on** - a `VOCABS` key, a
literal charset, or a list of keys whose union is the model's vocab (e.g.
`["multilingual"]` or `["german", "urdu", "odia"]`). Every generated label is
then guaranteed to contain only characters in that vocab, so docTR's label
encoder never hits an out-of-vocab character (which would otherwise crash
training on the first batch). Corpus words outside the vocab are dropped,
character coverage is guaranteed *within* the vocab (missing characters are
synthesised so the set stays balanced), and the corpus languages are derived
from the vocab keys automatically (`german` -> `de`). For detection, pass
`languages=[...]` (VOCABS keys like `"german"`, `"malay"`, or ISO codes); for
recognition you can pass `languages=[...]` too. A combined vocab like
`"multilingual"` maps to no single language, so pass `languages=[...]` if you
want real multilingual words rather than mostly-synthesised coverage. The same
restriction applies to the offline generator via
`GenerationConfig.flat(target_vocab=[...])`.

The `DataLoader` lines stay as they are - just keep
`collate_fn=train_set.collate_fn` and add `worker_init_fn=synth_worker_init_fn`
so every worker gets an independent RNG stream:

```python
train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
    collate_fn=train_set.collate_fn,
    worker_init_fn=synth_worker_init_fn,
)
```

Notes:

- **Dataset size & split.** `num_images` is the total dataset size; `val_percent`
  carves out the validation set, so `len(train_set)` / `len(val_set)` follow
  directly (50k @ 0.2 -> 40k / 10k). Samples are generated fresh on every access,
  so this is effectively the iterations per epoch. Pass `train_samples` /
  `val_samples` only if you want to override the derived sizes.
- **Seeding.** The train set draws a fresh random sample on every access (new
  data every epoch - the whole point of on-the-fly); the val set is a
  reproducible fixed virtual set (seeded per index) so metrics stay comparable.
- **Coverage carries over.** The recognition pools come from the same balancing
  and per-split character-coverage pipeline as the offline generator, so
  sampling from them covers the target vocab.
- **One-time setup.** Corpora, fonts and backgrounds are downloaded/resolved
  once when the datasets are built (in the parent process), not per worker.
- Requires PyTorch in your training environment (`pip install python-doctr`).
  Importing the rest of this package never requires torch. For lower-level
  control you can use `SyntheticDetectionDataset` / `SyntheticRecognitionDataset`
  directly instead of the `build_*` factories.

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

The config is organised into focused sub-configs - `core`, `resources`,
`corpus`, `balance`, `coverage`, `recognition`, `realism` and `detection` -
each a small dataclass you can construct on its own. Build `GenerationConfig`
nested (`GenerationConfig(detection=DetectionConfig(layout="form"))`), or pass
flat keyword names via `GenerationConfig.flat(...)` / `generate_dataset(...)`,
which route each keyword into the right sub-config.

Most runs need only a handful of options - the ones you are most likely to set
(as flat keywords):

| Option | Sub-config | Default | What it does |
| --- | --- | --- | --- |
| `output_dir` | core | - | where the dataset is written (`train/`, `val/`) |
| `num_images` | core | `1000` | total samples (split by `val_percent`) |
| `task` | core | `"recognition"` | `"recognition"` crops or `"detection"` pages |
| `languages` | core | `["en"]` | ISO 639-1 codes; resolves words, fonts and shaping |
| `val_percent` | core | `0.2` | validation fraction |
| `num_workers` | core | `4` | parallel worker processes |
| `output_jpeg` | core | `False` | write JPEG instead of PNG |
| `target_vocab` | coverage | `None` | recognition: restrict labels to a `VOCABS` key / list (the `vocab=` arg) |
| `det_layout` | detection | `"mixed"` | detection: `mixed`/`paragraph`/`newspaper`/`form`/`id_card` |
| `language_balance` | balance | `"balanced"` | `"balanced"` or `"proportional"` allocation across languages |
| `min_char_coverage` | balance | `0` | ensure every character appears >= N times (0 = off) |
| `wordlist_path` / `font_dir` / `bg_image_dir` | resources | `None` | bring your own resources (skips the matching download) |

For the complete set of options (realism, augmentation and detection-layout
knobs), see the sub-config dataclasses in `generator/components/config.py`.

## Resources

- **fonts_v1**: A collection of fonts used for text rendering can be downloaded from [Fonts_v1](https://github.com/felixdittrich92/docTR-Synth-Generator/releases/download/v0.0.1/fonts_v1.zip).
- **background_images_v1**: A collection of background images used for text rendering can be downloaded from [Background_Images_v1](https://github.com/felixdittrich92/docTR-Synth-Generator/releases/download/v0.0.1/background_images_v1.zip).

## Citation

If you wish to cite please refer to the base project citation, feel free to use this [BibTeX](http://www.bibtex.org/) references:

```bibtex
@misc{docTR-Synth-Generator,
    title={docTR-Synth-Generator: A tool to generate synthetic OCR text datasets - made for docTR},
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
make test      # pytest + coverage
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
