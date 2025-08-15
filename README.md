[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Build Status](https://github.com/felixdittrich92/docTR-Synth-Generator/workflows/builds/badge.svg)
[![codecov](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator/graph/badge.svg?token=31MDR20JGI)](https://codecov.io/gh/felixdittrich92/docTR-Synth-Generator)
[![CodeFactor](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator/badge)](https://www.codefactor.io/repository/github/felixdittrich92/doctr-synth-generator)
[![Pypi](https://img.shields.io/badge/pypi-v0.0.1-blue.svg)](https://pypi.org/project/docTR-Synth-Generator/)

# docTR-Synth-Generator
A tool to generate synthetic OCR text recognition datasets - made for docTR

# WORK IN PROGRESS

# NOTE: This is only a quick draft and will change in the next time.

```python

from generator import GenerationConfig, SyntheticDatasetGenerator

config = GenerationConfig(
    wordlist_path="/home/felix/Desktop/Synth_doctr/resources/corpus/latin_ext_balanced_words.txt",
    font_dir="/home/felix/Desktop/Synth_doctr/resources/font",
    bg_image_dir="/home/felix/Desktop/Synth_doctr/resources/image",
    output_dir="output_dataset",
    num_images=1000,
    val_percent=0.2,
    num_workers=6,  # Start with fewer workers to avoid memory issues
    queue_maxsize=100,  # Limit queue size
    font_size_range=(18, 35),
    padding=2,
    max_attempts=5,
    # Augmentation settings
    bold_prob=0.1,
    rotation_prob=0.5,
    blur_prob=0.3,
    perspective_prob=0.3,
    rotation_range=(-2, 2),
    blur_radius_range=(0.3, 1.0),
    perspective_margin=2,
)

generator = SyntheticDatasetGenerator(config)
generator.generate_dataset()
```

## Citation

If you wish to cite please refer to the base project citation, feel free to use this [BibTeX](http://www.bibtex.org/) references:

```bibtex
@misc{docTR-Synth-Generator,
    title={docTR-Synth-Generator: A tool to generate synthetic OCR text recognition datasets - made for docTR},
    author={{Dittrich, Felix}},
    year={2025},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/felixdittrich92/docTR-Synth-Generator}}
}
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
