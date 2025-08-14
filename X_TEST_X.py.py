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
