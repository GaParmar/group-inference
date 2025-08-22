# Command-Line Arguments

This document describes all the available command-line arguments for the text-to-image group inference script.

## Arguments

- `--prompt`: The prompt to generate images for.
- `--unary_term`: The image quality term to use. (options are: `text_image_similarity_clip` (default), `aesthetics`, `image_reward`). See [this](TODO) for defining your own unary scores.
- `--binary_term`: The image diversity term to use. (options are: `diversity_dino` (default), `diversity_clip`). See [this](TODO) for defining your own binary scores.
- `--starting_candidates`: The number of initial noise samples to start with.
- `--output_group_size`: The number of samples to generate for each group.
- `--pruning_ratio`: The fraction of samples to prune at each denoising step.
- `--lambda`: The relative weight between the unary (quality) and binary (diversity) terms. A higher value will prioritize diversity over quality.
- `--model_name`: The model to use for generation.
- `--output_dir`: The directory to save the generated images.
- `--num_inference_steps`: The number of inference steps to use for generation.
- `--guidance_scale`: The guidance scale to use for generation. 