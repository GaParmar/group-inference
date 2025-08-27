DEFAULT_VALUES = {
    "flux-schnell": {
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "starting_candidates": 64,
        "output_group_size": 4,
        "pruning_ratio": 0.9,
        "lambda_score": 1.5,
        "output_dir": "outputs/flux-schnell",
        "height": 768,
        "width": 768,
        "unary_term": "clip_text_img",
        "binary_term": "diversity_dino",
    },
    "flux-dev": {
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "starting_candidates": 128,
        "output_group_size": 4,
        "pruning_ratio": 0.5,
        "lambda_score": 1.5,
        "output_dir": "outputs/flux-dev",
        "height": 768,
        "width": 768,
        "unary_term": "clip_text_img",
        "binary_term": "diversity_dino",
    },
    "flux-depth": {
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "starting_candidates": 128,
        "output_group_size": 4,
        "pruning_ratio": 0.5,
        "lambda_score": 1.5,
        "output_dir": "outputs/flux-depth",
        "height": 768,
        "width": 768,
        "unary_term": "clip_text_img",
        "binary_term": "diversity_dino",
    },
    "flux-canny": {
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "starting_candidates": 128,
        "output_group_size": 4,
        "pruning_ratio": 0.5,
        "lambda_score": 1.5,
        "output_dir": "outputs/flux-canny",
        "height": 768,
        "width": 768,
        "unary_term": "clip_text_img",
        "binary_term": "diversity_dino",
    },
    "flux-kontext": {
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "starting_candidates": 128,
        "output_group_size": 4,
        "pruning_ratio": 0.5,
        "lambda_score": 1.0,
        "output_dir": "outputs/flux-kontext",
        "height": 1024,
        "width": 1024,
        "unary_term": "clip_text_img",
        "binary_term": "diversity_dino",
    },
}


def apply_defaults(args):
    model_name = args.model_name

    if model_name not in DEFAULT_VALUES:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(DEFAULT_VALUES.keys())}")

    defaults = DEFAULT_VALUES[model_name]

    for param_name, default_value in defaults.items():
        if hasattr(args, param_name) and getattr(args, param_name) is None:
            setattr(args, param_name, default_value)

    return args
