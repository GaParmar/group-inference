import os, argparse, json
import torch
from diffusers import FluxPipeline, AutoencoderTiny, FluxControlPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
import numpy as np
from PIL import Image
from my_utils.group_inference import run_group_inference
from my_utils.scores import build_score_fn
from my_utils.default_values import apply_defaults


def build_parser(parser):
    """Build command line argument parser for image generation parameters.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    # options about the base model generation
    parser.add_argument("--model_name", type=str, default="flux-schnell", choices=["flux-schnell", "flux-dev", "flux-depth", "flux-canny", "flux-kontext"], help="name of the base model")
    parser.add_argument("--num_inference_steps", type=int, help="number of inference steps to use for generation")
    parser.add_argument("--guidance_scale", type=float, help="guidance scale to use for generation")
    parser.add_argument("--prompt", type=str, required=True, help="prompt to generate images for")
    parser.add_argument("--input_depth_map", type=str, help="depth map to use for generation")
    parser.add_argument("--input_canny_edge_map", type=str, help="canny edge map to use for generation")
    parser.add_argument("--input_image", type=str, help="input image to use for image editing with flux-kontext")

    parser.add_argument("--height", type=int, help="height of the generated image")
    parser.add_argument("--width", type=int, help="width of the generated image")

    # options about the group inference
    parser.add_argument("--unary_term", type=str, default="clip_text_img", choices=["clip_text_img", "aesthetics", "image_reward"], help="image quality term to use")
    parser.add_argument("--binary_term", type=str, default="diversity_dino", choices=["diversity_dino", "diversity_clip", "dino_cls_pairwise", "rgb_histogram"], help="image diversity term to use")
    parser.add_argument("--starting_candidates", type=int, help="number of initial noise samples to start with")
    parser.add_argument("--output_group_size", type=int, help="number of samples to generate for each group")
    parser.add_argument("--pruning_ratio", type=float, help="fraction of samples to prune at each denoising step")
    parser.add_argument("--lambda_score", type=float, help="relative weight between the unary (quality) and binary (diversity) terms")

    # options about the file i/o and logging
    parser.add_argument("--output_dir", type=str, help="directory to save the generated images")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random number generator")

    return parser


def main(args):
    """Main function to run image generation with group inference.

    Args:
        args: Parsed command line arguments containing generation parameters
    """
    # apply default values based on model_name
    args = apply_defaults(args)

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # store the args in a json file
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    if args.model_name == "flux-schnell":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_name == "flux-dev":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_name == "flux-depth":
        pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_name == "flux-canny":
        pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_name == "flux-kontext":
        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")

    # make the unary and binary score functions
    unary_score_fn, _ = build_score_fn(args.unary_term, device="cuda")
    binary_score_fn, _ = build_score_fn(args.binary_term, device="cuda")

    # load control image if provided
    control_image = None
    if args.input_depth_map is not None:
        control_image = Image.open(args.input_depth_map).convert("RGB")
    elif args.input_canny_edge_map is not None:
        control_image = Image.open(args.input_canny_edge_map).convert("RGB")

    # load input image for flux-kontext
    input_image = None
    if args.input_image is not None:
        input_image = Image.open(args.input_image).convert("RGB")

    # create args dict for the group inference function
    inference_args = {
        # base generation args
        "model_name": args.model_name,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": 256,
        "height": args.height,
        "width": args.width,
        "l_generator": [torch.Generator("cpu").manual_seed(args.seed + _i) for _i in range(args.starting_candidates)],
        # group inference args
        "unary_score_fn": unary_score_fn,
        "binary_score_fn": binary_score_fn,
        "starting_candidates": args.starting_candidates,
        "output_group_size": args.output_group_size,
        "pruning_ratio": args.pruning_ratio,
        "lambda_score": args.lambda_score,
    }

    # add control image for flux-depth and flux-canny
    if control_image is not None:
        inference_args["control_image"] = control_image

    # add input image for flux-kontext
    if input_image is not None:
        inference_args["input_image"] = input_image

    # result is a list of PIL
    output_group = run_group_inference(pipe, **inference_args)
    # save images individually
    for i, image in enumerate(output_group):
        image.save(os.path.join(args.output_dir, f"result_{i}.jpg"))
    _grid = Image.fromarray(np.hstack([np.array(image) for image in output_group]))
    _grid.save(os.path.join(args.output_dir, "result_grid.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = build_parser(parser).parse_args()
    main(args)
