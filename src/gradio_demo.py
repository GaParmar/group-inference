import gradio as gr
import copy
import torch
from diffusers import FluxPipeline, AutoencoderTiny, FluxControlPipeline, FluxKontextPipeline
from my_utils.group_inference import run_group_inference
from my_utils.scores import build_score_fn
from my_utils.default_values import apply_defaults
import argparse
import os

# Global variables to hold the loaded models and config
pipe = None
unary_score_fn = None
binary_score_fn = None
current_model_name = None
current_unary_term = None
current_binary_term = None
args = None


def load_models(model_name, unary_term, binary_term):
    """Load diffusion pipeline and scoring functions based on model configuration.
    
    Args:
        model_name: Model type (flux-schnell, flux-dev, flux-depth, flux-canny, flux-kontext)
        unary_term: Unary scoring function name for quality assessment
        binary_term: Binary scoring function name for diversity assessment
    """
    global pipe, unary_score_fn, binary_score_fn, current_model_name, current_unary_term, current_binary_term, args
    
    if pipe is None or current_model_name != model_name:
        current_model_name = model_name
        
        # Initialize global args object with defaults
        args = argparse.Namespace(model_name=model_name,
            prompt=None,
            starting_candidates=None,
            output_group_size=None,
            pruning_ratio=None,
            lambda_score=None,
            seed=None,
            unary_term=unary_term,
            binary_term=binary_term,
            guidance_scale=None,
            num_inference_steps=None,
            height=None,
            width=None,
        )
        args = apply_defaults(args)
        
        if model_name == "flux-schnell":
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
        elif model_name == "flux-dev":
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
        elif model_name == "flux-depth":
            pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
        elif model_name == "flux-canny":
            pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
        elif model_name == "flux-kontext":
            pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    # Use defaults if None provided
    if unary_term is None:
        unary_term = args.unary_term
    if binary_term is None:
        binary_term = args.binary_term
    
    # Reload score functions if they changed
    if unary_score_fn is None or current_unary_term != unary_term:
        current_unary_term = unary_term
        unary_score_fn, _ = build_score_fn(unary_term, device="cuda")
    
    if binary_score_fn is None or current_binary_term != binary_term:
        current_binary_term = binary_term
        binary_score_fn, _ = build_score_fn(binary_term, device="cuda")


def generate_images(model_name, prompt, starting_candidates, output_group_size, pruning_ratio, lambda_score, seed, unary_term, binary_term, control_image=None, input_image=None):
    """Generate images using group inference with progressive pruning.
    
    Args:
        model_name: Model type for generation
        prompt: Text prompt for image generation
        starting_candidates: Initial number of noise samples
        output_group_size: Final number of images to generate
        pruning_ratio: Fraction to prune at each denoising step
        lambda_score: Weight between quality and diversity terms
        seed: Random seed for generation
        unary_term: Quality scoring function name
        binary_term: Diversity scoring function name
        control_image: Control image for depth/canny models
        input_image: Input image for flux-kontext editing
    """
    load_models(model_name, unary_term, binary_term)
    
    # Use global args and override with user inputs
    global args
    args.prompt = prompt
    args.starting_candidates = starting_candidates
    args.output_group_size = output_group_size
    args.pruning_ratio = pruning_ratio
    args.lambda_score = lambda_score
    args.seed = seed
    args.unary_term = unary_term
    args.binary_term = binary_term
    
    # Create inference args
    inference_args = {
        "model_name": model_name,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": 256,
        "height": args.height,
        "width": args.width,
        "unary_score_fn": unary_score_fn,
        "binary_score_fn": binary_score_fn,
        "output_group_size": args.output_group_size,
        "pruning_ratio": args.pruning_ratio,
        "lambda_score": args.lambda_score,
    }
    
    # Add control image for depth and canny models
    if model_name in ["flux-depth", "flux-canny"] and control_image is not None:
        inference_args["control_image"] = control_image
    
    # Add input image for flux-kontext
    if model_name == "flux-kontext" and input_image is not None:
        inference_args["input_image"] = input_image

    # Group inference (larger starting candidates)
    inference_args["l_generator"] = [torch.Generator("cpu").manual_seed(args.seed+i) for i in range(args.starting_candidates)]
    inference_args["starting_candidates"] = args.starting_candidates
    inference_args["skip_first_cfg"] = True
    output_group = run_group_inference(pipe, **inference_args)    
    return output_group


def create_interface(model_name):
    """Create Gradio interface for interactive image generation.
    
    Args:
        model_name: Model type to create interface for
    """
    # Load models and initialize global args
    load_models(model_name, None, None)  # Use defaults from apply_defaults
    global args
    
    # Load custom CSS
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path, "r") as f:
        custom_css = f.read()

    # Always render in light mode ...
    js_func = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """

    # Create Gradio interface
    with gr.Blocks(css=custom_css, js=js_func, theme=gr.themes.Soft(), elem_id="main-container") as demo:
        
        # Title and description
        model_title_map = {
            "flux-schnell": "FLUX.1-Schnell",
            "flux-dev": "FLUX.1-Dev", 
            "flux-depth": "FLUX.1-Depth",
            "flux-canny": "FLUX.1-Canny",
            "flux-kontext": "FLUX.1-Kontext"
        }
        
        demo_type_map = {
            "flux-schnell": "Text-to-Image",
            "flux-dev": "Text-to-Image",
            "flux-depth": "Depth-to-Image", 
            "flux-canny": "Canny-to-Image",
            "flux-kontext": "Image Editing"
        }

        gr.HTML(
            f"""
            <div class="title_left">
            <h1>Scaling Group Inference for Diverse and High-Quality Generation</h1>
            <div class="author-container">
                <div class="grid-item cmu"><a href="https://gauravparmar.com/">Gaurav Parmar</a></div>
                <div class="grid-item snap"><a href="https://orpatashnik.github.io/">Or Patashnik</a></div>
                <div class="grid-item snap"><a href="https://scholar.google.com/citations?user=uD79u6oAAAAJ&hl=en">Daniil Ostashev</a></div>
                <div class="grid-item snap"><a href="https://wangkua1.github.io/">Kuan-Chieh (Jackson) Wang</a></div>
                <div class="grid-item snap"><a href="https://kfiraberman.github.io/">Kfir Aberman</a></div>
            </div>
            <div class="author-container">
                <div class="grid-item cmu"><a href="https://www.cs.cmu.edu/~srinivas/">Srinivasa Narasimhan</a></div>
                <div class="grid-item cmu"><a href="https://www.cs.cmu.edu/~junyanz/">Jun-Yan Zhu</a></div>
            </div>
            <br>
            <div class="affiliation-container">
                <div class="grid-item cmu"> <p>Carnegie Mellon University</p></div>
                <div class="grid-item snap"> <p>Snap Research</p></div>
            </div>
            
            <br>
            <h2>DEMO: {demo_type_map[model_name]} Group Inference with {model_title_map[model_name]}</h2>

            </div>
            """
        )

        with gr.Row(scale=1):
            with gr.Column(scale=1.0):
                # Update prompt placeholder and default based on model type
                if model_name == "flux-kontext":
                    prompt_placeholder = "Cat is playing outside in nature."
                    prompt_default = "Cat is playing outside in nature."
                else:
                    prompt_placeholder = "A photo of a dog"
                    prompt_default = "A photo of a dog"
                    
                prompt = gr.Textbox(label="Prompt", placeholder=prompt_placeholder, lines=4, value=prompt_default)
                
                # Show control image upload for depth and canny models, input image for flux-kontext
                control_image = None
                input_image = None
                if model_name == "flux-depth":
                    control_image = gr.Image(label="Depth Map", type="pil", sources=["upload"])
                elif model_name == "flux-canny":
                    control_image = gr.Image(label="Canny Edge Map", type="pil", sources=["upload"])
                elif model_name == "flux-kontext":
                    input_image = gr.Image(label="Input Image", type="pil", sources=["upload"])
            
            with gr.Column(scale=1.0):
                with gr.Row(elem_id="starting-candidates-row"):
                    gr.Text("Starting Candidates:", container=False, interactive=False, scale=5)
                    starting_candidates = gr.Number(value=args.starting_candidates, precision=0, container=False, show_label=False, scale=1)
                
                with gr.Row(elem_id="output-group-size-row"):
                    gr.Text("Output Group Size:", container=False, interactive=False, scale=5)
                    output_group_size = gr.Number(value=args.output_group_size, precision=0, container=False, show_label=False, scale=1)
                
            with gr.Column(scale=1.0):
                with gr.Accordion("Advanced Options", open=False, elem_id="advanced-options-accordion"):
                    with gr.Row():
                        gr.Text("Pruning Ratio:", container=False, interactive=False, elem_id="pruning-ratio-label", scale=3)
                        pruning_ratio = gr.Number(value=args.pruning_ratio, precision=2, container=False, show_label=False, scale=1)

                    with gr.Row():
                        gr.Text("Lambda:", container=False, interactive=False, elem_id="lambda-label", scale=5)
                        lambda_score = gr.Number(value=args.lambda_score, precision=1, container=False, show_label=False, scale=1)

                    with gr.Row():
                        gr.Text("Seed:", container=False, interactive=False, elem_id="seed-label", scale=5)
                        seed = gr.Number(value=42, precision=0, container=False, show_label=False, scale=1)
                    
                    with gr.Row():
                        gr.Text("Unary:", container=False, interactive=False, elem_id="unary-term-label", scale=2)
                        unary_term = gr.Dropdown(choices=["clip_text_img"], value=args.unary_term, container=False, show_label=False, scale=3)
                    
                    with gr.Row():
                        gr.Text("Binary:", container=False, interactive=False, elem_id="binary-term-label", scale=2)
                        binary_term = gr.Dropdown(choices=["diversity_dino", "diversity_clip", "dino_cls_pairwise"], value=args.binary_term, container=False, show_label=False, scale=3)
        
        with gr.Row(scale=1):
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Row(scale=1):
            output_gallery_group = gr.Gallery(label="Group Inference", show_label=True, elem_id="gallery", columns=4, height="auto")
        
        # Set up the generate button click event
        inputs = [gr.State(model_name), prompt, starting_candidates, output_group_size, pruning_ratio, lambda_score, seed, unary_term, binary_term]
        
        # Always include both control_image and input_image in inputs, using None for unused ones
        if model_name in ["flux-depth", "flux-canny"]:
            inputs.extend([control_image, gr.State(None)])
        elif model_name == "flux-kontext":
            inputs.extend([gr.State(None), input_image])
        else:
            inputs.extend([gr.State(None), gr.State(None)])
        
        generate_btn.click(
            fn=generate_images,
            inputs=inputs,
            outputs=[output_gallery_group]
        )
    
    return demo


def main():
    """Launch Gradio demo interface with model specified by environment variable."""
    model_name = os.getenv("MODEL_NAME", "flux-schnell")
    demo = create_interface(model_name)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
