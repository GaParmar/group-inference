import os, sys
import math
import torch
import numpy as np
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from my_utils.solvers import gurobi_solver


def get_next_size(curr_size, final_size, keep_ratio):
    """Calculate next size for progressive pruning during denoising.

    Args:
        curr_size: Current number of candidates
        final_size: Target final size
        keep_ratio: Fraction of candidates to keep at each step
    """
    if curr_size < final_size:
        raise ValueError("Current size is less than the final size!")
    elif curr_size == final_size:
        return curr_size
    else:
        next_size = math.ceil(curr_size * keep_ratio)
        return max(next_size, final_size)


@torch.no_grad()
def decode_latent(z, pipe, height, width):
    """Decode latent tensor to image using VAE decoder.

    Args:
        z: Latent tensor to decode
        pipe: Diffusion pipeline with VAE
        height: Image height
        width: Image width
    """
    z = pipe._unpack_latents(z, height, width, pipe.vae_scale_factor)
    z = (z / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    z = pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    return z


@torch.no_grad()
def run_group_inference(
    pipe,
    model_name=None,
    prompt=None,
    prompt_2=None,
    negative_prompt=None,
    negative_prompt_2=None,
    true_cfg_scale=1.0,
    height=None,
    width=None,
    num_inference_steps=28,
    sigmas=None,
    guidance_scale=3.5,
    l_generator=None,
    max_sequence_length=512,
    # group inference arguments
    unary_score_fn=None,
    binary_score_fn=None,
    starting_candidates=None,
    output_group_size=None,
    pruning_ratio=None,
    lambda_score=None,
    # control arguments
    control_image=None,
    # input image for flux-kontext
    input_image=None,
    skip_first_cfg=True,
):
    """Run group inference with progressive pruning for diverse, high-quality image generation.

    Args:
        pipe: Diffusion pipeline
        model_name: Model type (flux-schnell, flux-dev, flux-depth, flux-canny, flux-kontext)
        prompt: Text prompt for generation
        unary_score_fn: Function to compute image quality scores
        binary_score_fn: Function to compute pairwise diversity scores
        starting_candidates: Initial number of noise samples
        output_group_size: Final number of images to generate
        pruning_ratio: Fraction to prune at each denoising step
        lambda_score: Weight between quality and diversity terms
        control_image: Control image for depth/canny models
        input_image: Input image for flux-kontext editing
    """
    if l_generator is None:
        l_generator = [torch.Generator("cpu").manual_seed(42 + _seed) for _seed in range(starting_candidates)]

    # use the default height and width if not provided
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    pipe._guidance_scale = guidance_scale
    pipe._current_timestep = None
    pipe._interrupt = False
    pipe._joint_attention_kwargs = {}

    device = pipe._execution_device

    lora_scale = None
    has_neg_prompt = negative_prompt is not None
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    # 3. Encode prompts
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=prompt, prompt_2=prompt_2, prompt_embeds=None, pooled_prompt_embeds=None, device=device, max_sequence_length=max_sequence_length, lora_scale=lora_scale)

    if do_true_cfg:
        negative_prompt_embeds, negative_pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=negative_prompt, prompt_2=negative_prompt_2, prompt_embeds=None, pooled_prompt_embeds=None, device=device, max_sequence_length=max_sequence_length, lora_scale=lora_scale
        )

    # 4. Prepare latent variables
    if model_name in ["flux-depth", "flux-canny"]:
        # for control models, the pipe.transformer.config.in_channels is doubled
        num_channels_latents = pipe.transformer.config.in_channels // 8
    else:
        num_channels_latents = pipe.transformer.config.in_channels // 4

    # Handle different model types
    image_latents = None
    image_ids = None
    if model_name == "flux-kontext":
        processed_image = pipe.image_processor.preprocess(input_image, height=height, width=width)
        l_latents = []
        for _gen in l_generator:
            latents, img_latents, latent_ids, img_ids = pipe.prepare_latents(processed_image, 1, num_channels_latents, height, width, prompt_embeds.dtype, device, _gen)
            l_latents.append(latents)
        # Use the image_latents and image_ids from the first generator
        _, image_latents, latent_image_ids, image_ids = pipe.prepare_latents(processed_image, 1, num_channels_latents, height, width, prompt_embeds.dtype, device, l_generator[0])
        # Combine latent_ids with image_ids
        if image_ids is not None:
            latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0)
    else:
        # For other models (flux-schnell, flux-dev, flux-depth, flux-canny)
        l_latents = [pipe.prepare_latents(1, num_channels_latents, height, width, prompt_embeds.dtype, device, _gen)[0] for _gen in l_generator]
        _, latent_image_ids = pipe.prepare_latents(1, num_channels_latents, height, width, prompt_embeds.dtype, device, l_generator[0])

    # 4.5. Prepare control image if provided
    control_latents = None
    if model_name in ["flux-depth", "flux-canny"]:
        control_image_processed = pipe.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=pipe.vae.dtype,
        )
        if control_image_processed.ndim == 4:
            control_latents = pipe.vae.encode(control_image_processed).latents
            control_latents = (control_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            height_control_image, width_control_image = control_latents.shape[2:]
            control_latents = pipe._pack_latents(control_latents, 1, num_channels_latents, height_control_image, width_control_image)

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latent_image_ids.shape[0]
    mu = calculate_shift(image_seq_len, pipe.scheduler.config.get("base_image_seq_len", 256), pipe.scheduler.config.get("max_image_seq_len", 4096), pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.15))
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    _dtype = l_latents[0].dtype

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(1)
    else:
        guidance = None
    guidance_1 = torch.full([1], 1.0, device=device, dtype=torch.float32).expand(1)

    # 6. Denoising loop
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue
            if guidance is not None and skip_first_cfg and i == 0:
                curr_guidance = guidance_1
            else:
                curr_guidance = guidance

            pipe._current_timestep = t
            timestep = t.expand(1).to(_dtype)
            # ipdb.set_trace()
            next_latents = []
            x0_preds = []
            # do 1 denoising step
            for _latent in l_latents:
                # prepare input for transformer based on model type
                if model_name in ["flux-depth", "flux-canny"]:
                    # Control models: concatenate control latents along dim=2
                    latent_model_input = torch.cat([_latent, control_latents], dim=2)
                elif model_name == "flux-kontext":
                    # Kontext model: concatenate image latents along dim=1
                    latent_model_input = torch.cat([_latent, image_latents], dim=1)
                else:
                    # Standard models (flux-schnell, flux-dev): use latents as is
                    latent_model_input = _latent

                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=curr_guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # For flux-kontext, we need to slice the noise_pred to match the latents size
                if model_name == "flux-kontext":
                    noise_pred = noise_pred[:, : _latent.size(1)]

                if do_true_cfg:
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=curr_guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    if model_name == "flux-kontext":
                        neg_noise_pred = neg_noise_pred[:, : _latent.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                # compute the previous noisy sample x_t -> x_t-1
                _latent = pipe.scheduler.step(noise_pred, t, _latent, return_dict=False)[0]
                # the scheduler is not state-less, it maintains a step index that is incremented by one after each step,
                # so we need to decrease it here
                if hasattr(pipe.scheduler, "_step_index"):
                    pipe.scheduler._step_index -= 1

                if type(pipe.scheduler) == FlowMatchEulerDiscreteScheduler:
                    dt = 0.0 - pipe.scheduler.sigmas[i]
                    x0_pred = _latent + dt * noise_pred
                else:
                    raise NotImplementedError("Only Flow Scheduler is supported for now! For other schedulers, you need to manually implement the x0 prediction step.")

                x0_preds.append(x0_pred)
                next_latents.append(_latent)

            if hasattr(pipe.scheduler, "_step_index"):
                pipe.scheduler._step_index += 1

            # if the size of next_latents > output_group_size, prune the latents
            if len(next_latents) > output_group_size:
                next_size = get_next_size(len(next_latents), output_group_size, 1 - pruning_ratio)
                print(f"Pruning from {len(next_latents)} to {next_size}")
                # decode the latents to pixels with tiny-vae
                l_x0_decoded = [decode_latent(_latent, pipe, height, width) for _latent in x0_preds]
                # compute the unary and binary scores
                l_unary_scores = unary_score_fn(l_x0_decoded, target_caption=prompt)
                M_binary_scores = binary_score_fn(l_x0_decoded)  # upper triangular matrix
                # run with Quadratic Integer Programming sover
                selected_indices = gurobi_solver(l_unary_scores, M_binary_scores, next_size, lam=lambda_score)
                l_latents = [next_latents[_i] for _i in selected_indices]
            else:
                l_latents = next_latents

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None

    l_images = [pipe._unpack_latents(_latent, height, width, pipe.vae_scale_factor) for _latent in l_latents]
    l_images = [(latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor for latents in l_images]
    l_images = [pipe.vae.decode(_image, return_dict=False)[0] for _image in l_images]
    l_images_tensor = [image.clamp(-1, 1) for image in l_images]  # Keep tensor version for scoring
    l_images = [pipe.image_processor.postprocess(image, output_type="pil")[0] for image in l_images]

    # Compute and print final scores
    print(f"\n=== Final Scores for {len(l_images)} generated images ===")

    # Compute unary scores
    final_unary_scores = unary_score_fn(l_images_tensor, target_caption=prompt)
    print(f"Unary scores (quality): {final_unary_scores}")
    print(f"Mean unary score: {np.mean(final_unary_scores):.4f}")

    # Compute binary scores
    final_binary_scores = binary_score_fn(l_images_tensor)
    print(f"Binary score matrix (diversity):")
    print(final_binary_scores)

    print("=" * 50)

    pipe.maybe_free_model_hooks()
    return l_images
