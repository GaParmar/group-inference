import functools
import torch
import numpy as np
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel, AutoModel
from transformers.models.clip.modeling_clip import _get_vector_norm


def validate_tensor_list(tensor_list, expected_type=torch.Tensor, min_dims=None, value_range=None, tolerance=0.1):
    """
    Validates a list of tensors with specified requirements.

    Args:
        tensor_list: List to validate
        expected_type: Expected type of each element (default: torch.Tensor)
        min_dims: Minimum number of dimensions each tensor should have
        value_range: Tuple of (min_val, max_val) for tensor values
        tolerance: Tolerance for value range checking (default: 0.1)
    """
    if not isinstance(tensor_list, list):
        raise TypeError(f"Input must be a list, got {type(tensor_list)}")

    if len(tensor_list) == 0:
        raise ValueError("Input list cannot be empty")

    for i, item in enumerate(tensor_list):
        if not isinstance(item, expected_type):
            raise TypeError(f"List element [{i}] must be {expected_type}, got {type(item)}")

        if min_dims is not None and len(item.shape) < min_dims:
            raise ValueError(f"List element [{i}] must have at least {min_dims} dimensions, got shape {item.shape}")

        if value_range is not None:
            min_val, max_val = value_range
            item_min, item_max = item.min().item(), item.max().item()
            if item_min < (min_val - tolerance) or item_max > (max_val + tolerance):
                raise ValueError(f"List element [{i}] values must be in range [{min_val}, {max_val}], got range [{item_min:.3f}, {item_max:.3f}]")


def build_score_fn(name, device="cuda"):
    """Build scoring functions for image quality and diversity assessment.

    Args:
        name: Score function name (clip_text_img, diversity_dino, dino_cls_pairwise, diversity_clip)
        device: Device to load models on (default: cuda)
    """
    d_score_nets = {}

    if name == "clip_text_img":
        m_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        prep_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        score_fn = functools.partial(unary_clip_text_img_t, device=device, m_clip=m_clip, preprocess_clip=prep_clip)
        d_score_nets["m_clip"] = m_clip
        d_score_nets["prep_clip"] = prep_clip

    elif name == "diversity_dino":
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        score_fn = functools.partial(binary_dino_pairwise_t, device=device, dino_model=dino_model)
        d_score_nets["dino_model"] = dino_model

    elif name == "dino_cls_pairwise":
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        score_fn = functools.partial(binary_dino_cls_pairwise_t, device=device, dino_model=dino_model)
        d_score_nets["dino_model"] = dino_model

    elif name == "diversity_clip":
        m_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        prep_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        score_fn = functools.partial(binary_clip_pairwise_t, device=device, m_clip=m_clip, preprocess_clip=prep_clip)
        d_score_nets["m_clip"] = m_clip
        d_score_nets["prep_clip"] = prep_clip

    else:
        raise ValueError(f"Invalid score function name: {name}")

    return score_fn, d_score_nets


@torch.no_grad()
def unary_clip_text_img_t(l_images, device, m_clip, preprocess_clip, target_caption, d_cache=None):
    """Compute CLIP text-image similarity scores for a list of images.

    Args:
        l_images: List of image tensors in range [-1, 1]
        device: Device for computation
        m_clip: CLIP model
        preprocess_clip: CLIP processor
        target_caption: Text prompt for similarity comparison
        d_cache: Optional cached text embeddings
    """
    # validate input images, l_images should be a list of torch tensors with range [-1, 1]
    validate_tensor_list(l_images, expected_type=torch.Tensor, min_dims=3, value_range=(-1, 1))

    _img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    _img_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    b_images = torch.cat(l_images, dim=0)
    b_images = F.interpolate(b_images, size=(224, 224), mode="bilinear", align_corners=False)
    # re-normalize with clip mean and std
    b_images = b_images * 0.5 + 0.5
    b_images = (b_images - _img_mean) / _img_std

    if d_cache is None:
        text_encoding = preprocess_clip.tokenizer(target_caption, return_tensors="pt", padding=True).to(device)
        output = m_clip(pixel_values=b_images, **text_encoding).logits_per_image / m_clip.logit_scale.exp()
        _score = output.view(-1).cpu().numpy()
    else:
        # compute with cached text embeddings
        vision_outputs = m_clip.vision_model(
            pixel_values=b_images,
            output_attentions=False,
            output_hidden_states=False,
            interpolate_pos_encoding=False
        )
        image_embeds = m_clip.visual_projection(vision_outputs.pooler_output)
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = d_cache["text_embeds"]
        _score = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)).t().view(-1).cpu().numpy()

    return _score


@torch.no_grad()
def binary_dino_pairwise_t(l_images, device, dino_model):
    """Compute pairwise diversity scores using DINO patch features.

    Args:
        l_images: List of image tensors in range [-1, 1]
        device: Device for computation
        dino_model: DINO model for feature extraction
    """
    # validate input images, l_images should be a list of torch tensors with range [-1, 1]
    validate_tensor_list(l_images, expected_type=torch.Tensor, min_dims=3, value_range=(-1, 1))

    b_images = torch.cat(l_images, dim=0)
    _img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    _img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    b_images = F.interpolate(b_images, size=(256, 256), mode="bilinear", align_corners=False)
    b_images = b_images * 0.5 + 0.5
    b_images = (b_images - _img_mean) / _img_std
    all_features = dino_model(pixel_values=b_images).last_hidden_state[:, 1:, :].cpu()  # B, 324, 768

    N = len(l_images)
    score_matrix = np.zeros((N, N))
    for i in range(N):
        f1 = all_features[i]
        for j in range(i + 1, N):
            f2 = all_features[j]
            cos_sim = (1 - F.cosine_similarity(f1, f2, dim=1)).mean().item()
            score_matrix[i, j] = cos_sim
    return score_matrix


@torch.no_grad()
def binary_dino_cls_pairwise_t(l_images, device, dino_model):
    """Compute pairwise diversity scores using DINO CLS token features.

    Args:
        l_images: List of image tensors in range [-1, 1]
        device: Device for computation
        dino_model: DINO model for feature extraction
    """
    # validate input images, l_images should be a list of torch tensors with range [-1, 1]
    validate_tensor_list(l_images, expected_type=torch.Tensor, min_dims=3, value_range=(-1, 1))

    b_images = torch.cat(l_images, dim=0)
    _img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    _img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    b_images = F.interpolate(b_images, size=(256, 256), mode="bilinear", align_corners=False)
    b_images = b_images * 0.5 + 0.5
    b_images = (b_images - _img_mean) / _img_std
    features = dino_model(pixel_values=b_images).last_hidden_state[:, 0, :]  # B x 768
    features_norm = F.normalize(features, p=2, dim=1)  # normalize the features
    sim_matrix = 1 - features_norm @ features_norm.T  # computer cosine distance
    score_matrix = sim_matrix.triu(diagonal=1).cpu().numpy()  # take the upper triangle and set digonal elements as 0
    return score_matrix


@torch.no_grad()
def binary_clip_pairwise_t(l_images, device, m_clip, preprocess_clip):
    """Compute pairwise diversity scores using CLIP image embeddings.

    Args:
        l_images: List of image tensors in range [-1, 1]
        device: Device for computation
        m_clip: CLIP model
        preprocess_clip: CLIP processor
    """
    # validate input images, l_images should be a list of torch tensors with range [-1, 1]
    validate_tensor_list(l_images, expected_type=torch.Tensor, min_dims=3, value_range=(-1, 1))

    _img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    _img_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    b_images = torch.cat(l_images, dim=0)
    b_images = F.interpolate(b_images, size=(224, 224), mode="bilinear", align_corners=False)
    # re-normalize with clip mean and std
    b_images = b_images * 0.5 + 0.5
    b_images = (b_images - _img_mean) / _img_std

    vision_outputs = m_clip.vision_model(
        pixel_values=b_images,
        output_attentions=False,
        output_hidden_states=False,
        interpolate_pos_encoding=False,
    )
    image_embeds = m_clip.visual_projection(vision_outputs.pooler_output)
    image_embeds = image_embeds / _get_vector_norm(image_embeds)

    N = len(l_images)
    score_matrix = np.zeros((N, N))
    for i in range(N):
        f1 = image_embeds[i]
        for j in range(i + 1, N):
            f2 = image_embeds[j]
            cos_sim = (1 - torch.dot(f1, f2)).item()
            score_matrix[i, j] = cos_sim
    return score_matrix
