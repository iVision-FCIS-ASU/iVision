import json
import torch

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

def list_to_tensor(x, device):
    return torch.tensor(x).to(device) if x is not None else None

def save_projections(path, vision_w, vision_b, text_w, text_b):
    data = {
        "vision_proj_weight": tensor_to_list(vision_w),
        "vision_proj_bias": tensor_to_list(vision_b) if vision_b is not None else None,
        "text_proj_weight": tensor_to_list(text_w),
        "text_proj_bias": tensor_to_list(text_b) if text_b is not None else None,
    }

    with open(path, "w") as f:
        json.dump(data, f)

def load_projections(path, device="cpu"):
    with open(path, "r") as f:
        data = json.load(f)

    vision_w = list_to_tensor(data["vision_proj_weight"], device)
    vision_b = list_to_tensor(data["vision_proj_bias"], device)

    text_w = list_to_tensor(data["text_proj_weight"], device)
    text_b = list_to_tensor(data["text_proj_bias"], device)

    return vision_w, vision_b, text_w, text_b
