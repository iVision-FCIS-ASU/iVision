import json
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel


clip_model_name = "openai/clip-vit-base-patch16"
cache_dir = "weights/blip"

def clip_export_tokenizer():
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir, use_fast=True, local_files_only=True)
    clip_processor.tokenizer.save_pretrained("weights/clip_tokenizer")


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().numpy().tolist()

def list_to_numpy(x: list | None) -> npt.NDArray | None:
    return np.array(x) if x is not None else None


def clip_save_projections(
        path: str, 
        vision_w: torch.Tensor, 
        vision_b: torch.Tensor, 
        text_w: torch.Tensor, 
        text_b: torch.Tensor
    ) -> None:
    data = {
        "vision_proj_weight": tensor_to_list(vision_w),
        "vision_proj_bias": tensor_to_list(vision_b) if vision_b is not None else None,
        "text_proj_weight": tensor_to_list(text_w),
        "text_proj_bias": tensor_to_list(text_b) if text_b is not None else None,
    }

    with open(path, "w") as f:
        json.dump(data, f)

def clip_export_projections():
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True)
    Path("weights/clip_projection").mkdir(parents=True, exist_ok=True)
    clip_save_projections(
        "weights/clip_projection/clip_projections.json",
        clip_model.visual_projection.weight,
        clip_model.visual_projection.bias,
        clip_model.text_projection.weight,
        clip_model.text_projection.bias
    )

def clip_load_projections(
        path: str
    ) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:
    with open(path, "r") as f:
        data = json.load(f)

    vision_w = list_to_numpy(data["vision_proj_weight"])
    vision_b = list_to_numpy(data["vision_proj_bias"])

    text_w = list_to_numpy(data["text_proj_weight"])
    text_b = list_to_numpy(data["text_proj_bias"])

    return vision_w, vision_b, text_w, text_b


def clip_export_vision_model():
    class CLIPVisionWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).eval()

        def forward(self, pixel_values):
            vision_outputs = self.clip_vision_model(pixel_values=pixel_values)
            return vision_outputs.pooler_output

    clip_vision_wrapper = CLIPVisionWrapper()        
    dummy_image = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        clip_vision_wrapper,
        (dummy_image,),
        "weights/clip_model/clip_vision.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "image_embeds": {0: "batch"}
        },
        opset_version=18
    )

def clip_export_text_model():
    class CLIPTextWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_text_model = CLIPTextModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).eval()

        def forward(self, input_ids, attention_mask):
            text_outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
            return text_outputs.pooler_output
    
    clip_text_wrapper = CLIPTextWrapper()
    dummy_input_ids = torch.ones((2, 77), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 77), dtype=torch.long)
    
    torch.onnx.export(
        clip_text_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        "weights/clip_model/clip_text.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_embeds": {0: "batch"}
        },
        opset_version=18
    )

def clip_compare_vision_models():
    torch_model = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).eval()
    onnx_model = ort.InferenceSession("weights/clip_model/clip_vision.onnx")

    dummy_image = torch.randn(8, 3, 224, 224)
    
    with torch.inference_mode():
        torch_out_vision = torch_model(dummy_image).pooler_output

    onnx_out_vision = onnx_model.run(
        None,
        {"pixel_values": dummy_image.cpu().numpy()}
    )[0]

    print(f"Vision Model: {np.allclose(torch_out_vision.cpu().numpy(), onnx_out_vision, atol=1e-4, rtol=1e-4)}")

def clip_compare_text_models():
    torch_model = CLIPTextModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).eval()
    onnx_model = ort.InferenceSession("weights/clip_model/clip_text.onnx")

    dummy_input_ids = torch.ones((8, 77), dtype=torch.long)
    dummy_attention_mask = torch.ones((8, 77), dtype=torch.long)

    with torch.inference_mode():
        torch_out_text = torch_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        ).pooler_output
    
    onnx_out_text = onnx_model.run(
        None,
        {
            "input_ids": dummy_input_ids.cpu().numpy(),
            "attention_mask": dummy_attention_mask.cpu().numpy()
        }
    )[0]

    print(f"Text Model: {np.allclose(torch_out_text.cpu().numpy(), onnx_out_text, atol=1e-4, rtol=1e-4)}")

@torch.inference_mode()
def clip_export_models():
    Path("weights/clip_model").mkdir(parents=True, exist_ok=True)

    clip_export_vision_model()
    clip_export_text_model()

    clip_compare_vision_models()
    clip_compare_text_models()
