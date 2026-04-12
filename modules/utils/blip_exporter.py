import json
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipVisionModel, BlipTextLMHeadModel


blip_model_name = "Salesforce/blip-image-captioning-base"
cache_dir = "weights/blip"


def blip_export_tokenizer():        
    blip_processor = BlipProcessor.from_pretrained(blip_model_name, cache_dir=cache_dir, use_fast=True, local_files_only=True)
    blip_processor.tokenizer.save_pretrained("weights/blip_tokenizer")


def blip_export_vision_model():
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, local_files_only=True).eval().vision_model
    dummy_image = torch.randn(1, 3, 384, 384)

    torch.onnx.export(
        blip_model,
        dummy_image,
        "weights/blip_model/blip_vision.onnx",
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "last_hidden_state": {0: "batch"}
        },
        opset_version=18
    )

def blip_compare_vision_models():
    torch_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, local_files_only=True).eval().vision_model
    onnx_model = ort.InferenceSession("weights/blip_model/blip_vision.onnx")

    dummy_image = torch.randn(1, 3, 384, 384)

    with torch.inference_mode():
        torch_out_vision = torch_model(dummy_image).last_hidden_state.cpu().numpy()
    
    onnx_out_vision = onnx_model.run(
        None, 
        {"pixel_values": dummy_image.numpy()}
    )[0]

    diff = np.abs(torch_out_vision - onnx_out_vision)
    print(f"Torch ({torch_out_vision.dtype}): {torch_out_vision.shape}")
    print(f"ONNX  ({onnx_out_vision.dtype}): {onnx_out_vision.shape}")
    print(f"Vision Model (Exact): {np.allclose(torch_out_vision, onnx_out_vision, atol=1e-4, rtol=1e-4)}")
    print(f"Max diff:  {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")
    print(f"99%:   {np.percentile(diff, 99)}")
    print(f"99.9%: {np.percentile(diff, 99.9)}")

def blip_export_models():
    blip_export_vision_model()

    blip_compare_vision_models()


