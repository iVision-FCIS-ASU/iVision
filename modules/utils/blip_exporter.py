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

def blip_compare_vision_models(image: torch.Tensor | None = None):
    torch_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, local_files_only=True).eval().vision_model
    onnx_model = ort.InferenceSession("weights/blip_model/blip_vision.onnx")

    if image is None:
        image = torch.randn(1, 3, 384, 384)

    with torch.inference_mode():
        torch_out_vision = torch_model(image).last_hidden_state.cpu().numpy()
    
    onnx_out_vision = onnx_model.run(
        None, 
        {"pixel_values": image.numpy()}
    )[0]

    diff = np.abs(torch_out_vision - onnx_out_vision)
    print(f"Torch ({torch_out_vision.dtype}): {torch_out_vision.shape}")
    print(f"ONNX  ({onnx_out_vision.dtype}): {onnx_out_vision.shape}")
    print(f"Vision Model (Exact): {np.allclose(torch_out_vision, onnx_out_vision, atol=1e-4, rtol=1e-4)}")
    print(f"Max diff:  {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")
    print(f"99%:   {np.percentile(diff, 99)}")
    print(f"99.9%: {np.percentile(diff, 99.9)}")

def blip_export_text_model():
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, local_files_only=True).eval().text_decoder

    dummy_input_ids = torch.ones(1, 5, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 5, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(1, 577, 768)
    # dummy_encoder_attention_mask = torch.ones(1, 577, dtype=torch.long)

    torch.onnx.export(
        model,
        args=(),
        kwargs={
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "encoder_hidden_states": dummy_encoder_hidden_states,
            # "encoder_attention_mask": dummy_encoder_attention_mask,
            "return_dict": False,
            "use_cache": False
        },
        f="weights/blip_model/blip_text.onnx",
        input_names=[
            "input_ids",
            "attention_mask",
            "encoder_hidden_states",
            "encoder_attention_mask"
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "encoder_hidden_states": {0: "batch"},
            # "encoder_attention_mask": {0: "batch"},
            "logits": {0: "batch", 1: "seq"}
        },
        opset_version=18,
        dynamo=False
    )

def blip_compare_text_models(
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
):
    torch_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, cache_dir=cache_dir, local_files_only=True).eval().text_decoder
    onnx_model = ort.InferenceSession("weights/blip_model/blip_text.onnx")

    for inp in onnx_model.get_inputs():
        print(inp.name)

    if input_ids is None:
        input_ids = torch.randint(0, 30521, (1, 5))
        attention_mask = torch.ones_like(input_ids)
    if encoder_hidden_states is None:
        encoder_hidden_states = torch.randn(1, 577, 768)
        encoder_attention_mask = torch.ones(1, 577, dtype=torch.long)

    with torch.inference_mode():
        torch_out = torch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        ).logits.cpu().numpy()

    onnx_out = onnx_model.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "encoder_hidden_states": encoder_hidden_states.numpy(),
            # "encoder_attention_mask": encoder_attention_mask.numpy()
        }
    )[0]

    diff = np.abs(torch_out - onnx_out)
    print(f"Torch ({torch_out.dtype}): {torch_out.shape}")
    print(f"ONNX  ({onnx_out.dtype}): {onnx_out.shape}")
    print(f"Text Model (Exact): {np.allclose(torch_out, onnx_out, atol=1e-4, rtol=1e-4)}")
    print(f"Max diff:  {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")
    print(f"99%:   {np.percentile(diff, 99)}")
    print(f"99.9%: {np.percentile(diff, 99.9)}")

def blip_export_models():
    # blip_export_vision_model()
    # blip_compare_vision_models()

    blip_export_text_model()
    blip_compare_text_models()
