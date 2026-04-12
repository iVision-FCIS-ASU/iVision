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
