import torch
import requests
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

def download_file(path: str, url: str):
    print(f"Downloading file: {path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    chunk_size = 1048576
    total_size = int(response.headers.get("content-length", 0))
    
    with open(path, "wb") as file, tqdm(
        desc=path, 
        total=total_size,
        ncols=100,
        unit="B", 
        unit_scale=True, 
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            size = file.write(chunk)
            bar.update(size)
    
    print(f"Download done!")

def download_models():
    print("\n-----Downloading Models-----\n")

    Path("weights").mkdir(parents=True, exist_ok=True)

    if not Path("weights/yolo26n.onnx").is_file():
        model_detect = YOLO("weights/yolo26n.pt", task="detect")
        model_detect.export(format="onnx", dynamic=True)
    
    if not Path("weights/yolo26n-seg.onnx").is_file():
        model_segment = YOLO("weights/yolo26n-seg.pt", task="segment")
        model_segment.export(format="onnx", dynamic=True)
    
    if not Path("weights/midas_v21_small_256.pt").is_file():
        download_file("weights/midas_v21_small_256.pt", "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt")

    if not Path("weights/dpt_swin2_tiny_256.pt").is_file():
        download_file("weights/dpt_swin2_tiny_256.pt", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt")

    huggingface_file_names = [
        "midas_v21_small_256.tflite",
        "depth_anything_v2_224.tflite",
        "complexity_estimation_v1.tflite",
        "best_mobilevit_merged.tflite",
        "best_mobilevit_indoor.tflite",
        "best_mobilevit_outdoor.tflite"
    ]

    for file_name in huggingface_file_names:
        if not Path(f"weights/{file_name}").is_file():
            download_file(f"weights/{file_name}", f"https://huggingface.co/Devil-Assassin/wandering-mode/resolve/main/{file_name}?download=true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True)
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip").to(device)
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_fast=True)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip").to(device)

    print("\n-----All Downloads Completed-----\n")
