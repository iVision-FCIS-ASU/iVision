import torch
import requests
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
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
        model_detect.export(format="onnx")
    
    if not Path("weights/yolo26n-seg.onnx").is_file():
        model_detect = YOLO("weights/yolo26n-seg.pt", task="segment")
        model_detect.export(format="onnx")
    
    if not Path("weights/midas_v21_small_256.pt").is_file():
        download_file("weights/midas_v21_small_256.pt", "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt")

    if not Path("weights/dpt_swin2_tiny_256.pt").is_file():
        download_file("weights/dpt_swin2_tiny_256.pt", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt")
    
    if not Path("weights/depth_anything_v2_vits.onnx").is_file():
        if not Path("weights/depth_anything_v2_vits.pth").is_file():
            download_file("weights/depth_anything_v2_vits.pth", "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true")
        else:
            print("weights/depth_anything_v2_vits.pth already exists!")
        
        print("Exporting to ONNX format...")
        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48,96,192,384])
        model.load_state_dict(torch.load("weights/depth_anything_v2_vits.pth", map_location="cpu"))
        model.eval()
        dummy = torch.randn(1, 3, 518, 518)

        with torch.no_grad():
            torch.onnx.export(
                model, dummy, "weights/depth_anything_v2_vits.onnx",
                input_names=['input'], output_names=['output'],
                opset_version=16,
                do_constant_folding=True,
                dynamic_axes={
                    'input': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                },
                export_params=True,
                keep_initializers_as_inputs=False,
            )
        print("Exported to weights/depth_anything_v2_vits.onnx")
    
    if not Path("weights/complexity_estimation_v1.keras").is_file():
        download_file("weights/complexity_estimation_v1.keras", "https://huggingface.co/Devil-Assassin/wandering-mode/resolve/main/complexity_estimation_v1.keras?download=true")

    if not Path("weights/best_mobilevit_merged.pth").is_file():
        download_file("weights/best_mobilevit_merged.pth", "https://huggingface.co/Devil-Assassin/wandering-mode/resolve/main/best_mobilevit_merged.pth?download=true")
    
    if not Path("weights/best_mobilevit_indoor.pth").is_file():
        download_file("weights/best_mobilevit_indoor.pth", "https://huggingface.co/Devil-Assassin/wandering-mode/resolve/main/best_mobilevit_indoor.pth?download=true")
    
    if not Path("weights/best_mobilevit_outdoor.pth").is_file():
        download_file("weights/best_mobilevit_outdoor.pth", "https://huggingface.co/Devil-Assassin/wandering-mode/resolve/main/best_mobilevit_outdoor.pth?download=true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True)
    # BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_safetensors=False).to(device)
    # CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_fast=True)
    # CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_safetensors=False).to(device)

    print("\n-----All Downloads Completed-----\n")
