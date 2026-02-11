import cv2
import torch
import requests
import line_profiler
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

@line_profiler.profile
def run_blip():
    print("-----MODELS LOADING-----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", local_files_only=True).to(device)
    model.eval()
    print("-----MODELS LOADED-----")

    print("-----RETRIEVING IMAGE-----")
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://thumbs.dreamstime.com/b/cute-cat-sleeping-street-car-random-58655731.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture frame!")
        return
    cap.release()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print("-----RETRIEVED IMAGE-----")

    print("-----PROCESSING IMAGE-----")
    inputs = processor(images=image, return_tensors="pt")
    print("-----PROCESSED IMAGE-----")

    print("-----GENERATING CAPTION-----")
    with torch.inference_mode():
        outputs = model.generate(**inputs)
    print("-----GENERATED CAPTION-----")
    print(processor.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_blip()
