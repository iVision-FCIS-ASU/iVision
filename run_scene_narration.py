# main.py
import torch
import timm
import time
from PIL import Image
import torchvision.transforms as transforms
from utils.captioning import iVisionNarrator
from utils.mapping import INDOOR_CLASSES, OUTDOOR_MERGE_RULES

# Auto-detect CUDA for best performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mobilevit_weights(model_path, num_classes):
    """
    Loads weights into mobilevit_xxs architecture.
    """
    print(f"Loading weights for: {model_path}")
    model = timm.create_model('mobilevit_xxs.cvnets_in1k', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# --- Initialization Phase ---
print("Initializing iVision System...")
binary_mdl = load_mobilevit_weights('models/best_mobilevit_merged.pth', num_classes=2)
indoor_mdl = load_mobilevit_weights('models/best_mobilevit_indoor.pth', num_classes=len(INDOOR_CLASSES))
outdoor_mdl = load_mobilevit_weights('models/best_mobilevit_outdoor.pth', num_classes=len(OUTDOOR_MERGE_RULES))
narrator = iVisionNarrator(device)

def run_scene_narration(img_path):
    # Prepare Image
    raw_img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(raw_img).unsqueeze(0).to(device)

    # 1. Timing MobileViT Phase
    start_mobilevit = time.time()
    with torch.no_grad():
        is_outdoor = torch.argmax(binary_mdl(tensor_img), dim=1).item()
        expert = outdoor_mdl if is_outdoor == 1 else indoor_mdl
        scene_idx = torch.argmax(expert(tensor_img), dim=1).item()
    mobilevit_time = time.time() - start_mobilevit
        
    scene_label = list(OUTDOOR_MERGE_RULES.keys())[scene_idx] if is_outdoor == 1 else list(INDOOR_CLASSES.keys())[scene_idx]

    # 2. Timing Narration & CLIP Phase
    caption, clip_score, blip_time, clip_eval_time = narrator.narrate(raw_img, scene_label)

    # 3. Final Performance Report
    print(f"\n{'='*30}")
    print(f" PERFORMANCE REPORT")
    print(f"{'='*30}")
    print(f"1. MobileViT Expert  : {mobilevit_time:.3f}s")
    print(f"2. BLIP Generation   : {blip_time:.3f}s")
    print(f"3. CLIP Ranking      : {clip_eval_time:.3f}s")
    print(f"4. CLIPScore         : {clip_score:.4f}")
    print(f"Total Inference Time : {(mobilevit_time + blip_time + clip_eval_time):.3f}s")
    print(f"{'='*30}")
    
    return f"You are in a {scene_label.replace('_', ' ')}. {caption}"

if __name__ == "__main__":
    # Test path - Use 'r' prefix for windows paths
    test_img = r"C:\Users\lenovo\Downloads\WhatsApp Image 2026-01-25 at 1.13.23 PM.jpeg"
    try:
        print("\n--- System Processing ---")
        final_output = run_scene_narration(test_img)
        print(f"Final Narration: {final_output}")
    except Exception as e:
        print(f"Critical Error: {e}")