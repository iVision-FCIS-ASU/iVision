# main.py
import torch
import timm
import time
import cv2
import torchvision.transforms as transforms
from PIL import Image
from utils.captioning import iVisionNarrator
from utils.mapping import INDOOR_CLASSES, OUTDOOR_MERGE_RULES

class MobileViT:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # models = timm.list_models('*mobilevit_xxs*')
        # print(models) 

        self.binary_mdl = self.load_mobilevit_weights('weights/best_mobilevit_merged.pth', num_classes=2)
        self.indoor_mdl = self.load_mobilevit_weights('weights/best_mobilevit_indoor.pth', num_classes=len(INDOOR_CLASSES))
        self.outdoor_mdl = self.load_mobilevit_weights('weights/best_mobilevit_outdoor.pth', num_classes=len(OUTDOOR_MERGE_RULES))

        self.narrator = iVisionNarrator(self.device)

    def load_mobilevit_weights(self, model_path, num_classes):
        print(f"Loading weights for: {model_path}")
        model = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        raw_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(raw_img)
        # raw_img = Image.open(image).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = transform(raw_img).unsqueeze(0).to(self.device)
        return raw_img, tensor_img

    def get_scene_type_binary(self, image):
        _, tensor_img = self.preprocess(image)
        is_outdoor = torch.argmax(self.binary_mdl(tensor_img), dim=1).item()
        return "outdoor" if is_outdoor else "indoor"

    def run_scene_narration(self, image):
        raw_img, tensor_img = self.preprocess(image)

        # 1. Timing MobileViT Phase
        print("-----PREDICTING SCENE TYPE-----")
        start_mobilevit = time.time()
        with torch.no_grad():
            is_outdoor = torch.argmax(self.binary_mdl(tensor_img), dim=1).item()
            expert = self.outdoor_mdl if is_outdoor == 1 else self.indoor_mdl
            scene_idx = torch.argmax(expert(tensor_img), dim=1).item()
        mobilevit_time = time.time() - start_mobilevit
            
        scene_label = list(OUTDOOR_MERGE_RULES.keys())[scene_idx] if is_outdoor == 1 else list(INDOOR_CLASSES.keys())[scene_idx]

        # 2. Timing Narration & CLIP Phase
        print(f"Outdoor:{is_outdoor}\nScene Label: {scene_label}")
        print("-----GETTING NARRATION-----")
        caption, clip_score, blip_time, clip_eval_time = self.narrator.narrate(raw_img, scene_label)

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
        model = MobileViT()
        final_output = model.run_scene_narration(test_img)
        print(f"Final Narration: {final_output}")
    except Exception as e:
        print(f"Critical Error: {e}")