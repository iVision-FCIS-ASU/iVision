# utils/captioning.py
import torch
import numpy as np
import time
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

class iVisionNarrator:
    def __init__(self, device):
        self.device = device
        print("Loading BLIP and CLIP models...")

        # use_fast=True to speed up preprocessing
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        # self.blip_processor.tokenizer.padding_side = "left"
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", local_files_only=True).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(device)
        self.blip_model.eval()
        self.clip_model.eval()

    @torch.no_grad()
    def generate_captions(self, image, scene_label):
        # Restore Multi-Prompt Strategy from your original notebook
        clean_label = scene_label.replace('_', ' ')
        prompts = [
            f"a detailed scene narration describing the {clean_label} environment",
            f"inside this {clean_label}, there is",
            f"a photo showing",
            f"" 
        ]

        # High Accuracy Parameters (num_beams=10)
        gen_params = dict(
            max_length=50, 
            min_length=20,
            num_beams=8, 
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        all_captions = []
        for p in prompts:
            inputs = self.blip_processor(image, text=p, return_tensors="pt").to(self.device)
            # Generate 2 options per prompt for CLIP to evaluate
            outputs = self.blip_model.generate(**inputs, **gen_params, num_return_sequences=2)
            all_captions += [self.blip_processor.decode(o, skip_special_tokens=True) for o in outputs]

        # inputs = self.blip_processor(image, text=prompts, return_tensors="pt", padding=True).to(self.device)
        # outputs = self.blip_model.generate(**inputs, **gen_params, num_return_sequences=2)
        # all_captions = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)

        # images = [image] * len(prompts)
        # inputs = {k: v.to(self.device) for k, v in self.blip_processor(images, text=prompts, return_tensors="pt", padding=True).items()}
        # outputs = self.blip_model.generate(**inputs, **gen_params, num_return_sequences=2)
        # all_captions = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)

        return list(set([c.strip() for c in all_captions if len(c.split()) > 5]))

    def narrate(self, image, scene_label):
        # 1. Generation Phase
        start_gen = time.time()
        captions = self.generate_captions(image, scene_label)
        gen_time = time.time() - start_gen
        
        if not captions:
            return "No description available.", 0.0, gen_time, 0.0

        # 2. Evaluation Phase (CLIPScore)
        start_clip = time.time()
        inputs = self.clip_processor(
            text=captions, 
            images=[image]*len(captions), 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        out = self.clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        
        # Solving Tensor to Numpy detachment issue
        scores = (img_emb * txt_emb).sum(dim=1).detach().cpu().numpy()
        
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_caption = self._post_process(captions[best_idx])
        clip_time = time.time() - start_clip
        
        return best_caption, best_score, gen_time, clip_time

    def _post_process(self, caption):
        # Clean up common redundant prefixes
        prefixes = ["a photo of", "image of", "scene description:", "there is"]
        for p in prefixes:
            if caption.lower().startswith(p):
                caption = caption[len(p):].strip()
        return caption.capitalize() + "."

if __name__ == "__main__":
    from PIL import Image
    print("Testing Narrator locally...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_narrator = iVisionNarrator(device)
    print("Models Loaded Successfully!")