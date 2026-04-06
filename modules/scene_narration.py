import cv2
import line_profiler
import numpy as np
import numpy.typing as npt
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from .scene_classification import SceneClassifier
from .utils.clip_tokenizer import ClipTokenizer

class SceneNarrator:
    def __init__(self):
        print("\n-----SCENE NARRATION INITIALIZATION-----")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        # self.clip_processor.tokenizer.save_pretrained("weights/clip_tokenizer")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # self.blip_processor.tokenizer.padding_side = "left"
        self.blip_model.eval()
        self.clip_model.eval()
        # self.blip_model = torch.compile(self.blip_model)
        # self.clip_model = torch.compile(self.clip_model)

        self.clip_tokenizer = ClipTokenizer("weights/clip_tokenizer/tokenizer.json")

        print("-----SCENE NARRATION INITIALIZED-----\n")

    def __preprocess(self, frame: npt.NDArray) -> Image:
        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(raw_img)
        return raw_img

    @torch.inference_mode()
    def __generate_captions(self, image: Image, scene_label: str) -> list[str]:
        clean_label = scene_label.replace('_', ' ')
        prompts = [
            f"a detailed scene narration describing the {clean_label} environment",
            f"inside this {clean_label}, there is",
            f"a photo showing",
            f"" 
        ]

        gen_params = dict(
            max_length=50, 
            min_length=20,
            num_beams=8, # or num_beans=10 for high accuracy
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        all_captions: list[str] = []
        for p in prompts:
            inputs = self.blip_processor(image, text=p, return_tensors="pt").to(self.device)
            # Generate 2 options per prompt for CLIP to evaluate
            outputs = self.blip_model.generate(**inputs, **gen_params, num_return_sequences=2)
            all_captions += [self.blip_processor.decode(o, skip_special_tokens=True) for o in outputs]

        # # faster alternative but less accurate (requires padding_side="left" in __init__)
        # images = [image] * len(prompts)
        # inputs = {k: v.to(self.device) for k, v in self.blip_processor(images, text=prompts, return_tensors="pt", padding=True).items()}
        # outputs = self.blip_model.generate(**inputs, **gen_params, num_return_sequences=2)
        # all_captions = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)

        return list(set([c.strip() for c in all_captions if len(c.split()) > 5]))
    
    def __post_process(self, caption: str) -> str:
        prefixes = ["a photo of", "image of", "scene description:", "there is"]
        for p in prefixes:
            if caption.lower().startswith(p):
                caption = caption[len(p):].strip()
        return caption.capitalize() + "."

    def __print_report(self, scene_time: float, blip_time: float, clip_time: float, clip_score: float):
        print(f"\n{'='*30}")
        print(f"-NARRATION PERFORMANCE REPORT-")
        print(f"{'='*30}")
        print(f"1. Scene Classifier  : {scene_time:.3f}s")
        print(f"2. BLIP Generation   : {blip_time:.3f}s")
        print(f"3. CLIP Ranking      : {clip_time:.3f}s")
        print(f"4. CLIPScore         : {clip_score:.4f}")
        print(f"Total Inference Time : {(scene_time + blip_time + clip_time):.3f}s")
        print(f"{'='*30}\n")

    def __clip_preprocess_image(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # converts to [0,1] and CHW
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return transform(image)
    
    def __clip_get_best_caption(self, raw_img: Image, captions: list[str]) -> tuple[str, float]:
        image_tensor = self.__clip_preprocess_image(raw_img)
        image_batch = torch.stack([image_tensor] * len(captions)).to(self.device)

        input_ids, attention_mask = self.clip_tokenizer.encode(captions)

        out = self.clip_model(
            pixel_values=image_batch,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # inputs = self.clip_processor(
        #     text=captions, 
        #     images=[raw_img]*len(captions), 
        #     return_tensors="pt", 
        #     padding=True
        # ).to(self.device)
        # out = self.clip_model(**inputs)

        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_emb * txt_emb).sum(dim=1).detach().cpu().numpy()
        
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_caption = self.__post_process(captions[best_idx])

        return best_caption, best_score

    @torch.inference_mode()
    def get_narration(self, frame: npt.NDArray, scene_classifier: SceneClassifier) -> str:
        print("\n-----PREDICTING SCENE TYPE-----")
        scene_start = time.time()
        scene_label_binary, scene_label = scene_classifier.get_scene_type(frame)
        scene_time = time.time() - scene_start
        print("-----PREDICTED SCENE TYPE-----")
        
        raw_img = self.__preprocess(frame)

        print("-----GENERATING NARRATION-----")
        blip_start = time.time()
        captions = self.__generate_captions(raw_img, scene_label)
        blip_time = time.time() - blip_start
        print("-----GENERATED NARRATION-----")

        if not captions:
            self.__print_report(scene_time, blip_time, 0.0, 0.0)
            return "No description available."
        
        print("-----EVALUATING CLIP-----")
        clip_start = time.time()
        best_caption, best_score = self.__clip_get_best_caption(raw_img, captions)
        clip_time = time.time() - clip_start
        print("-----EVALUATED CLIP-----")

        self.__print_report(scene_time, blip_time, clip_time, best_score)

        caption = f"You are {scene_label_binary} in a {scene_label.replace('_', ' ')}. {best_caption}"
        return caption
    