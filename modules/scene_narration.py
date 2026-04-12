import cv2
import line_profiler
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel, CLIPConfig
from .scene_classification import SceneClassifier
from .utils.clip_tokenizer import ClipTokenizer
from .utils.clip_exporter import clip_export_tokenizer, clip_export_projections, clip_load_projections, clip_export_models

class SceneNarrator:
    def __init__(self):
        print("\n-----SCENE NARRATION INITIALIZATION-----")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        # self.blip_processor.tokenizer.padding_side = "left"
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", local_files_only=True).to(self.device)
        self.blip_model.eval()

        # clip_export_tokenizer()
        self.clip_tokenizer = ClipTokenizer("weights/clip_tokenizer/tokenizer.json")
        
        # clip_export_projections()
        self.vision_proj_weight, self.vision_proj_bias, \
        self.text_proj_weight, self.text_proj_bias = clip_load_projections(
            "weights/clip_projection/clip_projections.json"
        )

        self.vision_proj_weight = self.vision_proj_weight.T
        self.text_proj_weight = self.text_proj_weight.T

        # self.clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # self.clip_vision_model.eval()
        # self.clip_text_model.eval()
        
        # clip_export_models()
        self.clip_onnx_vision = ort.InferenceSession("weights/clip_model/clip_vision.onnx")
        self.clip_onnx_text = ort.InferenceSession("weights/clip_model/clip_text.onnx")

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

    def __clip_preprocess_image(self, image: Image.Image) -> npt.NDArray:
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return transform(image).detach().cpu().numpy()
    
    def __clip_preprocess_image_np(self, image: Image.Image) -> npt.NDArray:
        width, height = image.size
        scale = 224 / min(width, height)
        new_w, new_h = int(width * scale), int(height * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))

        img = np.array(image).astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
        std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
        img = (img - mean) / std

        return img

    
    def __linear(self, x, weight, bias):
        out = x @ weight.T
        if bias is not None:
            out += bias
        return out

    def __linear_np(self, x: npt.NDArray, weight: npt.NDArray, bias: npt.NDArray) -> npt.NDArray:
        out = x @ weight
        if bias is not None:
            out = out + bias
        return out

    def __clip_get_best_caption_torch(self, raw_img: Image, captions: list[str]) -> tuple[str, float]:
        image_tensor = self.__clip_preprocess_image(raw_img)
        image_batch = torch.stack([image_tensor] * len(captions)).to(self.device)

        input_ids, attention_mask = self.clip_tokenizer.encode(captions)

        vision_outputs = self.clip_vision_model(pixel_values=image_batch)
        text_outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output
        image_embeds = self.__linear(image_embeds, self.vision_proj_weight, self.vision_proj_bias)
        text_embeds  = self.__linear(text_embeds, self.text_proj_weight, self.text_proj_bias)
        
        img_emb = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores = (img_emb * txt_emb).sum(dim=1).detach().cpu().numpy()
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_caption = self.__post_process(captions[best_idx])

        return best_caption, best_score
    
    @line_profiler.profile
    def __clip_get_best_caption(self, raw_img: Image, captions: list[str]) -> tuple[str, float]:
        image_np = self.__clip_preprocess_image_np(raw_img)
        image_batch = np.repeat(image_np[None, ...], len(captions), axis=0)

        input_ids, attention_mask = self.clip_tokenizer.encode(captions)

        vision_outputs = self.clip_onnx_vision.run(
            None,
            {"pixel_values": image_batch}
        )[0]

        text_outputs = self.clip_onnx_text.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )[0]

        image_embeds = vision_outputs
        text_embeds  = text_outputs

        # image_embeds = image_embeds @ self.vision_proj_weight
        # if self.vision_proj_bias is not None:
        #     image_embeds += self.vision_proj_bias

        # text_embeds = text_embeds @ self.text_proj_weight
        # if self.text_proj_bias is not None:
        #     text_embeds += self.text_proj_bias

        image_embeds = self.__linear_np(image_embeds, self.vision_proj_weight, self.vision_proj_bias)
        text_embeds = self.__linear_np(text_embeds, self.text_proj_weight, self.text_proj_bias)
        
        img_emb = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        txt_emb = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        
        scores = np.sum(img_emb * txt_emb, axis=1)
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
    