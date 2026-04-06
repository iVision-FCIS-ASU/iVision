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
from .utils.clip_projection_json import load_projections, save_projections

class SceneNarrator:
    def __init__(self):
        print("\n-----SCENE NARRATION INITIALIZATION-----")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", use_fast=True, local_files_only=True)
        # self.clip_processor.tokenizer.save_pretrained("weights/clip_tokenizer")
        
        # # backup code to regenerate projections
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        # save_projections(
        #     "weights/clip_projection/clip_projections.json",
        #     self.clip_model.visual_projection.weight,
        #     self.clip_model.visual_projection.bias,
        #     self.clip_model.text_projection.weight,
        #     self.clip_model.text_projection.bias
        # )
        
        self.clip_tokenizer = ClipTokenizer("weights/clip_tokenizer/tokenizer.json")
        
        self.vision_proj_weight, self.vision_proj_bias, \
        self.text_proj_weight, self.text_proj_bias = load_projections(
            "weights/clip_projection/clip_projections.json",
            device=self.device
        )

        self.clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="weights/blip", local_files_only=True).to(self.device)
        
        # self.blip_processor.tokenizer.padding_side = "left"
        self.blip_model.eval()
        # self.clip_model.eval()
        # self.blip_model = torch.compile(self.blip_model)
        # self.clip_model = torch.compile(self.clip_model)
        self.clip_vision_model.eval()
        self.clip_text_model.eval()

        # self.__export_models()

        self.sess_vision = ort.InferenceSession("weights/clip_model/clip_vision.onnx")
        self.sess_text = ort.InferenceSession("weights/clip_model/clip_text.onnx")

        # self.__compare_models()

        print("-----SCENE NARRATION INITIALIZED-----\n")

    @torch.inference_mode()
    def __export_models(self):
        class CLIPVisionWrapper(torch.nn.Module):
            def __init__(self, clip_vision_model: CLIPVisionModel):
                super().__init__()
                self.clip_vision_model = clip_vision_model

            def forward(self, pixel_values):
                vision_outputs = self.clip_vision_model(pixel_values=pixel_values)
                return vision_outputs.pooler_output

        clip_vision_wrapper = CLIPVisionWrapper(self.clip_vision_model)        
        dummy_image = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            clip_vision_wrapper,
            (dummy_image,),
            "weights/clip_model/clip_vision.onnx",
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_embeds": {0: "batch"}
            },
            opset_version=18
        )

        class CLIPTextWrapper(torch.nn.Module):
            def __init__(self, clip_text_model: CLIPTextModel):
                super().__init__()
                self.clip_text_model = clip_text_model

            def forward(self, input_ids, attention_mask):
                text_outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
                return text_outputs.pooler_output
        
        clip_text_wrapper = CLIPTextWrapper(self.clip_text_model)
        dummy_input_ids = torch.ones((2, 77), dtype=torch.long)
        dummy_attention_mask = torch.ones((2, 77), dtype=torch.long)
        torch.onnx.export(
            clip_text_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            "weights/clip_model/clip_text.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "text_embeds": {0: "batch"}
            },
            opset_version=18
        )     

    def __compare_models(self):
        # comparing vision models
        dummy_image = torch.randn(8, 3, 224, 224)
        
        with torch.inference_mode():
            torch_out_vision = self.clip_vision_model(dummy_image).pooler_output

        onnx_out_vision = self.sess_vision.run(
            None,
            {"pixel_values": dummy_image.cpu().numpy()}
        )[0]

        print(f"Vision Model: {np.allclose(torch_out_vision.cpu().numpy(), onnx_out_vision, atol=1e-4, rtol=1e-4)}")

        # comparing text models
        dummy_input_ids = torch.ones((1, 77), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, 77), dtype=torch.long)

        with torch.inference_mode():
            torch_out_text = self.clip_text_model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            ).pooler_output
        
        onnx_out_text = self.sess_text.run(
            None,
            {
                "input_ids": dummy_input_ids.cpu().numpy(),
                "attention_mask": dummy_attention_mask.cpu().numpy()
            }
        )[0]

        print(f"Text Model: {np.allclose(torch_out_text.cpu().numpy(), onnx_out_text, atol=1e-4, rtol=1e-4)}")

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
    
    def __linear(self, x, weight, bias):
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def __clip_get_best_caption_orig(self, raw_img: Image, captions: list[str]) -> tuple[str, float]:
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
    
    def __clip_get_best_caption(self, raw_img: Image, captions: list[str]) -> tuple[str, float]:
        image_tensor = self.__clip_preprocess_image(raw_img)
        image_batch = torch.stack([image_tensor] * len(captions)).to(self.device)

        input_ids, attention_mask = self.clip_tokenizer.encode(captions)

        vision_outputs = self.sess_vision.run(
            None,
            {"pixel_values": image_batch.cpu().numpy()}
        )[0]

        text_outputs = self.sess_text.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy()
            }
        )[0]

        image_embeds = torch.tensor(vision_outputs, device=self.device)
        text_embeds  = torch.tensor(text_outputs, device=self.device)

        image_embeds = self.__linear(image_embeds, self.vision_proj_weight, self.vision_proj_bias)
        text_embeds  = self.__linear(text_embeds, self.text_proj_weight, self.text_proj_bias)
        
        img_emb = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
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
    