import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "weights/paddleocr_data"

import cv2
import numpy as np
import numpy.typing as npt
from paddleocr import PaddleOCR
from typing import TypeAlias

BoundingBox: TypeAlias = tuple[int, int, int, int]

class ReadingMode:
    def __init__(self):
        print("\n-----READING MODE INITIALIZATION-----")
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
            text_det_unclip_ratio=1.2,
        )
        print("-----READING MODE INITIALIZED-----\n")

    def get_text(
        self, 
        frame: npt.NDArray
    ) -> tuple[list[str], list[float], list[BoundingBox]]:
        self.result = self.ocr.predict(frame)[0]

        self.texts  = self.result["rec_texts"]
        self.scores = self.result["rec_scores"]
        self.boxes  = self.result["rec_boxes"]

        return self.texts, self.scores, self.boxes
    
    def draw_boxes(
        self,
        frame: npt.NDArray
    ) -> npt.NDArray:
        img = frame.copy()

        if "preprocessed_img" in self.result.img:
            h, w, _ = img.shape
            img = cv2.cvtColor(np.array(self.result.img["preprocessed_img"]), cv2.COLOR_RGB2BGR)
            img = img[:h, -w:, :]

        for box in self.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return img
