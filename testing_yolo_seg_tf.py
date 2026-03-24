import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import line_profiler
from ultralytics import YOLO
from modules.object_detection import ObjectDetector
from typing import TypeAlias, Literal

# interpreter = tf.lite.Interpreter("weights/yolo26n-seg_float32.tflite", num_threads=4)
# interpreter.allocate_tensors()
# input_details_index = interpreter.get_input_details()[0]["index"]
# output_details_index = interpreter.get_output_details()[0]["index"]

# print(f"Input details:\n{interpreter.get_input_details()}\n")
# print(f"Output details:\n{interpreter.get_output_details()}\n")

npimagebgr : TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.uint8]
npimagegray: TypeAlias = np.ndarray[tuple[int, int], np.uint8]

Box = tuple[int, int, int, int, int, float]
Mask = npimagegray
Objects = tuple[list[Box], list[Mask]]

# class Results:
#     def __init__(self, boxes: list[Box], masks, centroids: ):  

class YOLOtflite:
    def __init__(self):
        self.target_size = 320
        self.interpreter = tf.lite.Interpreter(f"weights/yolo26n-seg_float32.tflite", num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output0_index = self.interpreter.get_output_details()[0]["index"]
        self.output1_index = self.interpreter.get_output_details()[1]["index"]

        self.classes = YOLO("weights/yolo26n-seg_float32.tflite", task="segment").names

        self.objects: Objects = ([], [])
        self.overlay_mask_color = 0.5 * np.array((0, 255, 0), dtype=np.uint8)
    
    @line_profiler.profile
    def __preprocess(
        self,
        frame: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[int, int], tuple[int, int, int, int]]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) 
        image /= 255.0

        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        image = np.expand_dims(image, axis=0)

        return image, frame.shape[:2], scale, (top, bottom, left, right)

    @line_profiler.profile
    def __postprocess(
        self,
        preds: npt.NDArray, 
        proto: npt.NDArray, 
        orig_shape: tuple[int, int],
        scale: float, 
        pad_info: tuple[int, int, int, int]
    ) -> Objects:
        preds = preds[0]
        proto = proto[0]

        boxes = preds[:, :4]
        confs = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)
        coeffs = preds[:, 6:]

        filter_mask = confs >= 0.25
        boxes = boxes[filter_mask]
        confs = confs[filter_mask]
        cls_ids = cls_ids[filter_mask]
        coeffs = coeffs[filter_mask]

        h, w = orig_shape
        top, bottom, left, right = pad_info

        x1, y1, x2, y2 = boxes.T

        x1: npt.NDArray = np.clip((x1 * self.target_size - left) / scale, 0, w).astype(np.int32)
        y1: npt.NDArray = np.clip((y1 * self.target_size - top) / scale, 0, h).astype(np.int32)
        x2: npt.NDArray = np.clip((x2 * self.target_size - left) / scale, 0, w).astype(np.int32)
        y2: npt.NDArray = np.clip((y2 * self.target_size - top) / scale, 0, h).astype(np.int32)

        # final_boxes = [
        #     (x1, y1, x2, y2, cls_id, conf) 
        #     for x1, y1, x2, y2, cls_id, conf 
        #     in zip(x1_np, y1_np, x2_np, y2_np, cls_ids, confs)
        # ]

        final_boxes = [
            (x1[i], y1[i], x2[i], y2[i], cls_ids[i], confs[i]) 
            for i in range(len(x1))
        ]

        proto_flat = proto.reshape(-1, 32)
        masks = proto_flat @ coeffs.T 
        masks: npt.NDArray = 1 / (1 + np.exp(-masks))
        masks = masks.reshape(80, 80, -1)
        masks = np.transpose(masks, (2, 0, 1))

        final_masks = []

        for i, mask in enumerate(masks):
            mask: npt.NDArray = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            mask = mask[top: self.target_size - bottom, left: self.target_size - right]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            cropped = np.zeros_like(mask, dtype=np.uint8)
            cropped[y1[i]: y2[i], x1[i]:x2[i]] = (mask[y1[i]: y2[i], x1[i]:x2[i]] > 0.5)
            final_masks.append(cropped.astype(bool))

        # return np.dstack((x1, y1, x2, y2, confs, cls_ids))[0]
        return final_boxes, final_masks
    
    @line_profiler.profile
    def __get_objects(
        self, 
        frame: npimagebgr
    ) -> Objects:
        image, orig_shape, scale, pad_info = self.__preprocess(frame)

        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output0_index)
        proto = self.interpreter.get_tensor(self.output1_index)

        objects = self.__postprocess(preds, proto, orig_shape, scale, pad_info)

        return objects

    def get_objects(
        self, 
        frame: npimagebgr
    ) -> Objects:
        self.objects = self.__get_objects(frame)
        return self.objects

    def draw_objects(
        self,
        output_image: npt.NDArray
    ):
        boxes, masks = self.objects
        
        if len(masks) == 0:
            for box in boxes:
                x1, y1, x2, y2, cls, conf = box
                label = f"{self.classes[cls]} {conf:0.2f}"
                # cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
        else:
            for box, mask in zip(boxes, masks):
                x1, y1, x2, y2, cls, conf = box
                output_image[mask] = 0.5 * output_image[mask] + self.overlay_mask_color

                label = f"{self.classes[cls]} {conf:0.2f}"
                # cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

        
        

@line_profiler.profile
def run_object_detection():
    model = YOLOtflite()
    model_old = ObjectDetector() 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    current_key = "1"
    keys = ["1", "2"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        # frame = cv2.imread("C:/Users/MoDo/Downloads/cat.webp")
        output_image = frame.copy()
        
        if current_key == "1":
            model.get_objects(frame)
            model.draw_objects(output_image)
        else:
            model_old.get_objects(frame, ObjectDetector.ModelType.YOLO_SEGMENT)
            model_old.draw_objects(output_image)
        
        # model.get_objects(frame)
        # model.draw_objects(output_image)
        cv2.imshow("YOLO 26 Nano", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        elif key_pressed != -1 and chr(key_pressed) in keys:
            current_key = chr(key_pressed)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()
