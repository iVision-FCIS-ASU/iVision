import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import line_profiler
from ultralytics import YOLO
from modules.object_detection import ObjectDetector

# interpreter = tf.lite.Interpreter("weights/yolo26n_float32.tflite", num_threads=4)
# interpreter.allocate_tensors()
# input_details_index = interpreter.get_input_details()[0]["index"]
# output_details_index = interpreter.get_output_details()[0]["index"]

class YOLOtflite:
    def __init__(self):
        self.target_size = 320
        self.interpreter = tf.lite.Interpreter(f"weights/yolo26n_float32.tflite", num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        self.classes = YOLO("weights/yolo26n.onnx", task="detect").names
    
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
        orig_shape: tuple[int, int],
        scale: float, 
        pad_info: tuple[int, int, int, int]
    ) -> npt.NDArray:
        preds = preds[0]
        boxes = preds[:, :4]
        scores = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)

        filter_mask = scores >= 0.25
        boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        cls_ids = cls_ids[filter_mask]

        h, w = orig_shape
        top, bottom, left, right = pad_info

        x1, y1, x2, y2 = boxes.T

        x1 = np.clip((x1 * self.target_size - left) / scale, 0, w)
        y1 = np.clip((y1 * self.target_size - top) / scale, 0, h)
        x2 = np.clip((x2 * self.target_size - left) / scale, 0, w)
        y2 = np.clip((y2 * self.target_size - top) / scale, 0, h)

        return np.dstack((x1, y1, x2, y2, scores, cls_ids))[0]
    
    @line_profiler.profile
    def get_objects(self, frame: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        image, orig_shape, scale, pad_info = self.__preprocess(frame)

        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_index)

        objects = self.__postprocess(preds, orig_shape, scale, pad_info)

        return objects

    # def get_objects(self, frame: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        

@line_profiler.profile
def run_object_detection():
    model = YOLOtflite()
    # model_old = ObjectDetector() 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break
        
        output = model.get_objects(frame)
        output_image = frame.copy()
        for x1, y1, x2, y2, conf, cls in output:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            conf = float(conf)
            cls = int(cls)
            # label = f"{self.classes[self.model_type][cls]} {conf:0.2f}"
            label = f"{cls} {conf:0.2f}"
            # cv2.circle(output_image, centroid, radius=self.overlay_centroid_radius, color=self.overlay_color, thickness=-1)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
        cv2.imshow("YOLO 26 Nano", output_image)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord("q") or key_pressed == ord("Q"):
            break
        # elif key_pressed != -1 and chr(key_pressed) in sizes:
        #     selected_size = sizes[chr(key_pressed)]
        # elif key_pressed != -1 and chr(key_pressed) in model_types:
        #     model_type = model_types[chr(key_pressed)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()
