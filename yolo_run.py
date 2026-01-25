import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


# model = YOLO("yolo26n.onnx", task="detect")
model = YOLO("yolo26n-seg.onnx", task="segment")

# model.export(format="onnx")

cap = cv2.VideoCapture(0)

np.random.seed(69)
color = np.random.randint(0, 255, (3,), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    
    if not ret:
        continue

    r = model(frame, verbose=True)[0]
    
    # print(frame.shape)
    # frame = r.plot()
    # print(frame.shape)
    img = r.orig_img.copy()
    H, W = r.orig_shape

    if r.masks is not None:
        masks = scale_masks(r.masks.data.unsqueeze(1), r.boxes.orig_shape, padding=True)

        for mask_model in masks:
            mask = mask_model[0].cpu().numpy()
            # mask = np.squeeze(mask)
            # mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

            mask = mask > 0.5
            
            # color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img[mask] = img[mask] * 0.5 + color * 0.5
            # img[mask] = img[mask] * 0.5

    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{model.names[cls]} {conf:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

    cv2.imshow("YOLO 26", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()