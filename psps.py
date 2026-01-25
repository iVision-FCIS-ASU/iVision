import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks
import line_profiler

# --- Load ONNX session ---
sess = ort.InferenceSession(
    "depth_anything_v2_vits.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# --- Helper functions ---

def preprocess(frame, target_size=518):
    """Resize to square, normalize, convert to NCHW float32"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # pad to square (center)
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # normalize
    padded = padded.astype(np.float32) / 255.0
    padded = padded.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    return padded, (h, w), (top, bottom, left, right)

def postprocess(depth, orig_shape, pad_info, target_size=518):
    """Remove padding and resize back to original aspect ratio"""
    top, bottom, left, right = pad_info
    depth = depth[top:target_size-bottom, left:target_size-right]
    depth = cv2.resize(depth, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
    return depth

# @profile
def get_depth_image(frame):
    inp, orig_shape, pad_info = preprocess(frame)

    # Run ONNX inference
    outputs = sess.run(None, {"input": inp})
    print("Output shape:", outputs[0].shape)
    # depth_raw = outputs[0][0, 0]  # shape (H, W)
    import numpy as np

    out = outputs[0]               # ONNX outputs list -> first output
    # collapse singleton dims so we end up with a 2D HxW depth map
    depth_raw = np.squeeze(out)    # works for (1,1,H,W), (1,H,W), (H,W), (1,H*W) if square-ish

    # sanity check
    if depth_raw.ndim != 2:
        # fallback: try reshaping flattened vector into square
        if depth_raw.ndim == 1:
            side = int(np.sqrt(depth_raw.shape[0]))
            if side * side == depth_raw.shape[0]:
                depth_raw = depth_raw.reshape(side, side)
            else:
                raise ValueError(f"Can't reshape flattened output of size {depth_raw.shape[0]} into square.")
        else:
            raise ValueError(f"Unexpected depth map shape after squeeze: {depth_raw.shape}")


    # Postprocess: remove padding + restore original shape
    depth_resized = postprocess(depth_raw, orig_shape, pad_info)

    # Normalize for display
    depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # return depth_colored
    return depth_norm, depth_colored

def get_mean_depth_box(depth_image, box, mask):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    mean_depth = int(np.mean(depth_image[y1:y2, x1:x2]))
    min_depth = int(np.min(depth_image[y1:y2, x1:x2]))
    max_depth = int(np.max(depth_image[y1:y2, x1:x2]))
    return mean_depth, min_depth, max_depth

def get_mean_depth_mask(depth_image, box, mask):
    mean_depth = int(np.mean(depth_image[mask]))
    min_depth = int(np.min(depth_image[mask]))
    max_depth = int(np.max(depth_image[mask]))
    return mean_depth, min_depth, max_depth

def get_canny(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)

    high = np.percentile(mag, 90)
    low = 0.5 * high

    blur = cv2.GaussianBlur(image, (5, 5), 1.4)
    edges = cv2.Canny(blur, low, high)

    return edges


# --- YOLO functions ---

# model = YOLO("yolo26n.onnx", task="detect")
model = YOLO("yolo26n-seg.onnx", task="segment")

# --- Webcam loop ---
cap = cv2.VideoCapture(0)

# @profile
def run():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        depth_bw, depth_image = get_depth_image(frame)

        # depth_edges = get_canny(depth_bw)
        # depth_edges = cv2.cvtColor(depth_edges, cv2.COLOR_GRAY2BGR)

        # min_depth = np.min(depth_image)
        # max_depth = np.max(depth_image)

        r = model(frame, verbose=True)[0]
        
        if r.boxes is None:
            continue

        masks = scale_masks(r.masks.data.unsqueeze(1), r.boxes.orig_shape, padding=True)

        for box, mask_model in zip(r.boxes, masks):
        # for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            mask = mask_model[0].cpu().numpy() > 0.5

            # mean_depth, min_depth, max_depth = get_mean_depth_box(depth_bw, box)
            mean_depth, min_depth, max_depth = get_mean_depth_mask(depth_bw, mask)
            label = f"{model.names[cls]} {conf:.2f} Depth:({mean_depth}, {min_depth}, {max_depth})"

            cv2.rectangle(depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                depth_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # cv2.imshow(f"Depth: ({min_depth}, {max_depth})", depth_image)
        cv2.imshow(f"Depth Image with Boxes", depth_image)

        # combined_images = np.hstack((depth_image, depth_edges))
        # cv2.imshow(f"Output", combined_images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run()