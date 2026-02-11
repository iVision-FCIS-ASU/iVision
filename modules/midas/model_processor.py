import cv2
import torch
import numpy as np

first_execution = True

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution or not use_camera:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction

def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    depth_bw = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    depth_bw = np.uint8(depth_bw)
    depth_rgb = cv2.applyColorMap(depth_bw, cv2.COLORMAP_INFERNO)

    if image is None:
        return depth_bw, depth_rgb
    else:
        if grayscale:
            return depth_bw, np.concatenate((image, depth_bw), axis=1)
        else:
            return depth_bw, np.concatenate((image, depth_rgb), axis=1)