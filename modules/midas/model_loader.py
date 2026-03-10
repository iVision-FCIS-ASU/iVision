import cv2
import torch

from .dpt_depth import DPTDepthModel
from .midas_net import MidasNet
from .midas_net_custom import MidasNet_small
from .transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

model_paths = {
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",
    "midas_v21_small_256": "weights/midas_v21_small_256.pt",
}

def load_model(device, model_path: str, model_type: str, optimize=True, square=False):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """

    keep_aspect_ratio = not square

    if model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_small_256":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and (device == torch.device("cuda")):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    return model, transform, net_w, net_h
