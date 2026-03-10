####################################################################
#
# KERAS TO TFLITE
#

# import tensorflow as tf
# from tensorflow.keras.models import load_model

# model_keras = load_model("weights/complexity_estimation_v1.keras")
# converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# model_tflite = converter.convert()
# with open("weights/complexity_estimation_v1.tflite", "wb") as f:
#     f.write(model_tflite)
####################################################################


####################################################################
#
# DepthAnythingV2 TO ONNX
#

# import torch
# from modules.depth_anything_v2.dpt import DepthAnythingV2

# print("Exporting to ONNX format...")
# model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48,96,192,384])
# model.load_state_dict(torch.load("weights/depth_anything_v2_vits.pth", map_location="cpu"))
# model.eval()
# dummy = torch.randn(1, 3, 518, 518)

# with torch.no_grad():
#     torch.onnx.export(
#         model, dummy, "weights/depth_anything_v2_vits.onnx",
#         input_names=['input'], output_names=['output'],
#         opset_version=16,
#         do_constant_folding=True,
#         dynamic_axes={
#             'input': {0: 'batch', 2: 'height', 3: 'width'},
#             'output': {0: 'batch', 2: 'height', 3: 'width'}
#         },
#         export_params=True,
#         keep_initializers_as_inputs=False,
#     )
# print("Exported to weights/depth_anything_v2_vits.onnx")
####################################################################

####################################################################
#
# MiDaSv21 TO ONNX
#

# torch.onnx.export(
#     model,
#     sample,
#     "temp.onnx",
#     input_names=['input'],
#     output_names=['output'],
#     dynamo=True,
#     export_params=True,
#     keep_initializers_as_inputs=False
# )
####################################################################