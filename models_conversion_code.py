import tensorflow as tf
from tensorflow.keras.models import load_model

#
# KERAS TO TFLITE
#

# model_keras = load_model("weights/complexity_estimation_v1.keras")
# converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# model_tflite = converter.convert()
# with open("weights/complexity_estimation_v1.tflite", "wb") as f:
#     f.write(model_tflite)
