from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.models import Model

from models.helpers import unetify_encoder


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False)
    encoder.trainable = False
    skip_connection_names = [
        "input_image",  # 512 x 512
        "block_1_expand_relu",  # 256 x 256
        "block_3_expand_relu",  # 128 x 128
        "block_6_expand_relu",  # 64 x 64
    ]

    encoder_output = encoder.get_layer("block_13_expand_relu").output  # 32 x 32

    num_filters = [96, 128, 256, 512]
    # num_filters = [16, 32, 64, 128]

    x = unetify_encoder(encoder_output, num_filters, skip_connection_names, encoder)

    model = Model(inputs, x, name="umobilev2net")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
