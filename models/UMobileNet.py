from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.models import Model

from models.helpers import unetify_encoder


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = MobileNet(input_tensor=inputs, weights="imagenet", include_top=False)

    encoder.trainable = False

    skip_connection_names = [
        "input_image",  # 512 x 512
        "conv_pw_1_relu",  # 256 x 256
        "conv_pw_3_relu",  # 128 x 128
        "conv_pw_5_relu",  # 64 x 64
    ]

    encoder_output = encoder.get_layer("conv_pw_11_relu").output  # 32 x 32

    num_filters = [96, 128, 256, 512]
    # num_filters = [16, 32, 64, 128]

    x = unetify_encoder(encoder_output, num_filters, skip_connection_names, encoder)

    model = Model(inputs, x, name="umobilenet")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
