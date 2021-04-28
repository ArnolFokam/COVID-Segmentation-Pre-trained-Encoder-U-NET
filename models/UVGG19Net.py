from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model

from models.helpers import unetify_encoder


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = VGG19(input_tensor=inputs, weights="imagenet", include_top=False)
    encoder.trainable = False

    skip_connection_names = [
        "block1_conv2",  # 512 x 512
        "block2_conv2",  # 256 x 256
        "block3_conv4",  # 128 x 128
        "block4_conv4",  # 64 x 64
    ]

    encoder_output = encoder.get_layer("block5_conv4").output  # 32 x 32

    num_filters = [64, 128, 256, 512]
    # num_filters = [16, 32, 64, 128]

    x = unetify_encoder(encoder_output, num_filters, skip_connection_names, encoder)

    model = Model(inputs, x, name="uvgg19net")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
