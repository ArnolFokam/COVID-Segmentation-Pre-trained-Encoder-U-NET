from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.models import Model

from models.helpers import unetify_encoder


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = DenseNet169(input_tensor=inputs, weights="imagenet", include_top=False)
    encoder.trainable = False
    skip_connection_names = [
        "input_image",  # 512 x 512
        "conv1/relu",  # 256 x 256
        "pool2_relu",  # 128 x 128
        "pool3_relu",  # 64 x 64
    ]

    encoder_output = encoder.get_layer("pool4_relu").output  # 32 x 32

    num_filters = [96, 128, 256, 512]
    # num_filters = [16, 32, 64, 128]

    x = unetify_encoder(encoder_output, num_filters, skip_connection_names, encoder)

    model = Model(inputs, x, name="udensenet169")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
