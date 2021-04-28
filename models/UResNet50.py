from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Model

from models.helpers import unetify_encoder


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = ResNet50(input_tensor=inputs, weights="imagenet", include_top=False)
    encoder.trainable = False
    skip_connection_names = [
        "input_image",  # 512 x 512
        "conv1_relu",  # 256 x 256
        "conv2_block3_out",  # 128 x 128
        "conv3_block4_out",  # 64 x 64
    ]

    encoder_output = encoder.get_layer("conv4_block6_out").output  # 32 x 32

    num_filters = [64, 128, 256, 512]
    # num_filters = [16, 32, 64, 128]

    x = unetify_encoder(encoder_output, num_filters, skip_connection_names, encoder)

    model = Model(inputs, x, name="RESNET50_U-Net")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
