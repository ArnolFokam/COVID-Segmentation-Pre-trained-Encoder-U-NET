from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation, Concatenate
from tensorflow.python.keras.models import Model


def build_model(input_shape, metrics, loss, optimizer):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = VGG16(input_tensor=inputs, weights="imagenet", include_top=False)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    for layer in encoder.layers:
        layer.trainable = False

    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names) + 1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
