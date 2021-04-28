import tensorflow.keras.backend as K

ALPHA = 0.8
GAMMA = 2


def focal_loss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return loss