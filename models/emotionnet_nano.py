from tensorflow import keras
from tensorflow.keras import layers, Model


def conv(filters, name, strides=(1, 1)):
    return layers.Conv2D(filters, 3, activation='relu', padding='same', name=name, strides=strides)


def add_conv(filters, name, strides=(1, 1)):
    def __add_conv_inner(inputs):
        s = layers.add(inputs)
        return conv(filters, name, strides=strides)(s)

    return __add_conv_inner


def identity(filters, name):
    def __identity_inner(inputs):
        s = layers.add(inputs)
        return layers.Conv2D(filters, 1, name=name, strides=(2, 2))(s)

    return __identity_inner


def emotion_nano_b(input_shape=(48, 48, 1), num_classes=7):
    block1 = [11, 9, 11, 8, 11, 7, 11, 27]
    block2 = [27, 19, 27, 26, 27, 36]
    block3 = [64, 39, 64, 24, 64]

    inputs = keras.Input(shape=input_shape)
    block1_conv1 = conv(block1[0], "block1_conv1")(inputs)
    block1_conv2 = conv(block1[1], "block1_conv2")(block1_conv1)
    block1_conv3 = conv(block1[2], "block1_conv3")(block1_conv2)
    block1_conv4 = add_conv(block1[3], "block1_conv4")([block1_conv1, block1_conv3])
    block1_conv5 = conv(block1[4], "block1_conv5")(block1_conv4)
    block1_conv6 = add_conv(block1[5], "block1_conv6")([block1_conv1, block1_conv3, block1_conv5])
    block1_conv7 = conv(block1[6], "block1_conv7")(block1_conv6)
    block1_conv8 = add_conv(block1[7], "block1_conv8", strides=(2, 2))([block1_conv1, block1_conv5, block1_conv7])

    identity1 = identity(27, "identity1")([block1_conv1, block1_conv3, block1_conv5])

    block2_conv1 = conv(block2[0], "block2_conv1")(block1_conv8)
    block2_conv2 = add_conv(block2[1], "block2_conv2")([block2_conv1, identity1])
    block2_conv3 = conv(block2[2], "block2_conv3")(block2_conv2)
    block2_conv4 = add_conv(block2[3], "block2_conv4")([block2_conv1, block2_conv3])
    block2_conv5 = conv(block2[4], "block2_conv5")(block2_conv4)
    block2_conv6 = add_conv(block2[5], "block2_conv6", strides=(2, 2))([block2_conv1, block2_conv3, identity1])

    identity2 = identity(64, "identity2")([block1_conv8, block2_conv3, block2_conv5, identity1])

    block3_conv1 = conv(block3[0], "block3_conv1")(block2_conv6)
    block3_conv2 = add_conv(block3[1], "block3_conv2")([block3_conv1, identity2])
    block3_conv3 = conv(block3[2], "block3_conv3")(block3_conv2)
    block3_conv4 = add_conv(block3[3], "block3_conv4")([block3_conv1, block3_conv3, identity2])
    block3_conv5 = conv(block3[4], "block3_conv5")(block3_conv4)

    pool_add = layers.add([block3_conv1, block3_conv3, block3_conv5, identity2])
    pool = layers.GlobalAveragePooling2D(name="global_pool")(pool_add)

    dense = layers.Dense(num_classes, activation='softmax', name="output")(pool)

    model = Model(inputs=inputs, outputs=dense)
    model.compile()
    return
