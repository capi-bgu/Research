import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def conv(filters, name, strides=(1, 1)):
    return layers.Conv2D(filters, 3, padding='same', name=name, strides=strides, use_bias=False)


def res_layer(filters, name, strides=1, downscale=None):
    def __inner_res_layer(inputs):
        residual = inputs

        x = conv(filters, f"{name}_conv1", strides)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = conv(filters, f"{name}_conv2")(x)
        x = layers.BatchNormalization()(x)

        if downscale is not None:
            residual = downscale(inputs)

        x = layers.Add()([x, residual])
        return layers.ReLU()(x)

    return __inner_res_layer


def res_block(filters, num_layers, name, strides=1):
    def __inner_res_block(inputs):
        downscale = None
        if strides != 1:
            downscale = keras.Sequential([
                layers.Conv2D(filters, 1, padding='same', strides=strides, use_bias=False),
                layers.BatchNormalization()
            ])

        x = res_layer(filters, f"{name}_layer1", strides=strides, downscale=downscale)(inputs)
        for i in range(1, num_layers):
            x = res_layer(filters, f"{name}_layer{i + 1}")(x)

        return x

    return __inner_res_block


def resnet(blocks, image_shape):
    inputs = layers.Input(shape=image_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = res_block(64, blocks[0], f"block1")(x)
    x = res_block(128, blocks[1], f"block2", strides=2)(x)
    x = res_block(256, blocks[2], f"block3", strides=2)(x)
    x = res_block(512, blocks[3], f"block4", strides=2)(x)
    out = layers.GlobalAveragePooling2D()(x)

    return Model(inputs=inputs, outputs=out, name='resnet')


def fanet(image_shape=(48, 48, 1), frames=5,
          attention_type='self', base=None, num_classes=6) -> Model:
    # input shape should be (examples, frames, w, h, d)
    inputs = layers.Input(shape=(frames,) + image_shape)
    if base is None:
        base = resnet((2, 2, 2, 2), image_shape)
    alpha = layers.Dense(1, activation='sigmoid')
    beta = layers.Dense(1, activation='sigmoid')
    dropout = layers.Dropout(0.5)
    outputs = []
    alphas = []

    for i in range(inputs.shape[1]):
        # we take the i'th frame from each of the examples
        # and propagate it through the base network.
        x = base(inputs[:, i, :, :, :])
        outputs.append(x)
        alphas.append(alpha(dropout(x)))

    # we stack all the outputs and alphas along the frame dimension
    # this brings us back to using tensors.
    outputs_stack = tf.stack(outputs, axis=1)
    alphas_stack = tf.stack(alphas, axis=1)

    # calculate self attention score
    attention = tf.reduce_sum(outputs_stack * alphas_stack, axis=1) / tf.reduce_sum(alphas_stack, axis=1)

    if attention_type == 'relation':
        betas = []
        # propagate self attention outputs through relation attention layer.
        for i in range(len(outputs)):
            outputs[i] = tf.concat([outputs[i], attention], axis=1)
            betas.append(beta(dropout(outputs[i])))

        # we again stack the outputs of the relation attention output
        # to bring us back to working with tensors.
        outputs_stack = tf.stack(outputs, axis=1)
        betas_stack = tf.stack(betas, axis=1)

        # calculate the relation attention scores.
        attention = tf.reduce_sum(outputs_stack * betas_stack * alphas_stack, axis=1)
        attention = attention / tf.reduce_sum(betas_stack * alphas_stack, axis=1)

    attention = layers.Dropout(0.5)(attention)
    pred = layers.Dense(num_classes, activation='softmax')(attention)

    return Model(inputs=inputs, outputs=pred, name="Frame-Attention")
