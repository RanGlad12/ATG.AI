import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2B3

def build_model(num_classes=2, img_height=299, img_width=299):
    '''
    Constructs an EfficientNet based Tensorflow model with the top layers removed.
    Added a Dense layer of 100 units and a Dense classifying layer of 2 units for the top layers
    Recieves:
    num_classes - the number of classes to classify
    img_height, img_width - the size of the image input to the network.
    Returns:
    model - Tensorflow model
    '''
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))

    model = EfficientNetV2B3(include_top=False, input_tensor=inputs, weights='imagenet') 
    # Freeze the pretrained weights
    #model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = tf.keras.layers.Dense(100, activation='relu', name='pre_pred')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
