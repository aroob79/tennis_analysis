from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19


def keyPointModel(number_of_layers=22):

    # use pretrain vgg19 model to train it on the custom data
    vgg19_model = VGG19(weights='imagenet', include_top=True)

    for layer in vgg19_model.layers[:number_of_layers]:
        layer.trainable = False
    output_layer = vgg19_model.get_layer(
        vgg19_model.layers[number_of_layers].name)

    # define the base model
    base_model = keras.Model(
        inputs=[vgg19_model.input], outputs=output_layer.output)
    _, inp_size = base_model.output.shape

    # using sequential model define the tain layers and output layer
    tail_model = keras.Sequential([keras.Input(shape=(inp_size,)),
                                  keras.layers.Dense(
                                      units=8000, activation='relu'),
                                  keras.layers.Dense(
                                      units=2000, activation='relu'),
                                   keras.layers.Dense(
        units=500, activation='relu'),
        keras.layers.Dense(units=28, activation='linear'),
        keras.layers.Reshape([-1, 2])])
    combine_output = tail_model(base_model.output)

    # combine the base net and taile network
    combined_model = keras.Model(
        inputs=base_model.input, outputs=combine_output)

    return combined_model
