from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import os
import dataset as ds
import utils
import matplotlib.pyplot as plt

models_path = './models/'
output_images = './output_images/'


def classifierCNN(input_shape, name='cls_cnn', load_weights=None):

    model = Sequential(name=name)
    model.add(Lambda(lambda x: x/255, input_shape=input_shape))  # normalize
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv_0', padding="same"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv_1', padding="same"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv_2', padding="same"))
    model.add(MaxPool2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))
    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8), activation="sigmoid", name='classifier'))

    if load_weights is not None:
        print('Loading weights', load_weights)
        model.load_weights(load_weights)
    else:
        print('Loading weights failed', load_weights)

    return model

def train(dataset, epochs=30, batch_size=32, load_weights=None, debug=False):
    timestamp = utils.standardDatetime()
    # load dataset generator and metrics
    gen_train, gen_valid, info = ds.loadDatasetGenerators(dataset, batch_size=batch_size)
    print(info)

    # create the model a eventually preload the weights (set to None or remove to disable)
    model = classifierCNN(input_shape=info['input_shape'], load_weights=load_weights)
    if debug:
        model_img_path = output_images + model.name + '.png'
        plot_model(model, to_file=model_img_path, show_shapes=True)
        utils.showImage(model_img_path)

    model_name = model.name +'_'+ timestamp

    # Intermediate model filename template

    filepath = models_path + model_name + "_{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5"
    # save model after every epoc, only if improved the val_loss.
    # very handy (with a proper environment (2 GPUs anywhere) you can test your model while it still train)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    # detect stop in gain on val_loss between epocs and terminate early, avoiding unnecessary computation cycles.
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=1)
    callbacks_list = [checkpoint, earlystopping]

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    history_object = model.fit_generator(gen_train, info['n_train_batch'], verbose=1, epochs=epochs,
                                         validation_data=gen_valid, validation_steps=info['n_valid_batch'],
                                         callbacks=callbacks_list)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['acc'])
    plt.plot(history_object.history['val_acc'])
    plt.title('model mean squared error accuracy')
    plt.ylabel('mean squared error accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
    plt.show()

    return model_name