import os
from random import shuffle

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def create_dir(image_dir):
    if not os.path.exists(image_dir):
        print("Directory Created", image_dir)
        os.makedirs(image_dir)


def list_files(path):
    for file_or_dir in os.listdir(path):
        if file_or_dir[0] != '.':
            yield file_or_dir


def randomly_assign_train_test(img_path, test_size=0.1):
    # Stores path to image and label
    data = []

    os.chdir(img_path)
    # # for i in os.listdir('output_arrs/'):
    for label, dir_name in enumerate(list_files(img_path)):
        train_url, test_url = 'data/train/', 'data/validation/'
        create_dir(os.path.join(test_url, dir_name))
        create_dir(os.path.join(train_url, dir_name))

        os.chdir(dir_name)

        for image_name in list_files(os.getcwd()):
            data.append({'startpath': os.path.join(os.getcwd(), image_name),
                         'trainpath': os.path.join(train_url, image_name),
                         'testpath': os.path.join(test_url, image_name)})

        os.chdir('../')

    shuffle(data)

    testing_size = int(test_size * len(data))

    train_data = data[testing_size:]
    test_data = data[:testing_size]

    for image_path_dict in train_data:
        img = Image.open(image_path_dict['startpath'])
        img.save(image_path_dict['trainpath'])

    for image_path_dict in test_data:
        img = Image.open(image_path_dict['startpath'])
        img.save(image_path_dict['testpath'])


def run_model():
    # dimensions of our images.
    img_width, img_height = 130, 130

    train_data_dir = 'data/train/'
    validation_data_dir = 'data/validation/'
    nb_train_samples = 1188
    nb_validation_samples = 134
    nb_epoch = 500

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3), dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

    model.load_weights('first_try.h5')


if __name__ == '__main__':
    # run_model()
    randomly_assign_train_test('images/')
