from os import listdir, makedirs
from random import shuffle

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from os.path import join, exists


def create_dir(image_dir):
    if not exists(image_dir):
        print("Directory Created", image_dir)
        makedirs(image_dir)


def randomly_assign_train_test(img_path, test_size=0.1):
    features = []
    fullpath = {}
    dirs = []
    # # for i in os.listdir('output_arrs/'):
    img_list = listdir(img_path)
    if img_list.__contains__('.DS_Store'):
        img_list.pop()
    for i, dir_type in enumerate(img_list):
        curr_dir = join(img_path, dir_type)
        dirs.append(dir_type)

        create_dir('data/train/' + dir_type)
        create_dir('data/validation/' + dir_type)

        for file in listdir(curr_dir):
            if file != ".DS_Store":
                features.append([file, i])
                fullpath[file] = join(curr_dir, file)

    shuffle(features)
    features = np.array(features)
    np.random.shuffle(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    print("The Dictionary:", dirs)
    for i, val in enumerate(train_x):
        img = Image.open(fullpath[val])
        print("Answer", dirs[train_y[i]])
        print("File Created", fullpath[val], 'data/train/' + join(dirs[train_y[i]], val))
        img.save('data/train/' + join(dirs[train_y[i]], val))

    for i, val in enumerate(test_x):
        img = Image.open(fullpath[val])
        print("Answer", dirs[test_y[i]])
        print("File Created", fullpath[val], 'data/validation/' + join(dirs[test_y[i]], val))
        img.save('data/validation/' + join(dirs[test_y[i]], val))

    return train_y, test_y


# randomly_assign_train_test(img_path='images/')


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
