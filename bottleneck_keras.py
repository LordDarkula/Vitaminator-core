import os
import numpy as np
from process_images import list_files, create_dir, convert_to_np

# from keras.models import Sequential

# path to the model weights file.
weights_path = 'vgg16_weights.npz'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 130, 130

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_epoch = 500


def batch_predict_generator(X, batch_size):
    number_of_batches = X.shape[0] / batch_size
    while 1:
        for i in range(number_of_batches):
            yield X[i * batch_size:(i + 1) * batch_size].todense()


def save_images_to_arrays():
    image_paths = [train_data_dir, validation_data_dir]
    array_dict = {train_data_dir: [[], []],
                  validation_data_dir: [[], []]}

    for train_or_test_dir in image_paths:
        for label, class_dir in enumerate(list_files(train_or_test_dir)):
            for image in list_files(os.path.join(train_or_test_dir, class_dir)):
                image_path = os.path.join(train_or_test_dir, class_dir, image)

                sample = convert_to_np(image_path)
                array_dict[train_or_test_dir][0].append(sample)

                one_hot = [0.0, 1.0] if label == 0 else [1.0, 0.0]
                array_dict[train_or_test_dir][1].append(one_hot)

    train_X, train_y = array_dict[train_data_dir]
    test_X, test_y = array_dict[validation_data_dir]

    create_dir('bottleneck')

    np.save('bottleneck/bottleneck_train_X', np.array(train_X))
    np.save('bottleneck/bottleneck_train_y', np.array(train_y))
    np.save('bottleneck/bottleneck_test_X', np.array(test_X))
    np.save('bottleneck/bottleneck_test_y', np.array(test_y))
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
