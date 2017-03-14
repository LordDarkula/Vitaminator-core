from random import shuffle

import numpy as np
import os
from PIL import Image
import hashlib

IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']
width, height = 130, 130
img_type = "RGB"
image_ndim = 3

unique = []
items = []
size = 0


def remove_duplicates(dir, size=1):
    for filename in os.listdir(dir):
        curr = os.path.join(dir, filename)
        if os.path.isdir(curr):
            remove_duplicates(curr, size=size)
        if os.path.isfile(curr) and filename != ".DS_Store":
            img = Image.open(curr)
            filehash = np.array(img).tolist()
            if filehash not in unique:
                unique.append(filehash)
                items.append(curr)
            else:
                file = items[unique.index(filehash)]
                print("Duplicates Found")
                os.remove(curr)


def convert_to_np(image_path):
    arr = np.array(Image.open(image_path).convert('L'))
    arr = np.reshape(arr, (width * height))
    arr = arr.astype(np.float32)
    return arr


def get_images(dir_path, image_dir, np_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)

    for file_path in os.listdir(dir_path):
        curr = os.path.join(dir_path, file_path)
        if os.path.isdir(curr):
            get_images(curr, image_dir, np_dir)
        else:
            file_type = curr.split('.')[-1]
            if file_type in IMAGE_FILE_TYPES:
                try:
                    img = Image.open(curr)
                    base_path = str(hash_name(curr))
                    image_path = image_dir + base_path + '.jpg'
                    img = img.resize((width, height))
                    img.save(image_path)
                    # convert_to_np(image_path, np_dir + base_path)
                except Exception:
                    print('exception occured')


def hash_name(curr):
    return hashlib.md5(os.path.splitext(curr)[0]).hexdigest()


def get_multiple_list(dirs):
    for dir in dirs:
        get_images(dir_path=dir,
                   image_dir='new_images/' + dir,
                   np_dir='output_arrs/' + dir)


def list_files(path):
    for file_or_dir in os.listdir(path):
        if file_or_dir[0] != '.':
            yield file_or_dir


def randomly_assign_train_test(img_path, test_size=0.1, remove_data_folder=False):
    # Stores path to image and label
    if os.path.exists('data'):
        if remove_data_folder:
            print("Data Folder Removed")
            os.remove('data')
        else:
            return
    data = []

    # # for i in os.listdir('output_arrs/'):
    for label, dir_name in enumerate(list_files(img_path)):
        train_url = os.path.join('data/train/', dir_name)
        test_url = os.path.join('data/validation/', dir_name)
        create_dir(train_url)
        create_dir(test_url)

        os.chdir(os.path.join(img_path, dir_name))

        for image_name in list_files(os.getcwd()):
            data.append({'startpath': os.path.join(os.getcwd(), image_name),
                         'trainpath': os.path.join(train_url, image_name),
                         'testpath': os.path.join(test_url, image_name)})

        os.chdir('../../')

    shuffle(data)

    testing_size = int(test_size * len(data))

    train_data = data[testing_size:]
    test_data = data[:testing_size]

    for image_path_dict in train_data:
        img = Image.open(image_path_dict['startpath'])
        img = img.convert('L')
        img.save(image_path_dict['trainpath'])
        print("File created {} {}".format(image_path_dict['startpath'],
                                          image_path_dict['trainpath']))

    for image_path_dict in test_data:
        img = Image.open(image_path_dict['startpath'])
        img = img.convert('L')
        img.save(image_path_dict['testpath'])
        print("File created {} {}".format(image_path_dict['startpath'],
                                          image_path_dict['testpath']))


def create_dir(image_dir):
    if not os.path.exists(image_dir):
        print("Directory Created", image_dir)
        os.makedirs(image_dir)
