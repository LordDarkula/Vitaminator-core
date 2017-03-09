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
            filehash = np.array(Image.open(curr)).tolist()
            if filehash not in unique:
                unique.append(filehash)
                items.append(curr)
            else:
                file = items[unique.index(filehash)]
                print "Duplicates Found"
                os.remove(curr)


def save_to_np(image_path, name):
    arr = np.array(Image.open(image_path).convert(img_type))
    arr = np.reshape(arr, (width * height, image_ndim))
    np.save(name, arr=arr)


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
                    # save_to_np(image_path, np_dir + base_path)
                except Exception:
                    print('exception occured')


def hash_name(curr):
    return hashlib.md5(os.path.splitext(curr)[0]).hexdigest()


def get_multiple_list(dirs):
    for dir in dirs:
        get_images(dir_path=dir,
                   image_dir='new_images/' + dir,
                   np_dir='output_arrs/' + dir)


if __name__ == '__main__':
    # get_multiple_list(['healthy_nails_news/'])
    # get_multiple_list(['Leukonychia _ Google Search/', 'calcium deficiency nails _ Google Search/',
    #                    'healthy_natural_nails/', 'leukonychia_nail_new/',
    #                    'natural_nails_new/', 'new_images_white_spot/', 'white_spot_on_nails_new/'])
    # remove_duplicates('images/')
    print os.listdir('new_images/healthy_nails_news/')
    list = ['042969c6af5d001b0b39d41d55eed31f.jpg', '149aba9425fda1e1f97b83746cda3165.jpg',
            '152bd7b6e3957bda5b0c450208209d03.jpg', '17475578e619a7c7b55cc6c9e6ce780e.jpg',
            '174907f7f949e5f778329aa0d30a7902.jpg', '174a98209ddbba43d5ab408c218856b3.jpg',]
    # get_multiple_list(['white_spot_nails/', 'healthy/'])
    # get_multiple_list(['white_spot_nails'])
    # convert_images(paths)
