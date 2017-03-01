import numpy as np
import os
from PIL import Image
import hashlib

IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']
width, height = 130, 130
img_type = "RGB"
image_ndim = 3


def save_to_np(image_path, name):
    arr = np.array(Image.open(image_path).convert(img_type))
    arr = np.reshape(arr, (width * height, image_ndim))
    np.save(name, arr=arr)


def get_images(dir_path, image_dir='images/', np_dir="numpy/"):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)

    for file_path in os.listdir(dir_path):
        curr = os.path.join(dir_path, file_path)
        if os.path.isdir(curr):
            get_images(curr)
        else:
            file_type = curr.split('.')[-1]
            if file_type in IMAGE_FILE_TYPES:
                try:
                    img = Image.open(curr)
                    hash = hashlib.md5(os.path.splitext(curr)[0]).hexdigest()
                    base_path = str(hash)
                    image_path = image_dir + base_path + '.png'
                    img = img.resize((width, height))
                    img.save(image_path)
                    save_to_np(image_path, np_dir + base_path)
                except Exception:
                    print('exception occured')


def get_multiple_list(dirs):
    for dir in dirs:
        get_images(dir_path=dir,
                   image_dir='images/' + dir,
                   np_dir='output_arrs/' + dir)


get_multiple_list(['white_spot/', 'healthy/'])
# get_multiple_list(['white_spot'])
# # convert_images(paths)
