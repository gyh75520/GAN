from glob import glob
import os
import scipy.misc
import numpy as np
import random

# data = glob('data_micro/case1/*.jpg')
# print(data)


def get_img(img_path, resize=False):
    image = scipy.misc.imread(img_path).astype(np.float32)
    h, w = image.shape[:2]
    w_s = int((w - h) / 2)
    w_e = int(w_s + h)
    # print(w_s, w_e)
    cropped_img = image[:, w_s:w_e]
    if resize:
        cropped_img = scipy.misc.imresize(cropped_img, [64, 64])
    return np.array(cropped_img) / 255


def get_data_micro():
    datas = []
    for (root, dir, file) in os.walk('data_micro'):
        data = glob(root + '/*.jpg')
        datas = datas + data

    # print('datas size: ', len(datas))
    return datas


def get_batch(batch_size):
    datas = get_data_micro()
    datas_batch = random.sample(datas, batch_size)
    imgs_batch = [get_img(data) for data in datas_batch]

    # imgs_batch = random.sample(imgs, batch_size)
    return imgs_batch


if __name__ == '__main__':
    batch_size = 64
    imgs_batch = get_batch(batch_size)
    print(len(imgs_batch))
    # print(imgs[0].shape)
    print('end')
    # scipy.misc.imsave("tes.png", a)
    # scipy.misc.imshow(a)
