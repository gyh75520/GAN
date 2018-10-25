from glob import glob
import os
import scipy.misc
import numpy as np
import random

# data = glob('data_micro/case1/*.jpg')
# print(data)


def _get_img(img_path, crop_img=True, resize=False, magnify_interval=False):
    image = scipy.misc.imread(img_path).astype(np.float32)
    if crop_img:
        h, w = image.shape[:2]
        w_s = int((w - h) / 2)
        w_e = int(w_s + h)
        # print(w_s, w_e)
        image = image[:, w_s:w_e]
    if resize:
        image = scipy.misc.imresize(image, [64, 64])

    image = np.array(image) / 255
    if magnify_interval:
        # transform from [0,1] to [-1,1] for generator tanh
        image = image * 2 - 1

    return image


def _get_data_micro():
    datas = []
    for (root, dir, file) in os.walk('data_micro'):
        data = glob(root + '/*.jpg')
        datas = datas + data

    # print('datas size: ', len(datas))
    return datas


def get_batch(batch_size, crop_img=True, resize=False, magnify_interval=False):
    datas = _get_data_micro()
    datas_batch = random.sample(datas, batch_size)
    imgs_batch = [_get_img(data, crop_img=crop_img, resize=resize, magnify_interval=magnify_interval) for data in datas_batch]

    # imgs_batch = random.sample(imgs, batch_size)
    return imgs_batch


if __name__ == '__main__':
    batch_size = 64
    imgs_batch = get_batch(batch_size, crop_img=False, magnify_interval=True)
    print(imgs_batch[0].shape)
    # print(imgs[0].shape)
    print('end')
    # scipy.misc.imsave("tes.png", a)
    # scipy.misc.imshow(a)
