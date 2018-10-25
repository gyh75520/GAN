from glob import glob
import os
import scipy.misc
import numpy as np
import random

# data = glob('data_micro/case1/*.jpg')
# print(data)


def _get_img(img_path, resize=False, magnify_interval=False):
    image = scipy.misc.imread(img_path).astype(np.float32)
    h, w = image.shape[:2]
    w_s = int((w - h) / 2)
    w_e = int(w_s + h)
    # print(w_s, w_e)
    cropped_img = image[:, w_s:w_e]
    if resize:
        cropped_img = scipy.misc.imresize(cropped_img, [64, 64])

    cropped_img = np.array(cropped_img) / 255
    if magnify_interval:
        # transform from [0,1] to [-1,1] for generator tanh
        cropped_img = cropped_img * 2 - 1

    return cropped_img


def _get_data_micro():
    datas = []
    for (root, dir, file) in os.walk('data_micro'):
        data = glob(root + '/*.jpg')
        datas = datas + data

    # print('datas size: ', len(datas))
    return datas


def get_batch(batch_size, resize=False, magnify_interval=False):
    datas = _get_data_micro()
    datas_batch = random.sample(datas, batch_size)
    imgs_batch = [_get_img(data, resize, magnify_interval=magnify_interval) for data in datas_batch]

    # imgs_batch = random.sample(imgs, batch_size)
    return imgs_batch


if __name__ == '__main__':
    batch_size = 64
    imgs_batch = get_batch(batch_size, magnify_interval=True)
    print(imgs_batch[0])
    # print(imgs[0].shape)
    print('end')
    # scipy.misc.imsave("tes.png", a)
    # scipy.misc.imshow(a)
