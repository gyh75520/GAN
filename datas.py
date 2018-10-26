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
    # print(image)
    if magnify_interval:
        # transform from [0,1] to [-1,1] for generator tanh
        image = image * 2 - 1

    return image


def _reverse_img(img, crop_img=True, resize=False, magnify_interval=False):
    return 0


def _get_box(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # print(width, height)
    box = []
    for object in root.findall('object'):
        xmin = int(object.find('bndbox').find('xmin').text) / width
        ymin = int(object.find('bndbox').find('ymin').text) / height
        xmax = int(object.find('bndbox').find('xmax').text) / width
        ymax = int(object.find('bndbox').find('ymax').text) / height
        box += [xmin, ymin, xmax, ymax]
    return box


def _get_data_micro():
    imgs_adr = []
    boxs_adr = []
    for (root, dir, file) in os.walk('data_micro'):
        imgs = glob(root + '/*.jpg')
        # print(img)
        boxs = [img.split('.')[0] + '.xml' for img in imgs]
        imgs_adr += imgs
        boxs_adr += boxs
    # print(imgs_adr[1], boxs_adr[1])
    return imgs_adr, boxs_adr


def get_batch(batch_size, crop_img=True, resize=False, magnify_interval=False):
    imgs_adr, boxs_adr = _get_data_micro()
    # datas_batch = random.sample(imgs_adr, batch_size)
    batch_index = random.sample(range(len(imgs_adr)), batch_size)

    # print(imgs_adr[batch_index[0]])
    # print(boxs_adr[batch_index[0]])
    imgs_batch = [_get_img(imgs_adr[index], crop_img=crop_img, resize=resize, magnify_interval=magnify_interval) for index in batch_index]
    box_batch = [_get_box(boxs_adr[index]) for index in batch_index]
    # imgs_batch = random.sample(imgs, batch_size)
    return imgs_batch, box_batch


if __name__ == '__main__':
    _get_box('/Users/howard/Desktop/GAN/data_micro/case1/88.xml')

    batch_size = 64
    imgs_batch, box_batch = get_batch(batch_size, crop_img=False, magnify_interval=True)
    print(len(imgs_batch))
    # print(imgs_batch[0].shape)
    # print(imgs[0].shape)
    print('end')
    # scipy.misc.imsave("tes.png", a)
    # scipy.misc.imshow(a)
