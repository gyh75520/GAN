def plot(samples):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        # ax = plt.subplot(gs[i])
        plt.axis('off')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_aspect('equal')
        plt.imshow(sample)
    plt.savefig('{}.png'.format(str(223).zfill(3)))
    # plt.show()
    return fig


def RGB2BGR(img):
    # opencv读取图片的默认像素排列是BGR
    return img[..., ::-1]


def save_img(dir, iter, imgs, boxs=None, magnify_interval=False):
    # from scipy import misc
    # for i, sample in enumerate(samples):
    #     misc.imsave('{}/{}_{}.png'.format(dir, i, iter), sample)
    import cv2
    if magnify_interval:
        imgs = (imgs + 1.) / 2.  # inverse transform from [-1,1] to [0,1]

    if boxs is not None:
        j = 0
        for img, box in zip(imgs, boxs):
            img = img * 255
            img = RGB2BGR(img)

            width = 640
            height = 480
            img = img.copy()
            # print(box)
            for i in range(len(box) // 4):
                xmin = int(box[0 + i * 4] * width)
                ymin = int(box[1 + i * 4] * height)
                xmax = int(box[2 + i * 4] * width)
                ymax = int(box[3 + i * 4] * height)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imwrite('{}/{}_{}.png'.format(dir, j, iter), img)
            j += 1
    else:
        for i, img in enumerate(imgs):
            img = img * 255
            img = RGB2BGR(img)
            cv2.imwrite('{}/{}_{}.png'.format(dir, i, iter), img)


if __name__ == '__main__':
    from datas import get_batch

    batch_size = 4
    imgs_batch, box_batch = get_batch(batch_size, crop_img=False, magnify_interval=False)
    print(len(box_batch))
    # save_img('test', 2, imgs_batch)
    # import cv2
    # from scipy import misc
    # misc.imsave('001_misc.png', imgs_batch[0])

    # img = imgs_batch[0] * 255
    # img = img[..., ::-1]
    # img = img.copy()
    # cv2.rectangle(img, (212, 300), (290, 300), (0, 255, 0), 2)
    # cv2.imwrite('001_cv.png', img)

    save_img('test', 1, imgs_batch, box_batch)
    save_img('test', 2, imgs_batch)
