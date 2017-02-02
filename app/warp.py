import numpy as np
import cv2

def get_warp_src(img):
    """ Source matrix """
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    return src

def get_warp_dst(img):
    """ Destination matrix """
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    return dst


def warp(img, src=None, dst=None):
    """ Warp image. Make it bird eye """
    img_size = (img.shape[1], img.shape[0])
    src = src if src is not None else get_warp_src(img)
    dst = dst if dst is not None else get_warp_dst(img)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    """ unwarp image """
    src = get_warp_dst(img)
    dst = get_warp_src(img)
    return warp(img, src, dst)
