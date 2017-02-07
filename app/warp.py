import numpy as np
import cv2

OFFSET = 250

def get_warp_src(image):
    """ Source matrix """
    src = np.float32([
        [image.shape[1]*0.4475, image.shape[0]*0.65],
        [image.shape[1]*0.5525, image.shape[0]*0.65],
        [image.shape[1]*0.175, image.shape[0]*0.95],
        [image.shape[1]*0.825, image.shape[0]*0.95],
    ])
    return src

def get_warp_dst(image):
    """ Destination matrix """
    dst = np.float32([
        [image.shape[1]*0.2, image.shape[0]*0.025],
        [image.shape[1]*0.8, image.shape[0]*0.025],
        [image.shape[1]*0.2, image.shape[0]*0.975],
        [image.shape[1]*0.8, image.shape[0]*0.975],
    ])
    return dst


def warp(img, src=None, dst=None):
    """ Warp image. Make it bird eye """
    img_size = (img.shape[1], img.shape[0])
    src = src if src is not None else get_warp_src(img)
    dst = dst if dst is not None else get_warp_dst(img)
    Mtx = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Mtx, img_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    """ unwarp image """
    src = get_warp_dst(img)
    dst = get_warp_src(img)
    return warp(img, src, dst)
