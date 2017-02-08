import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_lanes(img, left_fit, right_fit, unwarped_shape, fill=True):
    """ plot lanes on image """
    lanes_img = np.zeros_like(img)
    ploty = np.linspace(0, unwarped_shape[0]-1, unwarped_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    for idx, _ in enumerate(ploty):
        leftx = int(left_fitx[idx])
        rightx = int(right_fitx[idx])
        cv2.line(lanes_img, (leftx-10, idx), (leftx+10, idx), (255,255,0))
        cv2.line(lanes_img, (rightx-10, idx), (rightx+10, idx), (255,255,0))

        if fill is True:
            cv2.line(lanes_img, (leftx+10, idx), (rightx-10, idx), (0,255,0))

    return lanes_img
