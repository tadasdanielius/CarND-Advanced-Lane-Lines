import cv2
import numpy as np

class Edges:
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        self.combined = np.zeros_like(self.gray)

    def gradient(self, thresh_min = 20, thresh_max = 100, dx=1, dy=0, combine=True):
        img = self.img
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, dx, dy)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        self._combine(binary, combine)
        return binary

    def gradient_x(self, thresh_min = 20, thresh_max = 100, combine=True):
        img = self.gradient(
            thresh_min=thresh_min, 
            thresh_max=thresh_max, 
            dx=1, dy=0,
            combine=combine)
        return img

    def gradient_y(self, thesh_min = 20, thresh_max = 100, combine=True):
        img = self.gradient(
            thresh_min=thesh_min, 
            thresh_max=thresh_max, 
            dx=0, dy=1,
            combine=True)
        return img
    
    def gradient_color_channel(self,thresh_min=170, thresh_max=255, channel=2, combine=True):
        img_channel = self.hls[:,:,channel]
        s_thresh_min = 170
        s_thresh_max = 255
        binary = np.zeros_like(img_channel)
        binary[(img_channel >= s_thresh_min) & (img_channel <= s_thresh_max)] = 1

        self._combine(binary, combine)
        return binary

    def _combine(self, binary, combine=True):
        if combine:
            combined = self.combined
            combined[(combined == 1) | (binary == 1)] = 1
            self.combined = combined

    def laplacian(self, thresh_min=15, thresh_max=255, combine=True):
        l_img = cv2.Laplacian(self.gray,cv2.CV_64F)
        binary = np.zeros_like(l_img)
        binary[(l_img >= thresh_min) & (l_img <= thresh_max)] = 1
        self._combine(binary, combine)
        return binary
