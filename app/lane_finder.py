import numpy as np

class Lane_detector:
    def __init__(self,binary_img, nwindow=9, margin=100, minpix=50):
        self.binary = binary_img
        self.__histogram__()
        self.nwindow = nwindow
        self.margin = margin
        self.minpix = minpix
        self.window_height = np.int(self.binary.shape[0]/self.nwindow)
        self.nonzero = self.binary.nonzero()

        self.left_windows = []
        self.right_windows = []

        self.left_lane_inds = []
        self.right_lane_inds = []

    def __histogram__(self):
        self.histogram = np.sum(self.binary[self.binary.shape[0]/2:,:], axis=0)
        self.midpoint = np.int(self.histogram.shape[0]/2)
        self.leftx_base = np.argmax(self.histogram[:self.midpoint])
        self.rightx_base = np.argmax(self.histogram[self.midpoint:]) + self.midpoint

    def __find_lanes__(self):

        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        margin = self.margin

        y, x = self.binary.shape
        for window in range(self.nwindow):
            win_y_low = y - (window+1) * self.window_height
            win_y_high = y - window*self.window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            self.left_windows.append(
                ((win_xleft_low, win_y_low),
                (win_xleft_high,win_y_high))
            )
            self.right_windows.append(
                ((win_xright_low, win_y_low),
                (win_xright_high,win_y_high))
            )




    


    