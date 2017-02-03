import numpy as np
import matplotlib.pyplot as plt

class LaneWindow:
    """ Lane sliding window """
    def __init__(self, binary, nwindows):
        self.binary = binary
        self.nwindows = nwindows

    def boundaries(self, window, margin, base):
        """ Identify window boundaries in x and y (and right and left) """
        img_y, _ = self.binary.shape
        window_height = np.int(img_y/self.nwindows)
        leftx_current, rightx_current = base
        win_y_low = img_y - (window+1)*window_height
        win_y_high = img_y - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        return ((win_y_low, win_y_high),
                (win_xleft_low, win_xleft_high),
                (win_xright_low, win_xright_high))

    def detect_in_window(self, window, margin, base, minpix=50):
        """ Identify the nonzero pixels in x and y within the window """

        leftx_current, rightx_current = base

        nonzero = self.binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        win_y, win_xleft, win_xright = self.boundaries(window, margin, base)
        win_y_low, win_y_high = win_y
        win_xleft_low, win_xleft_high = win_xleft
        win_xright_low, win_xright_high = win_xright

        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        return good_left_inds, good_right_inds, leftx_current, rightx_current



class LaneDetector:
    """ Lane detector will take binary image and find lanes """
    def __init__(self, binary_img, nwindows=9, margin=100, minpix=50):
        self.binary = binary_img
        self.__histogram__()
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.window_height = np.int(self.binary.shape[0]/self.nwindows)
        self.nonzero = self.binary.nonzero()

        self.windows = ([], [])

        #self.left_lane_inds = []
        #self.right_lane_inds = []
        self.lane_inds = ([], [])

    def __histogram__(self):
        """Build histogram from image. Useful to find peaks where lanes might be"""
        self.histogram = np.sum(self.binary[self.binary.shape[0]/2:, :], axis=0)
        self.midpoint = np.int(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:self.midpoint])
        rightx_base = np.argmax(self.histogram[self.midpoint:]) + self.midpoint

        self.bases = (leftx_base, rightx_base)

    def find_lanes(self, degree=2):
        """ Find lanes """
        window = LaneWindow(self.binary, self.nwindows)
        base = self.bases
        left_lane_inds = []
        right_lane_inds = []

        nonzero = self.binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        for window_idx in range(self.nwindows):
            gli, gri, leftx, rightx = window.detect_in_window(
                window_idx, self.margin, base, self.minpix)
            base = (leftx, rightx)
            # Append these indices to the lists
            left_lane_inds.append(gli)
            right_lane_inds.append(gri)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, degree)
        right_fit = np.polyfit(righty, rightx, degree)
        return left_fit, right_fit, left_lane_inds, right_lane_inds

def plot_lanes(binary, left_fit, right_fit, left_lane_inds, right_lane_inds):
    """ Plot lane lines """
    out_img = np.dstack((binary, binary, binary))*255
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow', linewidth=10)
    plt.plot(right_fitx, ploty, color='yellow', linewidth=10)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    return out_img
