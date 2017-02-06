import numpy as np
import cv2

class CurveStabilizer:
    """ Smooth out fitted curves by keeping previous fits and averaging out """
    def __init__(self, frames=10):
        self.log = np.zeros((frames, 3))
        self.count = 0
        self.frames = frames

    def stabilize(self, fit):
        """ stabilize the coff """
        if self.count < self.frames:
            self.log[self.count,] = fit
            self.count += 1
            return fit

        frames = self.frames-1
        self.log[0:self.log.shape[0]-1,] = self.log[1:, ]
        self.log[frames,] = fit

        avg = self.log[:,0].mean(), self.log[:,1].mean(), self.log[:,2].mean()
        return avg

class CurveWindow:
    """ Represents single window """
    def __init__(self, shape, center, base_range, margin = 30, pos = 0, total = 9):
        self.center = center
        self.margin = margin
        self.nwindows = total
        self.shape = shape
        self.valid = False
        self.pos = pos
        self.base_range = base_range
        self.reason = 0

    def boundaries(self):
        """ Calculate boundaries of given window nr """
        img_y, _ = self.shape
        height = np.int(img_y/self.nwindows)
        ylow = img_y - (self.pos + 1)*height
        yhigh = ylow + height
        xlow = self.center - self.margin
        xhigh = self.center + self.margin
        return ylow, xlow, yhigh, xhigh

    def draw(self, img, color=(0, 255, 0)):
        """ Draw rectange on the given image """
        y1, x1, y2, x2 = self.boundaries()
        
        #print('color {} boundaries {}'.format(color, self.boundaries()))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img,str(self.reason), (x1+15,y1+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 4)

    def detect(self, img, minpix=40, maxpix=10000):
        min_base, max_base = self.base_range
        """ Check if current window has enough points to consider as a valid """
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ylow, xlow, yhigh, xhigh = self.boundaries()
        good_inds = (
            (nonzeroy >= ylow) & (nonzeroy < yhigh) &
            (nonzerox >= xlow) & (nonzerox < xhigh)).nonzero()[0]
        # If you found > minpix pixels, recenter next window on their mean position
        # print('window {}, has {} points, coords {}'.format(self.pos, len(good_inds), self.boundaries()))
        if len(good_inds) > minpix and len(good_inds) < maxpix:
            new_center = np.int(np.mean(nonzerox[good_inds]))
            if min_base < new_center < max_base:
                self.center = new_center
                self.valid = True
            else:
                self.reason = 4
                self.valid = False
        else:
            self.reason = 2
            if len(good_inds) < minpix:
                self.reason = 1
            else:
                self.reason = 3
            
            self.valid = False
        return good_inds, self.center, self.valid


class CurveWindows:
    """ Image split into small windows where to look for the points """
    def __init__(self, img, center, base_range, margin=30, nwindows=9, minpix=50, maxpix=20000, centers=None):
        # Initial place from where to start to place windows
        self.center = center
        # How wide windows should be
        self.margin = margin
        # Vertically how many windows should be
        self.nwindows = nwindows
        # Image on which to operate
        self.img = img
        # List of all windows
        self.windows = []
        # List of valid windows
        self.valid_windows = np.zeros(nwindows).astype(np.bool)
        # Width of the window
        self.minpix = minpix
        # Maximum number of pixels in window
        self.maxpix = maxpix

        self.centers = centers
        
        self.base_range = base_range
        #Points
        self.points_x = None
        self.points_y = None
        self._initialize()

    def _initialize(self):
        nonzero = self.img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        inds = []
        for i in range(self.nwindows):
            center = self.center
            if self.centers is not None:
                center = self.centers[i]
            win = CurveWindow(self.img.shape, center, self.base_range, self.margin, i, self.nwindows)
            good_inds, center, valid = win.detect(self.img, minpix=self.minpix, maxpix=self.maxpix)
            self.valid_windows[i] = valid
            if valid is True:
                self.center = center
                inds.append(good_inds)
            self.windows.append(win)
        if len(inds) > 0:
            curve_inds = np.concatenate(inds)
            self.points_x = nonzerox[curve_inds]
            self.points_y = nonzeroy[curve_inds]
            self.valid = True
        else:
            self.valid = False
    
    def get_centers(self):
        win_centers = np.zeros(self.nwindows)
        for i in range(self.nwindows):
            win = self.windows[i]
            win_centers[i] = int(win.center)
        return win_centers.astype(int)

    def draw_rect(self, img, valid_color=(0, 255, 0), invalid_color=(255, 0, 0)):
        """ Draw all windows on the given image """
        for i in range(self.nwindows):
            win = self.windows[i]
            color = valid_color if win.valid is True else invalid_color
            win.draw(img, color=color)

class Curve:
    """ Responsible for detecting curve """
    def __init__(self, center, base_range, minpix=40, threshold=3, margin=70, centers=None):
        # Initial place from where to start looking for curve
        self.center = center
        # Is curve currently correctly identified
        self.invalid = True
        # Fit cofficients
        self.fit = np.array([0, 0, 0])
        # Minimum points required to treat this as a valid curve
        self.minpix = minpix
        # Minimum valid windows required to be considered as valid curve
        self.threshold = threshold
        # How wide window should be
        self.margin = margin
        # Min base
        self.base_range = base_range
        # just initialized
        self.initialized = False
        # Centers
        self.centers = None
        # For submission use 8
        self.stabilizer = CurveStabilizer(10)
        self.curve_valid_win = None

    def adjust_center(self, center):
        """ Center gives approvimate location where to start looking for the curve """
        self.center = center

    def draw_curve(self, img, color=(255, 0, 0), thick=10):
        """ Draw fitted curve on the given image """
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        fitx = self.fit[0]*ploty**2 + self.fit[1]*ploty + self.fit[2]

        for idx, _ in enumerate(ploty):
            posx = int(fitx[idx])
            cv2.line(img, (posx-thick, idx), (posx+thick, idx), color)



    def curve_fit(self, img):
        """ Find curve in the given image """
        win = CurveWindows(img, self.center, self.base_range, minpix=self.minpix, margin=self.margin, centers=self.centers)
        self.curve_windows = win

        if win.valid is False:
            return False

        valid_windows = win.valid_windows.sum()
        if valid_windows < 3 and self.initialized is True:
            return False

        if len(win.points_x) < 10:
            return False

        self.centers = win.get_centers()
        c_fit = np.polyfit(win.points_y, win.points_x, 2)
        
        min_base, max_base = self.base_range
        max_y = img.shape[0]-10
        fitx = np.int(c_fit[0]*max_y**2 + c_fit[1]*max_y + c_fit[2])
        # Reject if center jumped too far
        if abs(fitx-self.center) > 100 and self.initialized is True:
            #print('base jumpeddd too far')
            return False
        if min_base < fitx < max_base:
            self.fit = self.stabilizer.stabilize(c_fit)
            #self.base_range = (fitx-250, fitx+250)
            self.center = fitx
            self.initialized = True
            self.curve_valid_win = win
            return True
        else:
            return False
    
    def fit_avg(self, shape):
        ploty = np.linspace(0, shape-1, shape)
        fitx = self.fit[0]*ploty**2 + self.fit[1]*ploty + self.fit[2]
        return fitx.mean()
        

def plot_lanes(img, left_fit, right_fit, unwarped_shape):
    """ plot lanes on image """
    lanes_img = np.zeros_like(img)
    ploty = np.linspace(0, unwarped_shape[0]-1, unwarped_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    for idx, _ in enumerate(ploty):
        leftx = int(left_fitx[idx])
        rightx = int(right_fitx[idx])
        cv2.line(lanes_img, (leftx+10, idx), (rightx-10, idx), (0,255,0))

    return lanes_img