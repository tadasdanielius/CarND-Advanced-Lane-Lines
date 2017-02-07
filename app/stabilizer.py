import numpy as np

import numpy as np

class Stabilizer:
    """ Smooth out fitted curves by keeping previous fits and averaging out """
    def __init__(self, window=5):
        self.log_left = np.zeros((window, 3))
        self.log_right = np.zeros((window, 3))
        self.count = 0
        self.window = window

    def stabilize(self, left, right, left_err, right_err, ignore_err=True):
        """ stabilize the coff """
        if self.count < self.window:
            self.log_left[self.count,] = left
            self.log_right[self.count,] = right
            self.count += 1
            return left, right

        if left_err < 22426300 or ignore_err is True:
            window = self.window-1
            self.log_left[0:self.log_left.shape[0]-1,] = self.log_left[1:, ]
            self.log_left[window,] = left

        if right_err < 22426300 or ignore_err is True:
            self.log_right[0:self.log_right.shape[0]-1,] = self.log_right[1:, ]
            self.log_right[window,] = right

        left_avg = self.log_left[:,0].mean(), self.log_left[:,1].mean(), self.log_left[:,2].mean()
        right_avg = self.log_right[:,0].mean(), self.log_right[:,1].mean(), self.log_right[:,2].mean()

        return left_avg, right_avg
    
    def avg(self):
        """ return average of collected cofficients """
        left_avg = self.log_left[:,0].mean(), self.log_left[:,1].mean(), self.log_left[:,2].mean()
        right_avg = self.log_right[:,0].mean(), self.log_right[:,1].mean(), self.log_right[:,2].mean()

        return left_avg, right_avg
