import numpy as np
import cv2
import glob
import _pickle as pickle

class Camera_calibrator:
    """ Camera calibration class is responsibole for calibrate and udistort images """
    def undistort(self, img):
        """ This method undistorts the image using cv2 library API """
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def initialize(self, filename='calibration.pkl', corners=(9,6), cal_image_loc='camera_cal/*.jpg'):
        """ Initialize calibrator. It will try to load calibration.pkl first, if not found then use provided images to detect corners and and calculate points """
        try:
            # try to load it from file
            self.load_calibration_points(filename)
        except:
            # in case file not found then calibrate and save
            _, _ = self.detect_corners(corners=corners, cal_image_loc=cal_image_loc)
            self.save_calibration_points(filename=filename)

    def save_calibration_points(self, filename='calibration.pkl'):
        obj = {"imgpoints": self.imgpoints, "objpoints": self.objpoints}
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def load_calibration_points(self, filename='calibration.pkl'):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            self.imgpoints = obj['imgpoints']
            self.objpoints = obj['objpoints']

    def detect_corners(self, corners=(9,6), cal_image_loc='camera_cal/*.jpg'):
        objp = np.zeros((np.product(corners),3), np.float32)
        objp[:,:2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(cal_image_loc)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, detected_corners = cv2.findChessboardCorners(gray, corners, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(detected_corners)

        self.imgpoints = imgpoints
        self.objpoints = objpoints

        return imgpoints, objpoints
