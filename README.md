## Advanced Lane Finding Project

Final results:

[Project Video 1](https://www.youtube.com/watch?v=ETCjhhBV19Y)

[Project Video 2 Added bird eye view](https://www.youtube.com/watch?v=1HO_AeByt6k)

[Challenge Video](https://www.youtube.com/watch?v=wssWq0_34sg)


### The Goal of this Project

The goal of this project is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

### Steps if this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle
position.

### Camera Calibration

The code of camera calibration is defined in *app/calibration.py*. First time the code will load all provided chessboard images. Using those images it will detect object points which will be the (x, y, z) coordinates of the chessboard corners in the world also image points pixel position of each of the corners in the image plane with each successful chessboard detection. of the chessboard. Then using all output object points and image points it will compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. Here is an example of the distortion correction

![Camera](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image0.png)

This is how undistorted image looks like on the given example image.

![Undistorted](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image1.png)



### Detecting lanes
First we need to create binary image which should approximatly show lane lines. To achieve this results we will use the following transformations:
* Equalize colour histogram
* Convert image to LUV colour space. On L channel apply threshold with min = 250 and max = 255. This is good for detecting white line
* Convert image to LAB colour space. On B channel apply threshold with min = 140 and max = 255. This is good for detecting yellow line
* Convert image to gray scale and apply threshold with min = 245 and max = 255
* All masks are combined into one binary image.

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image2.png)

#### Binary mask combined

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image3.png)

### Bird eye view
The next step is to apply perspective transformation to make the road look like from bird view. The functions which warps image and unwarps are defined in file app/warp.py. Function warp takes image calculates perspective transformation matrix and warps the image. Function unwarp does the same just use different source points and destination points

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image4.png)

Here is an example how warped image will look

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image5.png)

### Detecting lane
The detection process works in the following way:
* Using binary image creates a histogram
* Finds 2 peaks in the histogram. On each side left and right. The image is split at the center. Each peak is a starting point along x axis.
* Histogram peaks I use that as a starting point where to start searching for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. Finally using the pixels we found in the binary image we fit the second degree curve and we have final line.

Here is an example of fitted curve

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image6.png)

*red rectangles indicate invalid area which are ignored*

Finally we need to unwarp back image and place it on the original image. The function unwarp is defined in *app/warp.py* file.

![Binary Mask](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/images/image7.png)

For curve detection I have implemented 3 classes in the file *app/curve.py*:
* CurveWindow which implements functionality of sliding window. Responsible for defining boarders of the given window, can draw rectange on the given image and detect points inside the window. The detect method will also validate if the detected points "make sense" i.e. not to sparse or not too dense.
* CurveWindows holds the list of all sliding windows for a given line.
* Curve which is responsible for creating sliding windows, checking if it meets all requirements and marks either valid or not. Then fits the curve.

## Stabilisation

Not all road parts are nicely fitted. Some of the areas has different colour and cause problems separating lines, therefore some stabilisation technique is used: 

* Slidings windows cannot move too far from the last frame position
* Sliding windows which has less than 50 pixels or more than 20000 pixels will be ignored and previous frame values will be used
* If less than 3 sliding windows are found, then ignore the frame and use previous frame line values
* Fitting cofficients are averaged for 10 frames

In case some rules are invalid (less than 3 windows are valid) then it will roll-back fitted curve from the previous frame.

Curve stabilisation (smoothing logic) is implemented in file `app/curve.py` class `CurveStabiliser`

## Numerical estimation of lane curvature and vehicle position

**Vehicle position** is calculated by averaging each line then calculating average point between both curves. The value is then subtracted from the center (`image_x /2`) and converted to meters by multiplying absolute value by `3.7/700`

```python
# calculate mean for both lines
ploty = np.linspace(0, shape-1, shape)
fitx = self.fit[0]*ploty**2 + self.fit[1]*ploty + self.fit[2]
line_mean fitx.mean()
# ... Some code emitted ...
position = (right_mean+left_mean)/2
dist_from_center = image_width/2 - position
dist= 3.7/700 * abs(dist_from_center)
```


The following code will calculate **Lane Curvature**:

```python
ym_per_pix = 30./720
xm_per_pix = 3.7/700
fit_cr = np.polyfit(points_y*ym_per_pix, points_x*xm_per_pix, 2)
c_rad = ((1 + (2*fit_cr[0]*np.max(points_y) + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```

## Pipeline

Pipline code can be found [here](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/pipeline.ipynb)

Here is final results

[Project Video 1](https://www.youtube.com/watch?v=ETCjhhBV19Y)

[Project Video 2 Added bird eye view](https://www.youtube.com/watch?v=1HO_AeByt6k)

[Challenge Video](https://www.youtube.com/watch?v=wssWq0_34sg)

## Final notes

The project was tricky. Kind of trial and error technique, changing color spaces and adjusting thresholds. The parameters can be tuned to work well on both tracks, but it may fail under the different conditions like during night or different weather conditions. Also, it may fail if other car is close ahead.

I wasn't able to achieve great results for harder challange but I have some ideas which I haven't tried. Currently, sliding windows are stacked vertically and only move horizontally, but harder challenge road is very curved and stacking only vertically loose the line. Moving windows not only horizontally but also adjusting vertically may improve line detection. It is something worth trying in the future.

Some other ideas I haven't fully tried is adaptive brightness control. I noticed that adjusting brightness might work for some parts of the road (e.g. under the bridge) but fail when it's very light or vice versa. I believe that checking the density of the pixels would indicate that it's too bright or too dark therefore increasing/decreasing thresholds runtime could give better results.


Full report can be found [here](https://github.com/tadasdanielius/CarND-Advanced-Lane-Lines/blob/master/report.ipynb)

