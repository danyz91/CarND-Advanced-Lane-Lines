## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/step1_chessboard.png "Chessboard detection"
[image2]: ./output_images/original_image.jpg "Original"
[image3]: ./output_images/step2_undistort.jpg "Undistortion"
[image4]: ./output_images/step3_threshold.png "Color and Gradient Threshold"
[image5]: ./output_images/step4_perspective_trans.png "Perspective Transform"
[image6]: ./output_images/step5_lane_pixels.png "Lane Pixels"
[image60]: ./output_images/step60_result.jpg "Inverse Warping"
[image7]: ./output_images/step6_result.jpg "Final Result"
[image8]: ./output_images/points_selector_gui.png "Points Selector Gui"
[video1]: ./output_videos/project_video.mp4 "Video"

[img_distortion]: ./output_images/distortion.jpg "Distortion Correction"


### Code Organization
All code for Advanced Lane Finding Project is inside the IPython notebook in
"./AdvancedLaneFindingPipeline.ipynb", form here on referred as "project notebook"

---

### Camera Calibration

#### 1. Camera matrix - Distortion Coefficients computation

The code for this step is contained in **cell 4** of the project notebook. It is wrapped inside the function `calibrate_camera()`.

The function read all images inside the folder specified via the parameter `calib_folder`.
Then it is populated the array `objp`, which stores the (x, y, z) coordinates of the chessboard corners in the world. Here It is assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objpoints` is just a replicated array of coordinates, and `obj` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
In order to detect these, here it is used the OpenCV function `cv2.findChessboardCorners` for each image read from `calib_folder`
In the end are used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Then it has been applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion correction.

To demonstrate this step, in the image below it is presented a test image and its undistorted version.

![alt text][img_distortion]

The code for single image distortion correction is in the **cell 7** of the project notebook, inside the function  `advlf_undistort`

---

#### 2. Gradient and Color Threshold

In order to apply different criteria for gradient and color thresholding in **cell 10** of the project notebook there are different thresholding function:

- `abs_sobel_thresh`: the function applies Sobel operator over x or over y basing on input parameter and apply a two fold bounded threshold over it.
- `mag_thresh`: the function computes x and y gradient and computes the overall gradient magnitude. Then, it applies a two fold bounded threshold over it.
- `dir_threshold`: the function computes x and y gradient and computes the overall gradient direction. Then, it applies a two fold bounded threshold over it.
- `red_channel_threshold`: the function isolates red channel of given image and apply a two fold bounded threshold over it.
- `saturation_channel_threshold`:the function convert the image into HLS color space and isolates saturation channel. Then, it applies a two fold bounded threshold over it.

All these function are taken into account in **cell 11** where they are all combined in function `advlf_gradient_threshold`.
It has been decided to combine three different thresholding criteria:

- Saturation channel thresholding: this is done to select the most saturated pixel independently from their hue
- Direction and Magnitude of gradient: this is done to en light portions of image where gradient is oriented as a lane would and with a given overall intensity
- Gradient over x and over y combined: this is done to select the portion of image where direction changes over x and over y

Final threshold values chosen for this pipeline stage are presented in the table below:

Function |              Min Threshold       |     Max Threshold |
--------------------- | ------------------  | ---------------------
Gradient over X       |  20    | 100
Gradient over Y       |  20  |  100
Magnitude Gradient    |  20 | 100
Direction Gradient    |  0.7 | 1.3
Saturation Threshold  |  170 | 255


Final result is shown in the image below. Here you can notice the individual contribution of the three criteria according to the given legend. Then, on the right it is shown the result of combination of the criteria.

![alt text][image4]

---

#### 3. Perspective Transform

In order to compute the perspective transform two steps are necessary: 

- Source points selection
- Destination points selection

The two steps are detailed in the paragraphs below.

#### Source Points: points_selector script

In order to extrapolate the source points for perspective transform it has been created an helper python script located at "./perpsectiveTransform/points_selector.py"
The script loads an image passed via command line and plot it in an interactive window. In this plot it is possible to click on the image in order to store point location. The purpose of the script is to write out a pickle file with 4 coordinates stored. When the fourth point is selected, the script disconnect the click callback and save the file.
This routine is contained in function `on_click()` from line 46-70 of `points_selector.py` script.
In the image below it is shown the result of the selection of four points on a test image:

![alt text][image8]

The perspective transform routine is contained in **cell 14** of project notebook. 
The function checks if the `load_pickle` parameter is set. If so, the pickle file created with `points_selector.py` script is used. Otherwise an hand coded region of interest is selected.

#### Destination Points

Then destination point on result image are selected with code below:

``` python
    img_size = (img.shape[1], img.shape[0])
    offset = 100
    # For destination points, Choosing some points to be
    # a nice fit for displaying our warped result 
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                                 [img_size[0]-offset, img_size[1]], 
                                 [offset, img_size[1]]])
```

Then the OpenCV functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` are called with source and destination points previously selected.
The result of perspective transform is the showed in image below:

![alt text][image5]

---

#### 4. Lane Pixels Extrapolation

##### Line Object
In order to extract lane pixels, in **cell 16** it is defined a Line object with given parameters:

``` python
class Line():
    #polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]
    #polynomial coefficients for the previous fit
    self.prev_fit = [np.array([False])]
    #radius of curvature of the line in some units
    self.radius_of_curvature = None 
    #x values for detected line pixels
    self.allx = None  
    #y values for detected line pixels
    self.ally = None
    #x pixel to meter conversion factor
    self.xm_per_pix = None
    #y pixel to meter conversion factor
    self.ym_per_pix = None
```

This class is useful to take track of previous and current fit in order to optimize video performance. Then it stores also x and y axes of current detected line.

##### Pixel Detection

The code to extract lane pixels is contained in **cell 18** and **cell 19**.
Here there are two methods for fitting a second order polynomial over thresholded image.
The first detection method is based on the extraction of the whole histogram of the image. Then the most excited sections of the histogram are used to start fitting the polynomial using a sliding window algorithm.

The second detection method is basing on the extraction of section around lines already detected, for example in the previous frame of a video stream. This second method is indeed used during video processing to avoid searching over the whole image.

Then a sliding window algorithm is applied over the detection.
Starting from the bottom of the image a sliding window is setup over the first detection. The its position arises to the top of the image and new detection position is checked in order to move the window to the most excited section. This process is repeated until the top of the image is reached.

In the image below, the whole process is presented. Starting from perspective transform image, to histogram extraction and the the sliding window algorithm result.

![alt text][image6]

---

#### 5. Curvature Radius and Vehicle Position

##### Curvature Radius
The code to compute the curvature of a line is contained in **cell 17** of project notebook, in the function `measure_real_curvature()`.
In this function, firstly there are computed scale factor from pixel to real world using region of interest size and some fixed parameters:

- `MAX_LONG_LENGTH`: max longitudinal lane length taken from camera, in meters
- `MAX_LAT_LENGTH`: max lateral lane width taken from camera, in meters
- `PERP_TRANS_LONG_PIXELS`: longitudinal result of perspective transform
- `PERP_TRANS_LAT_PIXELS`: lateral result of perspective transform

Then a `np.poly_fit()` routine is called in order to fit values in real space. Then curvature is computed according to the formula:

``` python
    given
        f(y) = A*y^2 + B*y + C
    then 
        R_curve = ((1+(2Ay+B)^2)^3/2)/2*abs(A)
```

##### Vehicle Position

Vehicle position computation is performed in **cell 23** of project notebook.
Here basing on left line and right line detected in a given frame vehicle position is computed.
The formula take the x axis center of the image as reference and compute the displacement of two lines detected. Then this value is scaled via real value scale factor

---

#### 6. Unwarped Result back on original Image

In **cell 24** and **cell 25** there are drawing functions to get final result.
In `draw_result()` function the inverse warp is performed using `np.linalg.inv(M)` routine of numpy library.
The result is presented in the image below.

![alt text][image60]

Then in `draw_final_result()` function the previous plot is enriched with a transparent black overlay where lane pixels detection is shown on the right and text output with curvature and vehicle position computed per frame is presented.

The final result is shown in the image below. 

![alt text][image7]

---

### Pipeline (video)

In order to apply the pipeline to a video stream a `LaneFindingPipeline` class is created.
The code of this class is contained in **cell 27**.
This class has the role of initialize the pipeline by calibrating the camera and to choose which of the two detection algorithm use.
It firstly searches over the whole image and then uses the attention mechanism to speed up the computation.

Here's a video showing the result of the whole pipeline. Enjoy it!

 ![alt text][video1]

---

### Discussion

#### Known Issues

- Perspective transform needs to be updated with better projection in order to preserve all geometrical properties of lines.
- Thresholding functions needs to be more robust in handling poorly lighted conditions

#### Open Points

As presented in this project, the pipeline needs to take into account robustness more deeply. This could be done via:

- Check geometrical conditions of the two lane detected in order to classify a bad detection
- Check between successive frames in order to discard bad detections and preserve overall performance

