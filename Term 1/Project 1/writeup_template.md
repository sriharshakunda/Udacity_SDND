# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

# 1. Making a Pipeline that finds lane lines on the road

# 2. Description

[//]: # (Image References)

[image1]: ./results/solidWhiteCurve.jpg "solidWhiteCurve"
[image2]: ./results/solidWhiteRight.jpg "solidWhiteRight"
[image3]: ./results/solidYellowCurve.jpg "solidYellowCurve"
[image4]: ./results/solidYellowCurve2.jpg "solidYellowCurve2"
[image5]: ./results/solidYellowLeft.jpg "solidYellowLeft"
[image6]: ./results/whiteCarLaneSwitch.jpg "whiteCarLaneSwitch"

---

### Reflection

## 1. Describe the pipeline
###    My pipeline was divided into two forms. One for videos and one for Images.
###    For Images, I first converted the image into grayscale using the "grayscale" fucntion. Using Grayscale output, the image was passed into Gaussian Blur and Canny Edge functions.
###    Using Region of Interest Function, by defining the vertices on Canny Edge output image, the image was cropped. This was passed into hough transform function.
###	  Once the lines are detected, the data is passed through draw_lines function inorder to extrapolate the lane lines. For extrapolation I calculated the slope and used the slope and intercept values to find the next possible point at any givin point.
###    Similar Pipeline was used for Videos. However, a HSV filter was added to the video processing and the grayscale output is subdued to make it compatible with the challenge video.

## 2. Identify any shortcomings
###    One of the shortcoming of my code is not robust enough to detect the lines if they are faded or slightly visible. I had to modify the draw_lines function quite a bit to get the extrapolation of lane lines.
###    The other shorcoming is that I have to modify the vertices for ROI function to adapt for challenge video as the lane lines are not in the center of the video frame.

## 3. Suggest possible improvements
### For now there are no improvements. If I find any I will add them here.
