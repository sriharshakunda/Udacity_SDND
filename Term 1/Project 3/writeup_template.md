# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Nvidia Deep Learning Architecture"
[image2]: ./examples/nvidia_arch.png "Final Model Architecture"
[image3]: ./examples/sample.jpg "Sample Data"
[image4]: ./examples/forward.jpg  "Forward Data"
[image5]: ./examples/reverse.jpg  "Reverse Data"
[image6]: ./examples/mse.png "MSE Loss"
[image7]: ./examples/epochs.png "Training Results"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is based on [End to End Deep Learning for Self Driving Cars Paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) by NVIDIA.
Nvidia Deep Learning Architecture
![alt text][image1]

#### 2. Attempts to reduce overfitting in the model
I have used couple of dropout layers with a probability of 0.5. This reduced overfittin of the model. I also reduced number of epochs to avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  The following sets of training data was used for this project:
1. Udacity sample data
2. 2 Laps of Forward Driving
3. 3 Laps of Reverse Driving

Initially I trained the model with sample data and forward driving data and coludnt make the car drive inside the yellow lines for 1/4th distance of total track.
Reverse data definitely helped the training to generate a model, where the car stayed withing track all the time.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I have tried both LeNet -5 and Nvidia's models to understand the difference. LeNet-5 was easy to compute, however there was more parameter tuning involved to keep the car within track.
Nvidia model took a little more time with dropoout layers, but the results were consistent and there was less manual tuning involved.

Data was split into 80% training and 20% validation set's. The images were cropped on the top(to remove sky) and bottom (to remove bumper) and resized to (66,200).

I initially followed the lectures and started building by network as suggested in the videos.
However, I experimented with a large training data set and small training data set. They key for the network is to have a data set with variable driving sistuations.
Once I started feeding in the reverse data set, the difference between a program that uses just forward driving condition and a program that uses both forward and reverse driving condition is significantly different.

Once my model started working, I had tune the number of epochs to avoid overfitting.
I observed when the network is undertrained, the car was moving all the way to left and when over trained, the car moves all they way towards right.
I did try to use a steering offset value to compensate for undertrained or overtrained network, however having variable data sets was the key to my program.



#### 2. Final Model Architecture

The below image shows my final model architecture.

![alt text][image2]

#### 3. Creation of the Training Set & Training Process
The data set was created by merging all three camera images. This was performed on each data set. i.e sample, forward driving and reverse driving data sets.
A list was created consecutively by appending the images from each data set and also their corresponding steering values.
A numpy array was created with variables X_train (Image data) and y_train (steering values)  inorder to use it for training the network.

Sample Data Set (Center Camera)
![alt text][image3]

Forward Driving Data Set (Center Camera)
![alt text][image4]
Reverse Driving Data Set(Center Camera)
![alt text][image5]

The traing process was straight forward. Once the training data was collected with respective images and steering values, they were fed through the network.
I trained the network for three epochs


#![alt text][image6]
#![alt text][image7]

