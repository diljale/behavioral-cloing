#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also noticed that car had trouble staying on road near bridge so i had to collect data specifically for the bridge

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the tutorials and try each lab to understand principles behind different techniques.

My first step was to use a convolution neural network model similar to the LeNet to get baseline for model and also to verify my setup for training and validating a model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I also normalized and cropped the image to get better training accuracy.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I started using all 3 cameras and flipping images. I then collected more training data at curves and near bridge as car was driving off the road in these regions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
1] 5x5x24 Convolution layer
2] 5x5x36 Convolution layer
3] 5x5x36 Convolution layer
4] Dropout layer to drop50% samples
5] 3x3x64 Convolution layer
6] Flatten layer
7] Fully connected layer with 200 neurons output
8] Fully connected layer with 100 neurons output
9] Fully connected layer with 50 neurons output
10] Fully connected layer with 10 neurons output
11] Fully connected output layer

####3. Creation of the Training Set & Training Process

I started with sample training data provided in Project resources. I found my model doing well on most parts of road except some cruves and bridge. So i collected some data where i would steer to left and record the data of correctly steering back to center of road, same for bridge.

I used all 3 camera images to get better accuracy

After the collection process, I had 40,000 number of data points. I then preprocessed this data by normalizing the image and cropping top and bottom part of image to get roads and curbs


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as validation loss started increasing after 7 epochs