# AI Programming with Python Nanodegree

This Repo Contains the Projects I have done for the [AI Programming with Python Nanodegree Program](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) Offered By [Udacity](https://www.udacity.com/).This Course was Sponsorsed by AWS for Qualifing [Student DeepRacer](https://student.deepracer.com).



## Project 1: Pre-trained Image Classifier to Identify Dog Breeds

### Overview

This Project dealt with Identifying Dog Breeds from Images using Pre-trained Image Classifier.

### Principle Objective

* Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images aren't of dogs.

* Correctly classify the breed of dog, for the images that are of dogs.

* Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve objectives 1 and 2.

* Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative  solution would have given a "good enough" result, given the amount of time each of the algorithms takes to run.

### Outcomes

This Project main objective is to make ourselves familiar with python,neural networks and Necessary Skill for machine such as inference,Looking at Metrics,etc.

## Project 2: Create Your Own Image Classifier

### Overview

This Project dealt with Creating Your Own Image Classifier From Scratch.Task is to Classify 102 Different Species of Flowers.

By submitting this project, I have demonstrated the ability to build and train a neural network from scratch on a real-world image dataset.Through This Project I was inculcated with the skills of building a neural network from scratch, training it on a dataset, and testing its performance on new images using Pytorch.

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

#### Load and preprocess the image dataset

The first step is to load the data. I used the [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) function to create a dataset from the image folders. I used the [transforms](https://pytorch.org/vision/0.8/transforms.html) module to define the following data transformations:

Following all Transforms are applied with Normalization

* Training: Randomly rotate, crop, and flip the images
* Validation: Crop the images to a square and resize them to 224x224 pixels
* Testing: Crop the images to a square and resize them to 224x224 pixels


#### Train the image classifier on your dataset

Here we used Pretrained Model [VGG16](https://arxiv.org/abs/1409.1556v6) and Freeze the Parameters and then Train the Classifier.This Process is called Transfer Learning.**Why Transfer Learning**? Because we don't have to train the whole model from scratch and we can use the pretrained model to train our classifier.So We are using Pretrained Model to Extract Features and then Train the Classifier.

Through This Step I was able Understand Backpropagation,Loss Function,Optimizer,Hyperparameters,etc.How to use GPU for Training.and all other necessary skills for training a neural network.

#### Use the trained classifier to predict image content

Here we used the trained model to predict the class for an input image. We also calculated the model's accuracy on the test dataset.



# Key Learnings from Doing These Projects

* I was able Familiarize with Python and its Libraries.
* I was able to Understand Neural Networks and how to train them.
* I was able to Understand Transfer Learning and how to use it.
* I was able to Understand how to use GPU for Training.
* I was able to Understand how to use [Pytorch](https://pytorch.org) for Training a Neural Network,Testing it,Transforming Images,etc.

