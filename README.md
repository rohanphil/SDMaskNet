# SDMaskNet (Social Distancing and Facial Mask detector)
A Social Distancing and Facial Mask Monitor using Tensorflow 

# Overview
This project is inspired by pyimagesearch's covid-19 social distancing project for which I have included the link below.

https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector

The major difference is the use of a mask detector as well. The mask detector is built on a MobileNet2 base-model on which a basic network is built using average pooling.
The social distancing monitor is built using a yolo model with the centroids of the ROIs serving as the discriminating factor to measure distance between individuals.
However, the camera calibration will vary from camera to camera, meaning the number of pixels required to classify two individuals as being in violation of social distance norms,
will vary drastically.

This metric can be adjusted in the sdmask/social_distancing_config.py file.

# Dataset
The dataset used to train the face mask detector model consists of compiled images of individuals with and without masks. Approximately 700 images of each category.
Feel free to contact me at rohanphil98@gmail.com for the same.

Create a dataset folder with the following tree structure inside your main folder,

dataset
--- with_mask
--- without_mask

# Running the code:

Before running the code, ensure that you download the yolo weights file from the google drive link provided in the .txt file in the yolo-coco folder.

Install the necessary requirements:

pip install -r requirements.txt

(The network was trained on a tensorflow 1.14 system with windows OS)
Tensorflow is not included in the requirements file.

To train the mask detector network run:

python train_mask_detector --dataset dataset

Once the model is saved run:

python social_distance_mask.py -m mask_model

