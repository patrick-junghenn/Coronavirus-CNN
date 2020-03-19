# Diagnosing COVID-19 Patients Using A Pretrained VGG16
I conducted this project because I found an interesting dataset that was put together by Adrian Rosebrock, at pyimagesearch. I was inspired after seeing his version of a convolutional neural network that detects coronavirus patients through X-ray images of their lungs, so I decided to try and create my own. 

The dataset in this repository contains X-ray images of the lungs of individuals who tested positive for COVID-19 (coronavirus). The dataset also contains X-ray images of the lungs of individuals who had lung infections that were not related to the coronavirus. 

This CNN utilizes a VGG16 network, pre-trained on ImageNet, as the base model. Keras and TensorFlow are utilized for the creation of this project. 
