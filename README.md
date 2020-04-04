
# Diagnosing COVID-19 Patients Using Transfer Learning Model
The dataset in this repository contains lung X-ray images of individuals who tested positive for COVID-19 (coronavirus). The dataset also contains lungs X-ray images of the individuals who had lung complications but tested negative for the coronavirus. 

The model utilizes a VGG16 network, pre-trained on ImageNet, as a base model.

Unfortunately, due to a lack of COVID-19 X-ray images, the size of the dataset is very small (50 images). However, since transfer learning techniques were utilized, the results were decent (about 85% accuracy).

This project is a great example of the effectiveness of transfer learning when data is scarce.

Keras and TensorFlow were utilized for this project. 
