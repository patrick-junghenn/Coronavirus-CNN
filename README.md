# Using Deep Learning to Diagnose COVID-19 Cases

### Motivation

   I conducted this project after discovering an interesting dataset of lung x-ray images online. The dataset contained lung x-ray images of patients who were positive for COVID-19 as well as x-ray images of patients who had pneumonia. After reading various articles that discussed using deep learning models to diagnose COVID-19 patients, I was inspired to construct my own model. The model I created utilizes a Convolutional Neural Network, or CNN. CNNs are unique in their ability to preserve and account for spatial relationships in an image, making them extremely effective at computer vision tasks (e.g. image classification, object detection, etc).
     
   The use of a deep learning model as a diagnosis tool would be a more widely available and a faster alternative method of identifying individuals who are positive, especially while testing kits are in short supply.  Furthermore, deep learning models can be used as an objective second opinion to thwart false negatives. The following write-up explains the methodology, results, and conclusion for the creation of a deep neural network to identify patients who are positive for COVID-19.

### Data

   The dataset used for this project was found online and is comprised of x-ray images of the lungs of individuals who tested positive for COVID-19. The dataset also contains x-ray images of the lungs of individuals who were inflicted with a different kind of lung infection (in this case, pneumonia). A positive COVID-19 x-ray and a Pneumonia x-ray are show below, respectively.

   Due to a lack of available COVID-19 x-ray images during the modeling process, the dataset contained only 50 images—25 from each category. This posed a problem, as the smaller the number of images, the harder it is for the model to effectively learn and generalize well to new circumstances. To overcome this obstacle, a deep learning technique known as transfer learning was utilized. Various preprocessing tasks were applied to the data. This entailed resizing, reformatting, and scaling the images, all of which was performed by using the Python libraries, OpenCV and NumPy.

### Modeling Process

   As previously mentioned, the dataset is very small. To combat this issue, a deep learning technique known as transfer learning was used. Transfer learning is the process of using the information gained from solving one task (i.e. classifying cars, cats, dogs) and applying it to solve a new task (i.e. classifying flowers). The reason why transfer learning is effective is because low-level information stored in a model (edges, lines, shapes) is easily generalizable to new tasks. All neural networks require edge detection, line detection, and other high-level spatial attributes. Transfer learning exploits the fact that a previously trained neural network will contain information on high-level features that can be generalized to additional tasks.
     
   For building the model, Keras and TensorFlow were used. To construct a base model—the model that already contains low-level information—a VGG16 architecture, previously trained on ImageNet, was used. ImageNet is a large dataset that is frequently utilized as a benchmark for computer vision tasks. It contains 14 million images—making it a perfect dataset to extract broad, low-level features.

   To utilize the VGG16 architecture for transfer learning, certain layers of the model had to be frozen or removed. This is required to ensure that the weights pertaining to low-level spatial information are preserved, and thus transferable to the new tasks (i.e. classifying x-ray images). After freezing and deleting layers of the VGG16, a custom layer is added to the architecture to inherit said low-level information and apply it to the new data (i.e. lung x-ray images).

### Results

   After training the model for 30 epochs, it achieved an accuracy of 92%. A large problem associated with the training process was overfitting, which could easily be counteracted by increasing the number of images. After using a classification report, the model had a weighted f-1 score of 0.92.

  To highlight the effectiveness of transfer learning, I used a VGG16 without pretrained weights. The training process ended after 7 epochs due to the use of an early stopping callback. The model achieved an accuracy of approximately 54%, and a weighted f-1 score of 0.38. Although both models had the same architecture, they had very different outcomes. This is due purely to the efficacy of transfer learning.

### Conclusion

   The use of deep learning models to diagnose potential COVID-19 cases has a hopeful future. With enough data, these models should be able to diagnose individuals at a much faster rate than that of traditional testing kits. The models can also serve as a second opinion in the case of a false negative test result. As the number of COVID-19 cases is increasing, it is more and more important to look for alternative methods that fight the spread of the pandemic. It is essential that as much data be collected from COVID-19 cases as possible. This data could include attributes, measurements, or medical images of positive COVID-19 cases. At a time of such uncertainty, all possible solutions must be explored. Deep learning could be an incredibly powerful tool to combat the spread of the virus and make testing more widely available.
