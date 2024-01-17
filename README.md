# Real-Time-Sign-Language-Recognition

Real Time Sign Language Recognition Using Deep Learning

Input Dataset Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Description:

Sign Language is a distinct and essential way of interaction for a huge section of society. The signs vary in every sign language depending upon the shape of the hand, the position of the gesture, face, body parts, and the motion profile which contribute to the recognition of every gesture. Thus, in deep learning and computer vision, sign language recognition is an extensive field of study. Most of the proposed models by numerous researchers show noteworthy improvements to this field. Through this survey, we are reviewing these vision-based proposed models, using deep learning methodologies from the past few years, but they are neither viable nor pocket-friendly for the end-users. There are many challenges that are yet to be solved, considering the general of suggested models defines a profound improvement in the efficiency of recognizing sign language. Hence this project introduces a solution that proposes a design paradigm that helps to interpret sign language, thereby, aiding interaction among the hearing impaired more easily with others. We present an algorithm that detects sign language in real-time using Convolutional Neural Networks and accurately converts it into sentences to facilitate better interactions. It also converts the sentences into audio to increase accessibility and ease of use. 

Step By Step:

Step 1: Create an input folder and put all the content of the dataset in it.

Step 2: Run preprocess_image.py file depending on the dataset

Step 3: Then run the create_csv.py file for labels for each classes.

Step 4: Then create the cnn model

Step 5: Then run the train.py file for the training the model.

Step 6: Then run the test.py file to test the model.

Step 7: Run the cam_test.py file for Real Time Recognition Using Webcam.
