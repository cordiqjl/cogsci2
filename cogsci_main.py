from read_datasets import load_fer2013
from emotion_detection import angry_not_angry_labels
import cv2
import numpy as np

'''
FER2013 emotion abels:
0 - Angry
1 - Disgust
2 - Fear
3 - Happy
4 - Sad
5 - Surprise
6 - Neutral
'''
fer_images, fer_labels = load_fer2013()

angry_binary_labels = angry_not_angry_labels(fer_labels)

#print(angry_binary_labels)

# binary classification for FER
# happy/not happy
# angry/not angry

for i in range(len(fer_images)):
    if i % 2 == 0:
        X_train.append(fer_images[i])
        Y_train.append(angry_binary_labels[i])
    else:
        X_test.append(fer_images[i])
        Y_test.append(angry_binary_labels[i])


##SIFT DESCRIPTORS FOR ALL

sift = cv2.xfeatures2d.SIFT_create()


def get_SIFT_descriptors(images):
    SIFT_descr = []
    for image in images:
        cv_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, image_sift = sift.detectAndCompute(cv_gray_image, None)
        SIFT_descr.append(image_sift)
    return SIFT_descr

train_descriptors = get_SIFT_descriptors(X_train)
test_descriptors = get_SIFT_descriptors(X_test)

train_matrix = np.concatenate(train_descriptors)
test_matrix = np.concatenate(test_descriptors)

print(train_matrix.shape)


'''
CC emotion labels:
1 - Angry 45
2 - Contempt 18
3 - Disgust 59
4 - Fear 25
5 - Happy 69
6 - Sadness 28
7 - Surprise 83
'''
# read the other dataset (get the faces)
