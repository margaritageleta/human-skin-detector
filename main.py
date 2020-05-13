from SkinDetector import SkinDetector
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import keras
from keras.utils import to_categorical 

if __name__ == "__main__":
    
    sd = SkinDetector(dilate=2)
    segmented = sd.segment_dataset(sd.TR_DATA)
    X_train = []
    for i in range(0,60):
        result = np.zeros((480,640))
        result[:segmented[i].shape[0],:segmented[i].shape[1]] = segmented[i]
        X_train.append(result.flatten())

    X_train = np.asarray(X_train)
    Y_train = np.asarray(sd.TR_LABEL)
    Y_train = to_categorical(Y_train)
    
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train, sd.TR_LABEL)
    classifier = svm.SVC(kernel='linear', probability=True, random_state=42)
    classifier.fit(X_train, Y_train)

    segmented_test = sd.segment_dataset(sd.VD_DATA)
    X_test = []
    for i in range(0,segmented_test.shape[0]):
    	result = np.zeros((480,640))
    	result[:segmented_test[i].shape[0],:segmented_test[i].shape[1]] = segmented_test[i]
    	X_test.append(result.flatten())

    X_test = np.asarray(X_test)
    Y_test = np.asarray(sd.VD_LABEL)
    Y_test = to_categorical(Y_test)
    
    # Now predict the value of the digit on the second half:
    predicted = sgd_clf.predict(X_test)

    print(accuracy_score(Y_test, predicted))
   