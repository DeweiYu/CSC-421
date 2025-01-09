

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import datasets, svm
from sklearn import tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB


# Question 4.3 

digits = load_digits()
print("Digits shape: ", digits.data.shape)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)


# Train a GaussianNB classifier and store
# the predictions over Xd_test in yd_predict 

# YOUR CODE GOES HERE

clf = GaussianNB()
clf.fit(Xd_train, yd_train)
yd_predict = clf.predict(Xd_test)

print(
   f"Classification report for classifier {clf}:\n"
   f"{classification_report(yd_test, yd_predict)}\n"
)
gaussian_digits_report = classification_report(yd_test, yd_predict,output_dict=True)
# Find the gaussian_accuracy rounding to two digits 
gaussian_accuracy = np.round(gaussian_digits_report["accuracy"],2)


# Based on the classification report calculate a list of digits sorted by f1-score 
# Each item should be an integer

sorted_digits = sorted(
    [int(digit) for digit in gaussian_digits_report if digit.isdigit()],
    key=lambda x: gaussian_digits_report[str(x)]["f1-score"]
)

# Train a SVM classifier
svm_clf = svm.SVC()
svm_clf.fit(Xd_train, yd_train)
svm_yd_predict = svm_clf.predict(Xd_test)

print(
    f"Classification report for classifier {svm_clf}:\n"
    f"{classification_report(yd_test, svm_yd_predict)}\n"
)
svm_digits_report = classification_report(yd_test, svm_yd_predict, output_dict=True)
# Find the svm_accuracy rounding to two digits
svm_accuracy = np.round(svm_digits_report["accuracy"], 2)

# Update `sorted_digits` to include SVM results, sorted by F1-score
sorted_digits_svm = sorted(
    [int(digit) for digit in svm_digits_report if digit.isdigit()],
    key=lambda x: svm_digits_report[str(x)]["f1-score"]
)


print(sorted_digits) 
print("Digits - Gaussian Accuracy", gaussian_accuracy)
print("Digits - SVM Accuracy", svm_accuracy)

