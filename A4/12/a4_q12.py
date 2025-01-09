import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree

# Column names as per the dataset description
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Load the dataset
# Ensure the local CSV file exists and is properly loaded
data = pd.read_csv('mushroom.csv', header=None, names=columns, skiprows=1)

# Encode categorical features and labels using LabelEncoder
encoder = LabelEncoder()
for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop("class", axis=1)  # Features
y = data["class"]              # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to confirm loading and splitting worked
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# QUESTION 4.1
# Train the CategoricalNB classifier, predict the test set
# and get the corresponding classification report as a dictionary
categorical_nb = CategoricalNB()
categorical_nb.fit(X_train, y_train)
y_pred = categorical_nb.predict(X_test)
cl_report = classification_report(y_test, y_pred, output_dict=True)

# Create training and testing datasets with just the cap-shape
X_train_cap = X_train[["cap-shape"]]
X_test_cap = X_test[["cap-shape"]]

# Train the CategoricalNB classifier with just the cap-shape data
categorical_nb_cap = CategoricalNB()
categorical_nb_cap.fit(X_train_cap, y_train)
y_pred_cap = categorical_nb_cap.predict(X_test_cap)
cl_cap_report = classification_report(y_test, y_pred_cap, output_dict=True)

# Naive Bayes Accuracy Drop
cb_accuracy = np.round(cl_report["accuracy"], 2)
cb_cap_accuracy = np.round(cl_cap_report["accuracy"], 2)
cb_drop = np.round(cb_accuracy - cb_cap_accuracy, 2)
print("Naive Bayes Categorical Accuracy drop: ", cb_accuracy, " - ", cb_cap_accuracy, " = ", cb_drop)

# QUESTION 4.2
# Train a Decision Tree classifier for the whole dataset
decision_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
cl_report = classification_report(y_test, y_pred_dt, output_dict=True)

# Train a Decision Tree classifier for just the cap-shape data
decision_tree_cap = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
decision_tree_cap.fit(X_train_cap, y_train)
y_pred_dt_cap = decision_tree_cap.predict(X_test_cap)
cl_cap_report = classification_report(y_test, y_pred_dt_cap, output_dict=True)

# Decision Tree Accuracy Drop
dt_accuracy = np.round(cl_report["accuracy"], 2)
dt_cap_accuracy = np.round(cl_cap_report["accuracy"], 2)
dt_drop = np.round(dt_accuracy - dt_cap_accuracy, 2)
print("Decision Tree Accuracy drop: ", dt_accuracy, " - ", dt_cap_accuracy, " = ", dt_drop)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree

# Column names as per the dataset description
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Load the dataset
data = pd.read_csv('mushroom.csv', header=None, names=columns, skiprows=1)

# Encode categorical features and labels using LabelEncoder
encoder = LabelEncoder()
for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop("class", axis=1)  # Features
y = data["class"]              # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to confirm loading and splitting worked
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# QUESTION 4.1
# Train the CategoricalNB classifier, predict the test set
# and get the corresponding classification report as a dictionary
categorical_nb = CategoricalNB()
categorical_nb.fit(X_train, y_train)
y_pred = categorical_nb.predict(X_test)
cl_report = classification_report(y_test, y_pred, output_dict=True)

# Create training and testing datasets with just the cap-shape
X_train_cap = X_train[["cap-shape"]]
X_test_cap = X_test[["cap-shape"]]

# Train the CategoricalNB classifier with just the cap-shape data
categorical_nb_cap = CategoricalNB()
categorical_nb_cap.fit(X_train_cap, y_train)
y_pred_cap = categorical_nb_cap.predict(X_test_cap)
cl_cap_report = classification_report(y_test, y_pred_cap, output_dict=True)

# Naive Bayes Accuracy Drop
cb_accuracy = np.round(cl_report["accuracy"], 2)
cb_cap_accuracy = np.round(cl_cap_report["accuracy"], 2)
cb_drop = np.round(cb_accuracy - cb_cap_accuracy, 2)
print("Naive Bayes Categorical Accuracy drop: ", cb_accuracy, " - ", cb_cap_accuracy, " = ", cb_drop)

# QUESTION 4.2
# Train a Decision Tree classifier for the whole dataset
decision_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
cl_report = classification_report(y_test, y_pred_dt, output_dict=True)

# Train a Decision Tree classifier for just the cap-shape data
decision_tree_cap = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
decision_tree_cap.fit(X_train_cap, y_train)
y_pred_dt_cap = decision_tree_cap.predict(X_test_cap)
cl_cap_report = classification_report(y_test, y_pred_dt_cap, output_dict=True)

# Decision Tree Accuracy Drop
dt_accuracy = np.round(cl_report["accuracy"], 2)
dt_cap_accuracy = np.round(cl_cap_report["accuracy"], 2)
dt_drop = np.round(dt_accuracy - dt_cap_accuracy, 2)
print("Decision Tree Accuracy drop: ", dt_accuracy, " - ", dt_cap_accuracy, " = ", dt_drop)
