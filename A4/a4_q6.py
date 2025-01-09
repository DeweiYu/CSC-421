import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CategoricalNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []
        self.features = []
    
    def fit(self, X, y):
        self.classes = y.unique().tolist()  # Get unique class labels
        self.features = X.columns.tolist()  # Get feature names

        # Calculate prior probabilities for each class
        self.priors = y.value_counts(normalize=True).to_dict()

        # Calculate likelihoods
        self.likelihoods = {}
        for feature in self.features:
            self.likelihoods[feature] = {}
            for c in self.classes:
                # Filter data for class `c`
                class_data = X[y == c][feature]
                # Calculate probabilities for each feature value given the class
                value_counts = class_data.value_counts(normalize=True).to_dict()
                # Store likelihoods for the feature and class
                self.likelihoods[feature][c] = value_counts

    def predict_instance(self, instance):
        posteriors = {}
        for c in self.classes:
            posterior = self.priors[c]
            for feature in self.features:
                feature_value = instance[feature]
                # Multiply posterior with likelihood if feature_value exists for the class
                if feature_value in self.likelihoods[feature][c]:
                    posterior *= self.likelihoods[feature][c][feature_value]
                else:
                    posterior *= 0  # If a feature value is not present, posterior becomes 0
            posteriors[c] = posterior
        # Return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        predictions = X.apply(self.predict_instance, axis=1)
        return predictions

# Example Usage with Mushroom Dataset
data = pd.read_csv('mushroom.csv')

# Separate features (X) and target (y)
X = data.drop("class", axis=1)  # Features
y = data["class"]              # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
nb_classifier = CategoricalNaiveBayes()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
