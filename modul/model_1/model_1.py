import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

class ImageClassifier:
    def __init__(self, dataset_path, image_size=(100, 100), test_size=0.2, random_state=42):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.test_size = test_size
        self.random_state = random_state
        self.data = []
        self.labels = []
        self.classifier = None

    def extract_features(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, self.image_size)
        return resized_image.flatten()

    def load_data(self):
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.data.append(self.extract_features(image_path))
                self.labels.append(category)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=self.test_size, random_state=self.random_state)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.classifier.fit(X_train, y_train)
        
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        joblib.dump(self.classifier, './modeel_1.h5')
        return accuracy

    def predict(self, image_path):
        if self.classifier is None:
            print("Classifier not trained yet.")
            return None
        features = self.extract_features(image_path)
        prediction = self.classifier.predict([features])
        return prediction[0]


