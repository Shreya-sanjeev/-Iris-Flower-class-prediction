import numpy as np
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset from a CSV file
iris_dataset = pd.read_csv('iris.csv')

# Define a class for the KNN classifier
class KNN_Classifier:

    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def get_distance_metric(self, training_data_point, test_data_point):
        if self.distance_metric == 'euclidean':
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist += (training_data_point[i] - test_data_point[i])**2
            euclidean_dist = np.sqrt(dist)
            return euclidean_dist

        elif self.distance_metric == 'manhattan':
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist += abs(training_data_point[i] - test_data_point[i])
            manhattan_dist = dist
            return manhattan_dist

    def nearest_neighbors(self, X_train, test_data, k):
        distance_list = []
        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))
        distance_list.sort(key=lambda x: x[1])
        neighbors_list = []
        for j in range(k):
            neighbors_list.append(distance_list[j][0])
        return neighbors_list

    def predict(self, X_train, test_data, k):
        neighbors = self.nearest_neighbors(X_train, test_data, k)
        label = [data[-1] for data in neighbors]
        predicted_class = statistics.mode(label)
        return predicted_class

# Split the dataset into features (X) and target labels (Y)
X = iris_dataset.drop('species', axis=1)
Y = iris_dataset['species']

# Convert data to numpy arrays
X = X.to_numpy()
Y = Y.to_numpy()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)



# Print the training and testing data
print("X_train (Training Features):")
print(X_train)
print("\n")

print("X_test (Testing Features):")
print(X_test)
print("\n")

print("Y_train (Training Labels):")
print(Y_train)
print("\n")

print("Y_test (Testing Labels):")
print(Y_test)
print("\n")
# Create a KNN classifier
knn_classifier = KNN_Classifier(distance_metric='manhattan')

# Index of the data point you want to predict (e.g., index 24)
data_point_index = 29

# Get the specific test data point and its true label
test_data_point = X_test[data_point_index]
true_label = Y_test[data_point_index]

# Predict the Iris species for the specific test data point
prediction = knn_classifier.predict(X_train, test_data_point, k=5)

# Define a mapping for numeric labels to species names
species_mapping = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}
# Convert the prediction to an integer (assuming it's a float)
predicted_numeric_label = int(prediction)

# Check if the integer label is in the species_mapping dictionary
if predicted_numeric_label in species_mapping:
    predicted_species_name = species_mapping[predicted_numeric_label]
else:
    predicted_species_name = "Unknown"

# Print the full information of the data point at index 24
print(f"Full Information of Data Point at Index: ",data_point_index)
print(f"Features: {test_data_point}")
print(f"True Species Label: {true_label}\n")
print(f"species_mapping \n setosa='0',versicolor='1',virginica='2'\n")
print(f"Prediction Label: {prediction}")
print(f"Predicted Species Label: {predicted_species_name}")

