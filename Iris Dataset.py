# for data and number manipulation
import pandas as pd
import numpy as np

# for visualization of data
import matplotlib.pyplot as plt

filepath = "C:\Users\antho\VSCode\tutorial_ML_and_NN-main\tutorial_ML_and_NN-main\src\3_Iris\iris.csv"

# read in the file as 'dataset' into a pandas dataframe
dataset = pd.read_csv(filepath)

# preview first 5 lines
print(dataset.head(5))

# print the features and the labels
print(list(dataset.columns))

# count number of times a particular species has occurred
dataset["species"].value_counts()

# check for NAN and empty values in dataframe
NaN_values = dataset.isnull().sum()

# print the number of NaN values present in each column
print('Number of NaN values present: ' + str(NaN_values))

setosa = dataset[dataset.species == "setosa"]
versicolor = dataset[dataset.species=='versicolor']
virginica = dataset[dataset.species=='virginica']

fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

# lables and scatter points
ax.scatter(setosa['petal_length'], setosa['petal_width'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['petal_length'], versicolor['petal_width'], label="Versicolor", facecolor="green")
ax.scatter(virginica['petal_length'], virginica['petal_width'], label="Virginica", facecolor="red")


ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

import seaborn as sns

# pair plot of the Iris dataset
sns.pairplot(dataset, hue='species', markers=['o', 's', 'D'])
plt.show()

# calculate the correlation matrix
corr_matrix = dataset.drop('species', axis=1).corr()

# create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Feature Correlation Matrix')
plt.show()

# create box plots for each feature grouped by species
plt.figure(figsize=(12, 8))

# create box plots for each feature
sns.boxplot(x='species', y='sepal_length', data=dataset)
plt.title('Sepal Length by Species')
plt.show()

sns.boxplot(x='species', y='petal_length', data=dataset)
plt.title('Petal Length by Species')
plt.show()

# create violin plots for each feature
sns.violinplot(x='species', y='sepal_length', data=dataset)
plt.title('Sepal Length Distribution by Species')
plt.show()

sns.violinplot(x='species', y='petal_length', data=dataset)
plt.title('Petal Length Distribution by Species')
plt.show()

# Map the species names to numbers
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
dataset['species'] = dataset['species'].map(species_mapping)

# Display the updated DataFrame
print(dataset.head())

# import the train_test function as our short cut
from sklearn.model_selection import train_test_split

# dropping the species column from X so that it's just data
X = dataset.drop('species', axis=1)  # Features (input data)
# setting the labels to be the column of species (as numerical values)
y = dataset['species']  # Target (labels)

# split the data into training and test sets (80% train, 20% test by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# display the shapes of the resulting sets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

from sklearn.decomposition import PCA

# perform PCA to reduce the dimensions
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# create a DataFrame for the PCA components
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['species'] = y

# plot the PCA components
sns.scatterplot(x='PCA1', y='PCA2', hue='species', data=pca_df, palette='Set1')
plt.title('PCA of Iris Dataset')
plt.show()

# import the SVM libraries
from sklearn import svm # support vector machine function import
from sklearn.inspection import DecisionBoundaryDisplay # display decision

# we're only using the first 2 features (data columns) to keep this simple
Xsample = X.iloc[:, :2].to_numpy()

X0, X1 = X['sepal_length'].to_numpy(), X['sepal_width'].to_numpy()


C = 1.0  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000, dual=True),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(Xsample, y) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)



for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        Xsample,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel="sepal_length",
        ylabel="sepal_width",
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

 # A simple multi-layer perceptron neural network implementation in scikit-learn for classification tasks
from sklearn.neural_network import MLPClassifier

# The functions used to evaluate how well the model performed
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Create a neural network classifier (MLP)
#  with 1 hidden layer and 5 neurons in that layer
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally, display the predictions and the true labels
print("\nPredictions vs True Labels:")
for pred, true in zip(y_pred, y_test):
    print(f"Predicted: {pred}, True: {true}")

# Calculate the confusion matrix of predicted vs actual values
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Create a neural network classifier (MLP)
#  with 3 hidden layers with and 5 neurons, 12 neurons, and 3 neurons
clf = MLPClassifier(hidden_layer_sizes=(5, 12, 3), max_iter=1000, random_state=42)


# Train the model
clf.fit(X_train, y_train)

# make predictions on the test set and save it as y+pred
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# # Optionally, display the predictions and the true labels
# print("\nPredictions vs True Labels:")
# for pred, true in zip(y_pred, y_test):
#     print(f"Predicted: {pred}, True: {true}")

# Calculate accuracy
# * compares the true labels with the predicted labels and returns
# the accuracy score, which is the fraction of correctly classified samples
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Calculate the confusion matrix of predicted vs actual values
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
