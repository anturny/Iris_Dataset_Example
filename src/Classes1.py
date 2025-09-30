import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class IrisDataset:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
    def preview(self):
        print(self.df.head(5))
        print(list(self.df.columns))
        print('Number of NaN values present: ' + str(self.df.isnull().sum()))
    def value_counts(self):
        print(self.df['species'].value_counts())
    def split_by_species(self):
        return (
            self.df[self.df.species == "setosa"],
            self.df[self.df.species == "versicolor"],
            self.df[self.df.species == "virginica"]
        )
    def map_species(self):
        mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
        self.df['species'] = self.df['species'].map(mapping)
    def get_features_labels(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        return X, y

class Visualizer:
    def __init__(self, df):
        self.df = df
    def scatter_petals(self):
        setosa, versicolor, virginica = (
            self.df[self.df.species == s] for s in ["setosa", "versicolor", "virginica"]
        )
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 7)
        ax.scatter(setosa['petal_length'], setosa['petal_width'], label="Setosa", facecolor="blue")
        ax.scatter(versicolor['petal_length'], versicolor['petal_width'], label="Versicolor", facecolor="green")
        ax.scatter(virginica['petal_length'], virginica['petal_width'], label="Virginica", facecolor="red")
        ax.set_xlabel("petal length (cm)")
        ax.set_ylabel("petal width (cm)")
        ax.grid()
        ax.set_title("Iris petals")
        ax.legend()
        plt.show()
    def pairplot(self):
        sns.pairplot(self.df, hue='species', markers=['o', 's', 'D'])
        plt.show()
    def correlation_heatmap(self):
        corr_matrix = self.df.drop('species', axis=1).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('Feature Correlation Matrix')
        plt.show()
    def boxplots(self):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='species', y='sepal_length', data=self.df)
        plt.title('Sepal Length by Species')
        plt.show()
        sns.boxplot(x='species', y='petal_length', data=self.df)
        plt.title('Petal Length by Species')
        plt.show()
    def violinplots(self):
        sns.violinplot(x='species', y='sepal_length', data=self.df)
        plt.title('Sepal Length Distribution by Species')
        plt.show()
        sns.violinplot(x='species', y='petal_length', data=self.df)
        plt.title('Petal Length Distribution by Species')
        plt.show()
    def pca_plot(self, X, y):
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
        pca_df['species'] = y
        sns.scatterplot(x='PCA1', y='PCA2', hue='species', data=pca_df, palette='Set1')
        plt.title('PCA of Iris Dataset')
        plt.show()
    def svm_decision_boundaries(self, X, y):
        Xsample = X.iloc[:, :2].to_numpy()
        X0, X1 = X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy()
        C = 1.0
        models = (
            svm.SVC(kernel="linear", C=C),
            svm.LinearSVC(C=C, max_iter=10000, dual=True),
            svm.SVC(kernel="rbf", gamma=0.7, C=C),
            svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
        )
        models = (clf.fit(Xsample, y) for clf in models)
        titles = (
            "SVC with linear kernel",
            "LinearSVC (linear kernel)",
            "SVC with RBF kernel",
            "SVC with polynomial (degree 3) kernel",
        )
        fig, sub = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        for clf, title, ax in zip(models, titles, sub.flatten()):
            DecisionBoundaryDisplay.from_estimator(
                clf,
                Xsample,
                response_method="predict",
                cmap=plt.cm.coolwarm,
                alpha=0.8,
                ax=ax,
                xlabel=X.columns[0],
                ylabel=X.columns[1],
            )
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
        plt.show()

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    def train_mlp(self, hidden_layer_sizes, max_iter=1000, random_state=42):
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:")
        print(cm)

if __name__ == "__main__":
    filepath = "src/iris.csv"
    iris = IrisDataset(filepath)
    iris.preview()
    iris.value_counts()
    setosa, versicolor, virginica = iris.split_by_species()
    viz = Visualizer(iris.df)
    viz.scatter_petals()
    viz.pairplot()
    viz.correlation_heatmap()
    viz.boxplots()
    viz.violinplots()
    iris.map_species()
    X, y = iris.get_features_labels()
    print(iris.df.head())
    trainer = ModelTrainer(X, y)
    print(f'X_train shape: {trainer.X_train.shape}')
    print(f'X_test shape: {trainer.X_test.shape}')
    print(f'y_train shape: {trainer.y_train.shape}')
    print(f'y_test shape: {trainer.y_test.shape}')
    viz.pca_plot(X, y)
    viz.svm_decision_boundaries(X, y)
    print("MLP (5,):")
    trainer.train_mlp((5,))
    print("MLP (5,12,3):")
    trainer.train_mlp((5,12,3))