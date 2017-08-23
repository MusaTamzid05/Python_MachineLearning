import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from decision_tree_plotter import DecisionTree_Plotter
from matplotlib import pyplot as plt

def main():

    df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" , header = None)

    X , y = df_wine.iloc[: , 1:].values , df_wine.iloc[: , 0].values
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 0)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)

    pca = PCA(n_components = 2)
    lr = LogisticRegression()

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.fit_transform(X_test_std)
    lr.fit(X_train_pca , y_train)

    decision_tree = DecisionTree_Plotter(X_train_pca , y_train , classifier = lr)
    decision_tree.draw()

    plt.show()



if __name__ == "__main__":
    main()


