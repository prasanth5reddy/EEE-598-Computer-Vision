import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


def get_dataset():
    features = np.array([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1], [1.5, -0.5]])
    labels = np.array([1, -1, -1, -1, 1, 1])
    return features, labels


def main():
    X, y = get_dataset()
    svm = SVC(kernel='linear')
    svm.fit(X, y)

    # plot_decision_regions(X, y, clf=svm, legend=2)

    # separate classes
    X_1, X_2 = [], []
    for i in range(y.shape[0]):
        if y[i] == 1:
            X_1.append(X[i])
        else:
            X_2.append(X[i])
    X_1, X_2 = np.array(X_1), np.array(X_2)

    # get linear boundary
    w = svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-2, 2)
    yy = a * xx - (svm.intercept_[0]) / w[1]
    print(a, - (svm.intercept_[0]) / w[1])

    # plot features for each class seperately
    plt.scatter(X_1[:, 0], X_1[:, 1], marker='x')
    plt.scatter(X_2[:, 0], X_2[:, 1], marker='o')

    # plot decision boundary
    plt.plot(xx, yy, 'k-')
    plt.show()


if __name__ == '__main__':
    main()
