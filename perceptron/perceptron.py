import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    # can be linearly separable
    features = np.array([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1], [1.5, -0.5]])
    labels = np.array([1, -1, -1, -1, 1, 1])
    return features, labels


def get_additional_dataset():
    # cannot be linearly separable
    features = np.array([[1, 1], [2, -2], [-1, -1.5], [-2, -1], [-2, 1],
                         [1.5, -0.5], [0.5, 0.5], [2, 1], [0, -1.5], [-2, -2]])
    labels = np.array([1, -1, -1, -1, 1, 1, -1, -1])
    return features, labels


def predict(X, W):
    activation = np.dot(X, W[1:]) + W[0]
    return 1 if activation > 0 else -1


def initialise_weights(size):
    # zero initialisation
    return np.zeros((size + 1,))


def train(X, W, y, lr, epochs):
    # list to keep progress of weights
    W_list = np.zeros((epochs, W.shape[0]))
    for i in range(epochs):
        for inputs, label in zip(X, y):
            prediction = predict(inputs, W)
            # update weights, W[0] is a bias term
            W[1:] += lr * (label - prediction) * inputs
            W[0] += lr * (label - prediction)
        W_list[i] = W
    return W, W_list


def plot_weights(W):
    labels = ['b', 'w1', 'w2']
    for i in range(W.shape[1]):
        plt.plot(np.arange(W.shape[0]), W[:, i], label=labels[i])
    plt.legend()
    plt.show()


def perceptron(X, y):
    # initialise weights
    W = initialise_weights(X.shape[1])
    # set learning rate and number of epochs
    lr = 0.01
    epochs = 10
    # train
    W, W_list = train(X, W, y, lr, epochs)
    # plot weights vs epochs
    plot_weights(W_list)
    # perform prediction
    prediction = []
    for i in range(X.shape[0]):
        prediction.append(predict(X[i], W))

    return prediction


def main():
    # get features and labels
    X, y = get_dataset()
    # perform perceptron algorithm
    preds = perceptron(X, y)
    # output predictions
    for i in range(len(preds)):
        print(f'Data : {X[i]}, Prediction : {preds[i]}')

    # get additional data points and perform perceptron
    X_add, y_add = get_additional_dataset()
    preds_add = perceptron(X_add, y_add)


if __name__ == '__main__':
    main()
