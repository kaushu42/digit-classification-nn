import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Setting the seed to reproduce the results. You can remove this line
np.random.seed(1)

# Load the data set from sklearn
def load_data():
    digits = load_digits()
    X = digits.data
    m = X.shape[0]
    Y = digits.target.reshape(m, 1) # Need to reshape as numpy will return a 1D array otherwise
    return X, Y

# Use this if you want to scale the image
def scale(x, factor = 'auto'):
    if factor == 'auto':
        return x/x.max()
    else:
        return x/factor


# Randomly shuffle the set and split it into test and train set
def split(X, Y, ratio):
    x, y = shuffle(X, Y)
    return train_test_split(x, y, test_size = ratio)

# Visualize n images from the dataset
def visualize(X, n = 1):
    for i in range(n):
        plt.imshow(X[i, :].reshape(8, 8), cmap = 'gray')
        plt.show()

# Perform one hot encoding on the labels
def encode(z):
    onehot = OneHotEncoder()
    return onehot.fit_transform(z).toarray()

# Relu activation function for the hidden layers
def relu(z):
    return z*(z>0)

# Softmax activation to calculate the probabilities
def softmax(z):
    exps = np.exp(z)
    return exps/exps.sum(axis = 1, keepdims = True)

# Calculate the cross-entropy loss
def cost(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    loss = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return loss

# Randomly initialize the parameters
def init_parameters(input_size, hidden_size, output_size):
    w1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return w1, b1, w2, b2

# Move forward on the training process. Calculates the activation for a given input
def forward(x, w1, b1, w2, b2):
    z1 = x.dot(w1.T) + b1.T
    a1 = relu(z1)
    z2 = a1.dot(w2.T) + b2.T
    a2 = softmax(z2)
    return {'a1':a1, 'z1':z1, 'a2':a2, 'z2':z2, }

# Calculate the gradients for the weights and biases
def backward(y, w2, b2, w1, b1, x, cache):
    m = y.shape[0]

    z1 = cache['z1']
    a1 = cache['a1']
    z2 = cache['z2']
    a2 = cache['a2']

    dz2 = a2 - y
    dw2 = 1/m * np.dot(dz2.T, a1)
    db2 = np.mean(dz2, axis = 0, keepdims = True)
    dz1 = dz2.dot(w2)
    dz1[z1<0] = 0
    dw1 = 1/m * np.dot(dz1.T, x)
    db1 = np.mean(dz1, axis = 0, keepdims = True)
    return dw2, db2, dw1, db1


# Train the classifier
# Set verbose = True for training progess
# Set return_costs = True to return the costs at each iteration
def train(x_train, y_train, learning_rate, iterations, return_costs = False, verbose = False):
    w1, b1, w2, b2 = init_parameters(64, 32, 10)
    m = x_train.shape[0]
    costs = []

    if verbose:
        print("Starting....")

    for i in range(iterations):
        if verbose == True:
            if((i+1) % int(iterations/10) == 0):
                print("Iteration: ", i + 1)
                print(cost(y_train, cache['a2']))

        cache = forward(x_train, w1, b1, w2, b2)
        dw2, db2, dw1, db1 = backward(y_train, w2, b2, w1, b1, x_train, cache)

        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1.T
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2.T

        costs.append(cost(y_train, cache['a2']))

    # Save the weights and biases so that they can be loaded later
    np.savez('weights', w1=w1, w2=w2)
    np.savez('biases', b1=b1, b2=b2)

    if return_costs == True:
        return w2, b2, w1, b1, costs
    else:
        return w2, b2, w1, b1

# Predict output for a given input. You can tweak the threshold to get the required confidence
THRESHOLD = 0.8
def predict(x, w, b):
    w1 = w['w1']
    w2 = w['w2']
    b1 = b['b1']
    b2 = b['b2']
    cache = forward(x, w1, b1, w2, b2)
    a = (cache['a2'] >= THRESHOLD)
    return np.argmax(a, axis = 1).reshape(x.shape[0], 1)

# Calculates the accuracy of the predictions
def accuracy(y, y_pred):
    return np.mean(y_pred == y)

# Main driver function to test the program
def main():
    X, Y = load_data()
    X = scale(X)
    x_train, x_test, y_train, y_test = split(X, Y, 0.2)
    y_train_original = y_train.copy()
    
    print('Train Set Size: ', y_train.shape[0])
    print('Test Set Size: ', y_test.shape[0])

    '''Uncomment these lines to train the model'''
    # y_train = encode(y_train)
    # w2, b2, w1, b1, costs = train(x_train, y_train, learning_rate = 0.3, iterations = 4000, return_costs = True, verbose = True)
    # plt.plot(costs)
    # plt.show()

    print("\nTraining is done!\n")

    w = np.load('weights.npz')
    b = np.load('biases.npz')

    y_train_pred = predict(x_train, w, b)
    y_pred = predict(x_test, w, b)
    print("Train Accuracy: ", accuracy(y_train_original, y_train_pred))
    print("Test Accuracy: ", accuracy(y_test, y_pred))

if __name__ == '__main__':
	main()
