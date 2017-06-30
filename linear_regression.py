# import tensorflow as tf
import numpy as np
import os

""" def import_from_csv(path, pixel_depth):
    train_database = np.genfromtxt('{}'.format(path), delimiter=",", dtype=int)[1:,:]
    train_labels = train_database[:,0].reshape(42000,1)
    training_labels = np.eye(len(train_labels))[train_labels]
    training_data = np.delete(train_database,0,1)
    return normalize(training_data), training_labels, train_labels

def normalize(image_data, pixel_depth):
    data = (image_data - pixel_depth / 2) / pixel_depth
    return data.astype(np.float32) """

training_data = np.random.randn(20000, 784)
training_weights = np.random.randn(10, 784)
training_labels = np.dot(training_data, training_weights.T)

#correct_index = np.random.choice(10, np.shape(training_data)[0])
#training_labels = np.eye(10)[correct_index]

# training_data, training_labels, correct_index = import_from_csv('/Users/JAustin/Desktop/MNIST/train.csv', 255)

data_size = np.shape(training_data)[0] # 20000
num_params = np.shape(training_data)[1] # 200
num_classes = np.shape(training_labels)[1] # 10

weights = np.random.randn(num_classes, num_params) # or randn
biases = np.random.randn(num_classes, 1)

batch_size = 1000
iterations = 200

# learning_rate = np.array([.5*(1 - x/iterations) for x in range(iterations)]) # linear learning rate
learning_rate = .4*np.exp(-5*np.arange(0,iterations)/iterations) # exponential learning rate
#learning_rate = .1 * np.ones(iterations) # constant learning rate

def predict(data, batch_size, weights, biases):
    index = np.random.choice(len(data), batch_size, replace=False)
    prediction = np.dot(data[index], weights.T) + biases.T
    return index, prediction

def least_square_optimization(data, labels, weights, biases, batch_size, iterations, learning_rate):
    for i in range(iterations):
        index, prediction = predict(data, batch_size, weights, biases)
        error_arr = labels[index] - prediction
        loss = np.tensordot(error_arr, error_arr) / (2*batch_size)
        if i % 50 == 0: print("Loss at step %s is %s" % (i, loss))
        dW = - np.sum(error_arr[..., None] * data[index][:, None, :], axis=0) / batch_size
        dB = - np.sum(error_arr, axis=0)[:, None] / batch_size
        weights = weights - learning_rate[i]*dW
        biases = biases - learning_rate[i]*dB

    return weights, biases

def evaluate(data, labels, weights, biases, batch_size):
    index = np.random.choice(len(data), batch_size, replace=False)
    accuracy = (np.argmax(np.dot(data[index], weights.T) + biases.T, axis=1) == np.argmax(training_labels[index], axis=1)).sum() * 100 / batch_size
    print("The accuracy of your model is %s%%!" % accuracy)
    return accuracy, index

new_weights, new_biases = least_square_optimization(training_data, training_labels, weights, biases, batch_size, iterations, learning_rate)
accuracy, index = evaluate(training_data, training_labels, new_weights, new_biases, 512)


