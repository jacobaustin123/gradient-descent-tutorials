import numpy as np
# import os

training_data = np.random.rand(20000, 784)
correct_index = np.random.choice(10, np.shape(training_data)[0])
training_labels = np.eye(10)[correct_index]

#training_data = np.random.rand(10000, 784)
#training_weights = np.random.rand(10, 784)
#X = np.dot(training_data, training_weights.T)
#training_labels = X / np.sum(X, axis=1).reshape(10000,1)

# training_data, training_labels, correct_index = import_from_csv('/Users/JAustin/Desktop/MNIST/train.csv', 255)

data_size = np.shape(training_data)[0] # 20000
num_params = np.shape(training_data)[1] # 784
num_classes = np.shape(training_labels)[1] # 10

hidden_units = 256

weights1 = np.random.rand(hidden_units, num_params) # or randn
biases1 = np.random.rand(hidden_units, 1)

weights2 = np.random.rand(num_classes, hidden_units)
biases2 = np.random.rand(num_classes, 1)

batch_size = 100
iterations = 4000

# learning_rate = np.array([.01*(1 - x/iterations) for x in range(iterations)]) # linear learning rate
learning_rate = .01*np.exp(-5*np.arange(0,iterations)/iterations) # exponential learning rate
# learning_rate = .01 * np.ones(iterations) # constant learning rate

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:,None]

def predict(data, batch_size, weights1, biases1, weights2, biases2):
    index = np.random.choice(len(data), batch_size, replace=False)
    hidden_input = np.dot(data[index], weights1.T) + biases1.T # np.dot(weights[:,np.newaxis], np.transpose(data[index])) + biases
    hidden_output = softmax(hidden_input)
    output = np.dot(hidden_output, weights2.T) + biases2.T
    return index, hidden_output, softmax(output)

def evaluate(data, labels, weights1, biases1, weights2, biases2, batch_size):
    index = np.random.choice(len(data), batch_size, replace=False)
    accuracy = (np.argmax(softmax(np.dot(softmax(np.dot(data[index], weights1.T) + biases1.T), weights2.T) + biases2.T), axis=1)==np.argmax(training_labels[index], axis=1)).sum()*100 / batch_size
    print("The accuracy of your model is %s%%!" % accuracy)
    return accuracy, index

def gradient_descent(data, labels, weights1, biases1, weights2, biases2, batch_size, iterations, learning_rate):
    for i in range(iterations):
        index, hidden_prediction, prediction = predict(data, batch_size, weights1, biases1, weights2, biases2)
        loss = - np.tensordot(labels[index], np.log(prediction), axes=2) / batch_size # + .1*np.linalg.norm(weights)**2 # cross entropy loss
        if i % 50 == 0: print("Loss at step %s is %s" % (i, loss))
        error_arr = prediction - labels[index]
        dW2 = np.sum(error_arr[..., None] * data[index][:, None, :], axis=0) / batch_size
        dB2 = np.sum(error_arr, axis=0)[:, None] / batch_size

        dW1 = np.zeros_like(weights1)
        dB1 = np.zeros_like(biases2)

        for m in range(np.shape(weights1)[0]):
            for n in range(np.shape(weights1)[1]):
                dW1[m,n] = np.sum((prediction-labels[index])[:]*np.sum(weights1[:,:]*hidden_prediction[:]*(1-hidden_prediction[m])*data[n]))

        weights1 -= learning_rate[i] * dW1
        biases1 -= learning_rate[i] * dB1
        weights2 -= learning_rate[i] * dW2
        biases2 -= learning_rate[i] * dB2

    return weights1, biases1, weights2, biases2

new_weights, new_biases = gradient_descent(training_data, training_labels, weights1, biases1, weights2, biases2, batch_size, iterations, learning_rate)
accuracy, index = evaluate(training_data, training_labels, new_weights, new_biases, batch_size) # probably should be larger than batch_size


