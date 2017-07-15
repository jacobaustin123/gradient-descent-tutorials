# import tensorflow as tf
import numpy as np

weights = np.random.rand(2,2) # or randn
biases = np.random.rand(1,2)
batch_size = 256
data_size = 20000
iterations = 300
# learning_rate = np.array([.5*(1 - x/iterations) for x in range(iterations)]) # linear learning rate
learning_rate = .5*np.exp(-5*np.arange(0,iterations)/iterations) # exponential learning rate

bool_values = np.random.randint(2, size=(data_size,1))
training_labels = np.concatenate([bool_values, np.logical_not(bool_values).astype(int)], axis=1)

training_data = np.array([[ np.random.normal(loc=.3, scale=.1) if (bool_values[x] == 0) else np.random.normal(loc=.7, scale=.1) for x in range(data_size) ], [ np.random.normal(loc=.3, scale=.1) if (bool_values[x] == 1) else np.random.normal(loc=.7, scale=.1) for x in range(data_size) ]]).T

#training_data = np.random.rand(data_size,2)

def predict(data, batch_size, weights, biases):
    index = np.random.choice(len(data), batch_size, replace=False)
    return index, 1/(1+np.exp(data[index].dot(weights.T) + biases))

def gradient_descent(data, labels, weights, biases, batch_size, iterations, learning_rate):
    for i in range(iterations):
        index, result = predict(data, batch_size, weights, biases)
        diff = result - training_labels[index]
        loss = np.sum(np.inner(diff, diff))/(2*batch_size)
        print("Loss at step %s is %s" % (i, loss))
        deriv_arr = diff*result*(1-result)
        w1 = np.inner(deriv_arr[:,0], data[index][:,0]) / batch_size
        w2 = np.inner(deriv_arr[:,0], data[index][:,1]) / batch_size
        w3 = np.inner(deriv_arr[:,1], data[index][:,0]) / batch_size
        w4 = np.inner(deriv_arr[:,1], data[index][:,1]) / batch_size
        weights = weights + np.array([[w1, w2],[w3, w4]])
        biases = biases + learning_rate[i]*np.sum(deriv_arr) / batch_size

    return weights, biases

def evaluate(data, labels, weights, biases, batch_size):
    index = np.random.choice(len(data), batch_size, replace=False)
    accuracy = (np.argmax(1 / (1 + np.exp(training_data[index].dot(weights.T) + biases)), axis=1)==np.argmax(training_labels[index], axis=1)).sum()*100/batch_size
    print("The accuracy of your model is %s%%!" % accuracy)
    return accuracy, index

new_weights, new_biases = gradient_descent(training_data, training_labels, weights, biases, batch_size, iterations, learning_rate)
accuracy, index = evaluate(training_data, training_labels, new_weights, new_biases, 512)


