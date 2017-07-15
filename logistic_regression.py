import numpy as np

def import_from_csv(path, pixel_depth): # method for importing training dataset and training labels from MNIST dataset
    train_database = np.genfromtxt('{}'.format(path), delimiter=",", dtype=int)[1:,:]
    train_labels = train_database[:,0].reshape(42000,1)
    training_labels = np.eye(len(train_labels))[train_labels]
    training_data = np.delete(train_database,0,1)
    return normalize(training_data), training_labels, train_labels

def normalize(image_data, pixel_depth): # method for normalizing image feature in MNIST dataset
    data = (image_data - pixel_depth / 2) / pixel_depth
    return data.astype(np.float32)

def random_data_init(data_size, num_features, num_classes): # method for generating random data and labels
    training_data = np.random.rand(data_size, num_features) # e.g. 20000, 784
    correct_index = np.random.choice(num_classes, np.shape(training_data)[0])
    training_labels = np.eye(num_classes)[correct_index]
    return training_data, training_labels, correct_index

def random_linear_data_init(data_size, num_features, num_classes): # method for generating random data and normalized labels with linear dependence on data
    training_data = np.random.rand(data_size, num_features)
    training_weights = np.random.rand(num_classes, num_features)
    X = np.dot(training_data, training_weights.T)
    training_labels = X / np.sum(X, axis=1).reshape(data_size,1)
    return training_data, training_labels

# training_data, training_labels, correct_index = import_from_csv('/Users/.../Desktop/MNIST/train.csv', 255)
training_data, training_labels, correct_index = random_data_init(20000, 784, 10)
# training_data, training_labels = random_linear_data_init(20000, 784, 10)

data_size = np.shape(training_data)[0] # 20000
num_features = np.shape(training_data)[1] # 784
num_classes = np.shape(training_labels)[1] # 10

weights = np.random.rand(num_classes, num_features) # or randn
biases = np.random.rand(num_classes, 1)

batch_size = 64
iterations = 4000

# learning_rate = np.array([.01*(1 - x/iterations) for x in range(iterations)]) # linear learning rate
learning_rate = .01*np.exp(-5*np.arange(0,iterations)/iterations) # exponential learning rate
# learning_rate = .01 * np.ones(iterations) # constant learning rate

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:,None]

def predict(data, batch_size, weights, biases):
    index = np.random.choice(len(data), batch_size, replace=False)
    output = np.dot(data[index], weights.T) + biases.T #np.dot(weights[:,np.newaxis], np.transpose(data[index])) + biases
    return index, softmax(output)

def evaluate(data, labels, weights, biases, batch_size):
    index = np.random.choice(len(data), batch_size, replace=False)
    accuracy = (np.argmax(softmax(np.dot(data[index], weights.T) + biases.T), axis=1)==np.argmax(training_labels[index], axis=1)).sum()*100 / batch_size
    print("The accuracy of your model is %s%%!" % accuracy)
    return accuracy, index

def gradient_descent(data, labels, weights, biases, batch_size, iterations, learning_rate):
    for i in range(iterations):
        index, prediction = predict(data, batch_size, weights, biases)
        loss = - np.tensordot(labels[index], np.log(prediction), axes=2) / batch_size # + .1*np.linalg.norm(weights)**2 # cross entropy loss
        if i % 50 == 0: print("Loss at step %s is %s" % (i, loss))
        error_arr = prediction - labels[index]
        dW = np.sum(error_arr[..., None] * data[index][:, None, :], axis=0) / batch_size
        dB = np.sum(error_arr, axis=0)[:, None] / batch_size
        weights -= learning_rate[i]*dW
        biases -= learning_rate[i]*dB

    return weights, biases

new_weights, new_biases = gradient_descent(training_data, training_labels, weights, biases, batch_size, iterations, learning_rate)
accuracy, index = evaluate(training_data, training_labels, new_weights, new_biases, batch_size) # probably should be larger than batch_size


