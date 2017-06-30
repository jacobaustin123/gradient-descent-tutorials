import numpy as np

# def import_from_csv(self, path, pixel_depth, length, class_num):
#     self.train_database = np.genfromtxt('{}'.format(path), delimiter=",", dtype=int)[1:, :]
#     self.train_labels = self.train_database[:, 0].reshape(length, )
#     training_labels = np.eye(class_num)[train_labels]
#     training_data = np.delete(train_database, 0, 1)
#     return self.normalize(training_data, pixel_depth), training_labels, train_labels

class Data:
    def __init__(self, training_data, training_labels, valid_data, valid_labels, pixel_depth):
        self.training_data = training_data
        self.training_labels = training_labels
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.data_size = np.shape(self.training_data)[0]  # 20000
        self.num_params = np.shape(self.training_data)[1]  # 784
        self.num_classes = np.shape(self.training_labels)[1]  # 10

        self.weights = np.random.rand(self.num_classes, self.num_params)  # or randn
        self.biases = np.random.rand(self.num_classes, 1)

class Learning_Rate:
    def __init__(self, learning_rate, init_rate):
        self.learning_rate = learning_rate
        self.init_rate = init_rate

class Linear_Rate(Learning_Rate):

class Model(Data):
    def __init__(self, iterations, batch_size, momentum_rate=.9, learning_rate='exponential'):
        super(Child, self).__init__()
        self.batch_size = batch_size
        self.iterations = iterations
        self.momentum_rate = momentum_rate

    # learning_rate = np.array([.01*(1 - x/iterations) for x in range(iterations)]) # linear learning rate
    learning_rate = .3 * np.exp(-5 * np.arange(0, iterations) / iterations)  # exponential learning rate

    # learning_rate = .01 * np.ones(iterations) # constant learning rate

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

    def predict(data, batch_size, weights, biases):
        index = np.random.choice(len(data), batch_size, replace=False)
        output = np.dot(data[index],
                        weights.T) + biases.T  # np.dot(weights[:,np.newaxis], np.transpose(data[index])) + biases
        return index, softmax(output)

    def evaluate(data, labels, weights, biases, batch_size):
        index = np.random.choice(len(data), batch_size, replace=False)
        accuracy = (np.argmax(softmax(np.dot(data[index], weights.T) + biases.T), axis=1) == np.argmax(labels[index],
                                                                                                       axis=1)).sum() * 100 / batch_size
        print("The accuracy of your model is %s%%!" % accuracy)
        return accuracy, index

    def gradient_descent(data, labels, weights, biases, batch_size, iterations, learning_rate, momentum_rate):
        weight_momentum = np.zeros_like(weights)
        bias_momentum = np.zeros_like(biases)
        gamma = momentum_rate
        for i in range(iterations):
            temp_weights = weights - gamma * weight_momentum
            temp_biases = biases - gamma * bias_momentum
            index, prediction = predict(data, batch_size, temp_weights, temp_biases)
            loss = - np.tensordot(labels[index], np.log(prediction),
                                  axes=2) / batch_size  # + .1*np.linalg.norm(weights)**2 # cross entropy loss
            if i % 50 == 0: print("Loss at step %s is %s" % (i, loss))
            if i % 1000 == 0 and test_eval: evaluate(test_data, test_labels, weights, biases, len(test_data))
            error_arr = prediction - labels[index]
            dW = np.sum(error_arr[..., None] * data[index][:, None, :], axis=0) / batch_size
            dB = np.sum(error_arr, axis=0)[:, None] / batch_size
            weight_momentum = gamma * weight_momentum + learning_rate[i] * dW
            bias_momentum = gamma * bias_momentum + learning_rate[i] * dB
            weights -= weight_momentum
            biases -= bias_momentum

        return weights, biases


new_weights, new_biases = gradient_descent(training_data, training_labels, weights, biases, batch_size, iterations,
                                           learning_rate, momentum_rate)
accuracy, index = evaluate(training_data, training_labels, new_weights, new_biases,
                           batch_size)  # probably should be larger than batch_size

# evaluate(test_data, test_labels, weights, biases, len(test_data))

