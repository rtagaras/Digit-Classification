import random
import numpy as np
import gzip
import pickle
#import mnist_loader

# Sigmoid
def sig(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid
def d_sig(z):
    return sig(z)*(1-sig(z))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data(filename="mnist.pkl.gz"):
    """
    Returns an array [training_data, validation_data, test_data]. Each "x_data" in the array is a zip constructed from a pair of arrays.
    Each element of the first array in a pair is a 784-dimensional vector that holds an input image.
    Each element of the second array in a pair gives the correct identification of the corresponding input image. For training_data, this
    is a 10-dimensional vector with a 1 in the entry corresponding to the result and zeros elsewhere. For validation_data and test_data,
    this is simply an int. 
    """
    f = gzip.open(filename, 'rb')

    # Creates 3 pairs of arrays. Elements of the first array in a pair are (784,1)-shaped arrays that hold image data.
    # Second array in the pair holds the correct identification for each element of the first array.
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    training_data = zip([np.reshape(n,(784,1)) for n in training_data[0]], [vectorized_result(x) for x in training_data[1]])
    validation_data = zip([np.reshape(n,(784,1)) for n in validation_data[0]], validation_data[1])
    test_data = zip([np.reshape(n,(784,1)) for n in test_data[0]], test_data[1])

    print("Data import complete.")

    return [training_data, validation_data, test_data]

class Node(object):
    def __init__(self, prev_layer_size):
        # Bias for each node is initiated at random, for now. 
        self.bias = np.array([np.random.randn()])

        # Number of nodes in previous layer. Needed so that we know how many weights come in to specific node. 
        self.prev_layer_size = prev_layer_size

        # Column vector of weights that come in to specific node. Initiated at random, for now. 
        self.input_weights = np.random.randn(prev_layer_size,1)
        
class Layer(object):
    def __init__(self, size, prev_layer_size):
        self.size = size
        self.prev_layer_size = prev_layer_size
        self.nodes = np.array([Node(prev_layer_size) for i in range(size)])
        
        # Square matrix of weights. Each row contains all connections going into a single node. Rows are used instead
        # of columns so that we can write the output of a node in matrix form as output=sig(WA+B), where W is the matrix defined here, 
        # A is the column vector of values given by the previous layer, and B is the column vector of biases of the previous layer.
        # Note that the first layer and layers 2,...,N are all from the same class, so there can be weights going "in" to the first layer,
        # but nothing is ever calculated with these, if they exist. 
        self.weights = np.concatenate([n.input_weights for n in self.nodes],axis=1).T

        # Column vector of biases corresponding to each node in the layer
        self.biases = np.array([n.bias for n in self.nodes])

        self.activation = 0
        self.error = 0
        self.z = 0

        # Running sum that holds the change to be applied to the bias vector after each minibatch
        self.bias_sum = np.zeros(self.biases.shape)

        # Running sum that holds change in weight matrix
        self.weight_sum = np.zeros(self.weights.shape)

class Network(object):
    def __init__(self,sizes):
        # Array of number of nodes in each layer
        self.sizes = sizes
        
        # Prepends zero to list of layer sizes so that the first layer doesn't get any incoming weights.
        self.sizes.insert(0,0)

        # Array of layers of varying size
        self.layers = []
        for i in range(1,len(self.sizes)):
            self.layers.append(Layer(self.sizes[i],self.sizes[i-1]))

        # Array that stores predicted digit for each input in test data
        self.test_outputs = []

    def train(self, training_data, epochs, batch_size, rate, test_data):
        
        # A subset of data that we can compare with after each batch to check progress. 
        test_data = list(test_data)
        n_test = len(test_data)
        
        # Data is list of tuples (x,y), where x is a 784-dimensional numpy array that holds the input image and y is a 10-dimensional
        # numpy array that indicates the correct number that corresponds to the image. 
        training_data = list(training_data)
        
        # Total number of inputs to train with
        n = len(training_data)

        # Repeatedly train on shuffled dataset 
        for i in range(epochs):

            # Reset output vector for each epoch so that we can see how learning is improving
            self.test_outputs = []

            # Randomly shuffle training data and partition into subsets for batch training
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]
            
            for batch in batches:
                self.backpropagate(batch, rate)
                
            # Do another forward pass to get the outputs for test data
            for drawing, answer in test_data:
                self.forward_pass(drawing)
                        
                # Add the number that the network believes corresponds to the training image to the output vector
                self.test_outputs.append(np.argmax(self.layers[-1].activation))

            # Calculate the number of correct identifications in test data after each training epoch
            test_results = [(self.test_outputs[x], test_data[x][1]) for x in range(len(test_data))]
            num_correct = sum(int(drawing == answer) for (drawing, answer) in test_results)
            print("Epoch ", i+1,': ', num_correct, '/', n_test)
            
    def backpropagate(self, batch, rate):
        m = len(batch)
        
        for drawing, answer in batch:                    
            self.forward_pass(drawing)
            
            # The final layer's activation error needs to be calculated so we can compare to the desired result. 
            layer = self.layers[-1]
            layer.error = np.multiply(layer.activation-answer,d_sig(layer.z))

            self.backwards_pass()               
        
        self.update_network(rate, m)
        
    def forward_pass(self, data):
        # Set input layer
        self.layers[0].activation = data

        # Forward pass - iterate over layers after the input
        for k in range(1,len(self.layers)):
            layer = self.layers[k]
            prev_layer = self.layers[k-1]

            layer.z = np.matmul(layer.weights, prev_layer.activation)+np.reshape(layer.biases,(len(layer.biases),1))
            layer.activation = sig(layer.z)

    def update_network(self, rate, batch_size):
        # update weights and biases and reset sums for next batch
        for layer in self.layers:
            layer.weights -= (rate/batch_size)* layer.weight_sum
            layer.biases -= (rate/batch_size)* layer.bias_sum 

            layer.weight_sum = np.zeros(layer.weights.shape)
            layer.bias_sum = np.zeros(layer.biases.shape)

    def backwards_pass(self):
        # Backwards pass
        for l in range(len(self.layers)-2,0,-1):
            layer = self.layers[l]
            next_layer = self.layers[l+1]
            prev_layer = self.layers[l-1]
            
            layer.error = np.multiply(np.matmul(next_layer.weights.T,next_layer.error),d_sig(layer.z))              
            layer.bias_sum = layer.bias_sum + layer.error
            layer.weight_sum = layer.weight_sum + np.matmul(layer.error, prev_layer.activation.T)

[training_data, validation_data, test_data] = load_data()
net = Network([784, 30, 10])
net.train(training_data, 30, 10, 3.0, test_data)
