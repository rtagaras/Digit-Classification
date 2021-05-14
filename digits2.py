import random
import numpy as np
import pickle
import gzip

# Sigmoid
def sig(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid
def d_sig(z):
    return sig(z)*(1-sig(z))

# Rectified linear unit activation function
def relu(x):
    return max(0,x)

# Derivative of relu
def d_relu(x):
    if x<= 0:
        return 0

    else:
        return 1

# Only used in data loading
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data(filename="mnist.pkl.gz"):
    """
    Returns an array [training_data, validation_data, test_data]. Each "x_data" in the array is a zip constructed from a pair of arrays.
    Each element of the first array in a pair is a 28x28 array that holds an input image.
    Each element of the second array in a pair gives the correct identification of the corresponding input image. For training_data, this
    is a 10-dimensional vector with a 1 in the entry corresponding to the result and zeros elsewhere. For validation_data and test_data,
    this is simply an int. 
    """
    f = gzip.open(filename, 'rb')

    # Creates 3 pairs of arrays. Elements of the first array in a pair are (784,1)-shaped arrays that hold image data.
    # Second array in the pair holds the correct identification for each element of the first array.
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    # For convolutional and pooling layers, it's nicer to have the images as 28x28 squares instead of 784-dimensional vectors like in the datafile.
    training_data = zip([np.reshape(n,(28,28)) for n in training_data[0]], [vectorized_result(x) for x in training_data[1]])
    validation_data = zip([np.reshape(n,(28,28)) for n in validation_data[0]], validation_data[1])
    test_data = zip([np.reshape(n,(28,28)) for n in test_data[0]], test_data[1])
   
    return [training_data, validation_data, test_data]

class Node(object):
    """
    Class for a single node in a fully connected layer. Has a bias and array of input weights.
    """
    def __init__(self, number_of_inputs):
        # Bias for each node is initiated at random, for now. 
        self.bias = np.array([np.random.randn()])

        # Number of nodes in previous layer. Needed so that we know how many weights come in to specific node. 
        self.number_of_inputs = number_of_inputs

        # Column vector of weights that come in to specific node. Initiated at random, for now. 
        self.input_weights = np.random.randn(number_of_inputs,1)

class Layer(object):
    def __init__(self, size, prev_layer_size):

        # size is an int that gives the number of nodes in the layer
        self.size = size
        self.prev_layer_size = prev_layer_size
        self.nodes = np.array([Node(self.prev_layer_size) for i in range(size)])
        self.layer_input = 0
        # Square matrix of weights. Each row contains all connections going into a single node. Rows are used instead
        # of columns so that we can write the output of a node in matrix form as output=sig(WA+B), where W is the matrix defined here, 
        # A is the column vector of values given by the previous layer, and B is the column vector of biases of the previous layer.
        # Note that the first layer and layers 2,...,N are all from the same class, so there will be weights going "in" to the first layer,
        # but nothing is ever calculated with these. 
        #self.weights = np.concatenate([n.input_weights for n in self.nodes],axis=1).T

        self.weights = np.array([[1.,	0.03125,	0.00411523,	0.000976563],
                        [0.00032,	0.000128601,	0.000059499,	0.0000305176],
                        [0.0000169351,	0.00001,	6.20921E-6,	4.01878E-6],
                        [2.69329E-6,	1.85934E-6,	1.31687E-6,	9.53674E-7],
                        [7.04296E-7,	5.29221E-7,	4.03861E-7,	3.125E-7],
                        [2.44852E-7,	1.94038E-7,	1.55368E-7,	1.25587E-7],
                        [1.024E-7,	8.41653E-8,	6.96917E-8,	5.81045E-8],
                        [4.8754E-8,	4.11523E-8,	3.49294E-8,	2.98023E-8],
                        [2.55523E-8,	2.20093E-8,	1.90397E-8,	1.65382E-8],
                        [1.44209E-8,	1.26207E-8,	1.10835E-8,	9.76563E-9]])

        # Column vector of biases corresponding to each node in the layer
        #self.biases = np.array([n.bias for n in self.nodes])
    
        self.biases = np.ones((10,1))

        # Running sum that holds the change to be applied to the bias vector after each minibatch
        self.bias_sum = np.zeros(self.biases.shape)

        # Running sum that holds change in weight matrix
        self.weight_sum = np.zeros(self.weights.shape)

        # Sometimes, we need to reuse a previously calculated value of z, so I store it here. 
        self.z = np.zeros(np.shape(self.biases))

        self.error = np.zeros(np.shape(self.biases))

    def output(self):

        # reshape the rectangular matrix input into a column vector
        self.layer_input[0] = np.reshape(self.layer_input[0], (-1,1))
       
        z_temp = np.matmul(self.weights, self.layer_input[0]) + self.biases
        self.z = z_temp

        return sig(z_temp)

    # sets self.error, given the error and weights from the next layer
    def err(self, next_layer_error, next_layer_weights):
        e = np.multiply(np.matmul(next_layer_weights.T, next_layer_error), d_sig(self.z)) 
        self.error = e

        return e

class Softmax_layer(Layer):
    def __init__(self, size, prev_layer_size):
        #super().__init__(size, layer_input)

        # size is an int that gives the number of nodes in the layer
        self.size = size
        self.prev_layer_size = prev_layer_size
        self.nodes = np.array([Node(self.prev_layer_size) for i in range(size)])
        self.layer_input = 0
        # Square matrix of weights. Each row contains all connections going into a single node. Rows are used instead
        # of columns so that we can write the output of a node in matrix form as output=sig(WA+B), where W is the matrix defined here, 
        # A is the column vector of values given by the previous layer, and B is the column vector of biases of the previous layer.
        # Note that the first layer and layers 2,...,N are all from the same class, so there will be weights going "in" to the first layer,
        # but nothing is ever calculated with these. 
        #self.weights = np.concatenate([n.input_weights for n in self.nodes],axis=1).T

        m = np.arange(1,101)
        m = np.reshape(m,(10,10))
        self.weights = m
        # Column vector of biases corresponding to each node in the layer
        #self.biases = np.array([n.bias for n in self.nodes])
    
        self.biases = np.reshape(np.array([-41,-117,-193,-269,-345,-421,-497,-573,-649,-725]), (-1,1))

        # Running sum that holds the change to be applied to the bias vector after each minibatch
        self.bias_sum = np.zeros(self.biases.shape)

        # Running sum that holds change in weight matrix
        self.weight_sum = np.zeros(self.weights.shape)

        # Sometimes, we need to reuse a previously calculated value of z, so I store it here. 
        self.z = np.zeros(np.shape(self.biases))

        self.error = np.zeros(np.shape(self.biases))


    def output(self):
        m = np.matmul(self.weights, self.layer_input)
        self.z = m.reshape(-1,1) + self.biases

        exp = np.exp(self.z - np.max(self.z))
        return exp / np.sum(exp)

    # This gives the error in the final layer, if this layer type is used as the last layer in the network.
    # Note that the mathematical expression can be different for different layer types. 
    # Also note that I assume that a softmax layer will use log-likelhood as the cost function. 
    #
    # "guess" and "answer" should both be column vectors.
    def final_error(self, guess, answer):

        e = guess - answer
        self.error = e
        return e

class Filter(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        #self.weights = np.random.randn(width, height)
        #self.bias = np.random.randn()

        self.weights = np.array([[1,2],[3,4]])
        self.bias = 1

        self.weight_sum = np.zeros(np.shape(self.weights))
        self.bias_sum = np.zeros(np.shape(self.bias))

class ConvolutionalLayer(object):
    def __init__(self, filters, pooling_filter):
        '''
        "filters" is an array of filter object that define individual feature maps, "pooling_filter" is a filter that is used as part of the built-in
        pooling function. layer_input" is a rectangular numpy array of nodes. 
        '''
        
        self.layer_input = 0
        self.filters = filters
        self.pooling_filter = pooling_filter
        
        #vector of grids, one for each feature map. This is populated in the next loop, when we have easier access to the filter shapes.
        self.grids = []
        self.pooled_grids = []

        # When we pool the data in the layer, we need to keep track of the index of the maximum value at each step of the pooling filter. 
        # This is an array of matrices, one for each filter map. Each matrix is the same size as the output of the pooling layer, and each element
        # is an ordered pair that gives the coordinates (in terms of the pre-pooled layer) of the largest value. 
        self.max_vals = []

        # array of error vectors - one vector for each feature map. The weights and error are included as part of the filter.
        self.error = []
        for f in filters:
            self.error.append(np.zeros(np.shape(f)))


    # This is the output that would be fed into a pooling layer. Since I have combined the convolutional layer and the pooling layer, 
    # this function is not the final output of the layer. 
    def pre_output(self):
        for n,f in enumerate(self.filters):
            filter_width = f.width
            filter_height = f.height

            # The new layer is a grid of nodes. The size of the layer is determined by the size of the input, the filter, and the stride length.
            # For example, if we have a 28x28 input and a 5x5 filter moved with stride length 1, we are left with a 24x24 layer, since we can 
            # only move the filter 23 units in each direction before reaching the edge of the input. Here, we only use stride length 1. 
            width = np.shape(self.layer_input[0])[0]-filter_width+1
            height = np.shape(self.layer_input[0])[1]-filter_height+1

            self.grids.append(np.zeros((width, height)))
            for j in range(width):
                for k in range(height):
                    
                    s=0
                    for l in range(filter_width):
                        for m in range(filter_height):
                            s += f.weights[l][m]* self.layer_input[0][j+l][k+m]
                    
                    self.grids[n][j][k] = relu(f.bias+s)

        return self.grids

    # I plan on always having a pooling layer after each convolutional layer, so instead of creating a new class, I"m just going to do the pooling 
    # in the convolutional layer class, using the unpooled output provided by the output function. 
    def output(self):

        #standard_output = self.pre_output(self.layer_input)
        standard_output = self.pre_output()

        pooling_filter_width = self.pooling_filter.width 
        pooling_filter_height = self.pooling_filter.height

       
        for n, g in enumerate(standard_output):
            pooled_width = int(np.shape(g)[0]/pooling_filter_width)
            pooled_height = int(np.shape(g)[1]/pooling_filter_height)

            self.pooled_grids.append(np.zeros((pooled_width, pooled_height)))
            self.max_vals.append(np.zeros((pooled_width, pooled_height)))


            for j in range(pooled_width):
                for k in range(pooled_height):

                    # Take the largest value in region selected by filter
                    m = g[j*pooling_filter_height:(j+1)*pooling_filter_height, k*pooling_filter_width:(k+1)*pooling_filter_width]
                    self.pooled_grids[n][j][k] = np.amax(m)
                    
                    # index of largest value, using indices of the matrix that we apply the pooling filter to
                    # a,b = np.unravel_index(m.argmax(), m.shape)
                    #self.max_vals[n][j][k] = [a,b]

                    # Don't forget that this needs to be unravelled later. 
                    self.max_vals[n][j][k] = m.argmax()

        # for g in self.pooled_grids:
        #     print(g)

        return self.pooled_grids  

    # Gives the error for a particular layer in terms of the error in the next layer in the network.
    # See derivation at https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    def err(self, next_layer_error, next_layer_weights):
        for n,f in enumerate(self.filters):
            error_vec = []
            error_vec.append(np.zeros(np.shape(f)))
            s = 0

            for x in range(f.width):
                for y in range(f.height):
                    
                    # Is this line broken?
                    s += next_layer_error[x-n][y-m] * next_layer_weights[x][y] * d_relu(self.pre_output(self.layer_input))

            error_vec[n][x][y] = s
        
        return error_vec

class Network(object):
    def __init__(self,layers):
        """
        "layers" is an array that holds layer objects of various types. The last element should be 10-dimensional, fully-connected layer. 
        """

        self.layers = layers

        # Array that stores predicted digit for each input in test data
        self.test_outputs = []

    def cost(self, x, lam, weights, n):
        
        # We use L2 regularization to help with overfitting.
        s = 0
        for i in range(np.shape(weights)[0]):
            for j in range(np.shape(weights)[1]):
                s += (weights[i][j])**2
        
        return -np.log(x) + lam*s/(2.0*n)

    def train(self, training_data, epochs, batch_size, rate, test_data):
        
        # A subset of data that we can compare with after each batch to check progress. 
        test_data = list(test_data)
        n_test = len(test_data)        

        #validation_data = list(validation_data)
        #n_val = len(validation_data)
        
        # Data is list of tuples (x,y), where x is a 28x28-dimensional numpy array that holds the input image and y is a 10-dimensional
        # numpy array that indicates the correct number that corresponds to the image. 
        training_data = list(training_data)
        n_train = len(training_data)

        # Repeatedly train on shuffled dataset 
        for i in range(epochs):

            # Reset output vector for each epoch so that we can see how learning is improving
            self.test_outputs = []

            # Randomly shuffle training data and partition into subsets for batch training
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size] for j in range(0, n_train, batch_size)]
            
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
            
    def forward_pass(self, data):
            # Set input layer
            self.layers[0].layer_input = data
            
            print(data[0])
            # Iterate over layers after the input
            for k in range(1,len(self.layers)):
                layer = self.layers[k]
                prev_layer = self.layers[k-1]

                # layer.layer_input = prev_layer.output(prev_layer.layer_input)
                layer.layer_input = prev_layer.output()
                #print(layer.output()[0])
            return (layer.output(), layer.z)

    # correct this too
    def backwards_pass(self, last_layer_error):
        # Backwards pass
        for l in range(len(self.layers)-2,0,-1):
            layer = self.layers[l]
            next_layer = self.layers[l+1]
            prev_layer = self.layers[l-1]
            
            err = layer.error(next_layer.error, next_layer.weights)

            layer.error = np.multiply(np.matmul(next_layer.weights.T,next_layer.error), d_sig(layer.z))              
            layer.bias_sum = layer.bias_sum + layer.error
            layer.weight_sum = layer.weight_sum + np.matmul(layer.error, prev_layer.activation.T)

    # this needs to be modified to reflect the changes made to forward_pass
    def backpropagate(self, batch, rate):
        m = len(batch)
        
        for drawing, answer in batch:                    
            result = self.forward_pass(drawing)
            
            layer = self.layers[-1]

            # The final layer's activation error needs to be calculated so we can compare to the desired result. 
            delta = np.multiply(result-answer, layer.activation(layer.z))

            self.backwards_pass(delta)               
        
        self.update_network(rate, m)

    def update_network(self, rate, batch_size):
        # update weights and biases and reset sums for next batch
        for layer in self.layers:
            layer.weights -= (rate/batch_size)* layer.weight_sum
            layer.biases -= (rate/batch_size)* layer.bias_sum 

            layer.weight_sum = np.zeros(layer.weights.shape)
            layer.bias_sum = np.zeros(layer.biases.shape)

    

#[training_data, validation_data, test_data] = load_data()
#net = Network([784, 30, 10])
#net = Network([Layer(784), Layer(), Layer()])
#net.train(training_data, 30, 10, 3.0, test_data=test_data)

#5x5
training_data = (np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]), np.reshape(np.array([1,0,0,0,0,0,0,0,0,0]), (-1,1)))

f = [Filter(2,2)]
pf = Filter(2,2)
l_c = ConvolutionalLayer(f, pf)
# print(l_c.pre_output(test_img)[0])
# print(np.reshape(l_c.output()[0][0], (-1, 1)))

l_fc = Layer(10, 4)
# print(l_fc.output())

l_s = Softmax_layer(10, 10)
# print(l_s.output())
net = Network([l_c, l_fc, l_s])


#print(training_data[0])
# l_c.layer_input = training_data

print(net.forward_pass(training_data))
# #print(training_data)

# lco = l_c.output()
# print(lco)

# l_fc.layer_input = lco
# lfco = l_fc.output()
# print(lfco)

# l_s.layer_input = lfco
# lso = l_s.output()
# print(lso)

