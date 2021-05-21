import random
import numpy as np
import pickle
import gzip

# Sigmoid
def sig(x):
    return 1.0/(1.0+np.exp(-x))

# Derivative of sigmoid
def d_sig(x):
    return sig(x)*(1-sig(x))

# Rectified linear unit activation function
def relu(x):
    return max(0,x)

# Derivative of relu
def d_relu(x):
    return np.where(x<=0, 0, 1)

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
        self.layer_input = []

        # Square matrix of weights. Each row contains all connections going into a single node. Rows are used instead
        # of columns so that we can write the output of a node in matrix form as output=sig(WA+B), where W is the matrix defined here, 
        # A is the column vector of values given by the previous layer, and B is the column vector of biases of the previous layer.
        # Note that the first layer and layers 2,...,N are all from the same class, so there will be weights going "in" to the first layer,
        # but nothing is ever calculated with these. 
        self.weights = np.concatenate([n.input_weights for n in self.nodes],axis=1).T

        # Column vector of biases corresponding to each node in the layer
        self.biases = np.array([n.bias for n in self.nodes])
    
        #self.biases = np.ones((10,1))

        # Running sum that holds the change to be applied to the bias vector after each minibatch
        self.bias_sum = np.zeros(self.biases.shape)

        # Running sum that holds change in weight matrix
        self.weight_sum = np.zeros(self.weights.shape)

        # Sometimes, we need to reuse a previously calculated value of z, so I store it here. 
        self.z = np.zeros(np.shape(self.biases))

        self.error = np.zeros(np.shape(self.biases))

    # I changed all self.layer_input[0] to self.layer_input here as well.
    def output(self):

        # reshape the rectangular matrix input into a column vector
        self.layer_input = np.reshape(self.layer_input, (-1,1))
       
        z_temp = np.matmul(self.weights, self.layer_input) + self.biases
        self.z = z_temp

        return sig(z_temp)

    # sets self.error, given the error and weights from the next layer
    def error_output(self, error_in, weights):
        e = np.multiply(np.matmul(weights.T, error_in), d_sig(self.z)) 
        self.error = e

        return e

    def gradients(self):
        w_shape = np.shape((self.weights))
        w = np.zeros(w_shape)

        for i in range(w_shape[0]):
            for j in range(w_shape[1]):
                w[i][j] = self.error[i] * self.layer_input[j]

        self.weight_sum += w
        self.bias_sum += self.error

        return w, self.error

    def update_parameters(self, rate, batch_size):
        self.weights -= (rate/batch_size)* self.weight_sum
        self.biases -= (rate/batch_size)* self.bias_sum 

    def reset_gradient_sums(self):
        self.weight_sum = np.zeros(self.weights.shape)
        self.bias_sum = np.zeros(self.biases.shape)


class Softmax_layer(Layer):
    def __init__(self, size, prev_layer_size):

        # size is an int that gives the number of nodes in the layer
        self.size = size
        self.prev_layer_size = prev_layer_size
        self.nodes = np.array([Node(self.prev_layer_size) for i in range(size)])
        self.layer_input = []

        # Square matrix of weights. Each row contains all connections going INTO a single node. Rows are used instead
        # of columns so that we can write the output of a node in matrix form as output=sig(WA+B), where W is the matrix defined here, 
        # A is the column vector of values given by the previous layer, and B is the column vector of biases of the previous layer.
        # Note that the first layer and layers 2,...,N are all from the same class, so there will be weights going "in" to the first layer,
        # but nothing is ever calculated with these. 
        self.weights = np.concatenate([n.input_weights for n in self.nodes],axis=1).T

        # Column vector of biases corresponding to each node in the layer
        self.biases = np.array([n.bias for n in self.nodes])
    
        # Running sum that holds the change to be applied to the bias vector after each minibatch
        self.bias_sum = np.zeros(self.biases.shape)

        # Running sum that holds change in weight matrix
        self.weight_sum = np.zeros(self.weights.shape)

        # Sometimes, we need to reuse a previously calculated value of z, so I store it here. 
        self.z = np.zeros(np.shape(self.biases))

        self.error = np.zeros(np.shape(self.biases))

        self.activation = []

    def output(self):
        m = np.matmul(self.weights, self.layer_input)
        self.z = m.reshape(-1,1) + self.biases

        exp = np.exp(self.z - np.max(self.z))
        self.activation = exp / np.sum(exp)
        return exp / np.sum(exp)

    # This gives the error in the final layer, if this layer type is used as the last layer in the network.
    # Note that the mathematical expression can be different for different layer types. 
    # Also note that I assume that a softmax layer will use log-likelhood as the cost function. 
    #
    # "guess" and "answer" should both be column vectors.
    def error_output(self, guess, answer):        
        e = guess - answer
        self.error = e

        return e

    # Returns dC/dW and dC/db as arrays
    def gradients(self):
        w_shape = np.shape((self.weights))
        w = np.zeros(w_shape)

        for i in range(w_shape[0]):
            for j in range(w_shape[1]):
                w[i][j] = self.error[i] * self.layer_input[j]

        self.weight_sum += w
        self.bias_sum += self.error

        return w, self.error

    def update_parameters(self, rate, batch_size):
        self.weights -= (rate/batch_size)* self.weight_sum
        self.biases -= (rate/batch_size)* self.bias_sum 

    def reset_gradient_sums(self):
        self.weight_sum = np.zeros(self.weights.shape)
        self.bias_sum = np.zeros(self.biases.shape)

class Filter(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.weights = np.random.randn(width, height)
        self.bias = np.random.randn()
        self.weight_sum = np.zeros(np.shape(self.weights))
        self.bias_sum = np.zeros(np.shape(self.bias))

class ConvolutionalLayer(object):
    def __init__(self, filters, pooling_filter):
        '''
        "filters" is an array of filter object that define individual feature maps, "pooling_filter" is a filter that is used as part of the built-in
        pooling function. layer_input" is a rectangular numpy array of nodes. 
        '''
        
        self.layer_input = []
        self.filters = filters
        self.pooling_filter = pooling_filter
        
        # Vectors of grids, one for each feature map. These are populated in a later function, when we have access to the layer input. 
        
        self.grids = []
        self.pooled_grids = []

        # When we pool the data in the layer, we need to keep track of the index of the maximum value at each step of the pooling filter. 
        # This is an array of matrices, one for each filter map. Each matrix is the same size as the output of the pooling layer, and each element
        # is an ordered pair that gives the coordinates (in terms of the pre-pooled layer) of the largest value.         
        self.max_vals = []
        
        # array of error vectors - one vector for each feature map. The weights and error are included as part of the filter.
        self.error = []
        self.poolerror = []
        
        # This is an array of matrices. Each matrix holds the error associated with each site in the convolutional layer.
        # Most values here will be zero, except for the ones that correspond to the largest value in each pooling region. 
        self.error_mats = []

        # This is an array of matrices, one for each feature map. Each matrix holds a running sum of the derivative of the cost function with
        # respect to the weights of the corresponding filter. This is used in gradient descent.
        self.weight_sum = []

        # Similar to above, but for biases instead of weights. 
        self.bias_sum = []

    # This populates the arrays above with matrices of zeros. The layers in the network are all created at the same time, before any calculations 
    # are done, so we can't populate the arrays when the layer is created, because we don't know what size they need to be. The solution is to use this
    # function, which is called by the network class after the layers are created. We really should have a similar function for each layer type, but 
    # since I only ever plan to have a convolutional layer first, I'm skipping that.  
    def populate_arrays(self, first_layer_input):
        for f in self.filters:

            self.error.append(np.zeros(np.shape(f)))
            self.poolerror.append(np.zeros(np.shape(f)))
            self.error_mats.append(np.zeros((f.width, f.height)))
            self.weight_sum.append(np.zeros(np.shape(f.weights)))
            self.bias_sum.append(np.zeros(np.shape(f.bias)))

            # The new layer is a grid of nodes. The size of the layer is determined by the size of the input, the filter, and the stride length.
            # For example, if we have a 28x28 input and a 5x5 filter moved with stride length 1, we are left with a 24x24 layer, since we can 
            # only move the filter 23 units in each direction before reaching the edge of the input. Here, we only use stride length 1.          
            width = np.shape(first_layer_input)[0]-f.width+1
            height = np.shape(first_layer_input)[1]-f.height+1

            self.grids.append(np.zeros((width, height)))

            # The size of these matrices is determined by the size of the pooling filter and the grid of nodes we defined previously. We split that grid
            # into equally spaced regions the size of the pooling filter and, iterating across the colvolved image, take the max value in each region. 
            self.pooled_grids.append(np.zeros((int(width/self.pooling_filter.width), int(height/self.pooling_filter.height))))
            self.max_vals.append(np.zeros((int(width/self.pooling_filter.width), int(height/self.pooling_filter.height)), dtype=int))

    
    # Given a matrix with the single-valued index for the location of the greatest value in each submatrix of the convolutional layer,
    # return the coordinates of each greatest value, in terms of the coordinates of the convolutional layer. 
    # sub_matrix_shape should be (rows, columns)
    def full_indices(self, matrix):
        x = np.shape(matrix)[0]
        y = np.shape(matrix)[1]

        fx = self.pooling_filter.width
        fy = self.pooling_filter.height

        m = np.empty((x,y), dtype = tuple)

        for i in range(x):
            for j in range(y):

                # x and y indices within each submatrix
                ux = np.unravel_index(matrix[i][j], (fx,fy))[0]
                uy = np.unravel_index(matrix[i][j], (fx,fy))[1]
                
                m[i][j] = (ux + fx*i, uy + fy*j)

        return m
    
    # Used with full_indices to pass error backwards through pooling. Given a matrix with the error for the max values ("data") from each pooling 
    # region and a matrix with the indices of those max values ("locations"), this will return a matrix (with a shape that should match an element 
    # of self.grids) that contains the error values at the locations corresponding to the max values and zero everywhere else. 
    def unpool(self, data, locations, size):
        m = np.zeros(size)
        data = np.reshape(data, (-1,1))
        locations = np.reshape(self.full_indices(locations), (-1,1))
        
        for n,x in enumerate(locations):
            i = x[0][0]
            j = x[0][1]

            m[i][j] = data[n]

        return m


    # This is the output that would be fed into a pooling layer. Since I have combined the convolutional layer and the pooling layer, 
    # this function is not the final output of the layer. 
    #
    # I changed all self.layer_input[0] to self.layer_input. This may need to be changed back, although it seems like it fixed the proble.
    def pre_output(self):
        for n,f in enumerate(self.filters):
            filter_width = f.width
            filter_height = f.height

            # The new layer is a grid of nodes. The size of the layer is determined by the size of the input, the filter, and the stride length.
            # For example, if we have a 28x28 input and a 5x5 filter moved with stride length 1, we are left with a 24x24 layer, since we can 
            # only move the filter 23 units in each direction before reaching the edge of the input. Here, we only use stride length 1. 
            width = np.shape(self.layer_input)[0]-filter_width+1
            height = np.shape(self.layer_input)[1]-filter_height+1

            for j in range(width):
                for k in range(height):
                    
                    s=0
                    for l in range(filter_width):
                        for m in range(filter_height):
                            s += f.weights[l][m]* self.layer_input[j+l][k+m]
                    
                    self.grids[n][j][k] = relu(f.bias+s)

        return self.grids

    # I plan on always having a pooling layer after each convolutional layer, so instead of creating a new class, I"m just going to do the pooling 
    # in the convolutional layer class, using the unpooled output provided by the output function. 
    def output(self):

        standard_output = self.pre_output()
        pooling_filter_width = self.pooling_filter.width 
        pooling_filter_height = self.pooling_filter.height

        for n, g in enumerate(standard_output):
            pooled_width = int(np.shape(g)[0]/pooling_filter_width)
            pooled_height = int(np.shape(g)[1]/pooling_filter_height)

            for j in range(pooled_width):
                for k in range(pooled_height):

                    # Take the largest value in region selected by filter
                    m = g[j*pooling_filter_height:(j+1)*pooling_filter_height, k*pooling_filter_width:(k+1)*pooling_filter_width]
                    self.pooled_grids[n][j][k] = np.amax(m)
                    
                    # Don't forget that this needs to be unravelled later. 
                    self.max_vals[n][j][k] = m.argmax()

        return self.pooled_grids  

    # Given an error matrix from the fully connected layer, this returns the error in the each of the convolution kernels. 
    def error_output(self, error_in, weights):

        for g,f in enumerate(self.filters):

            e = self.unpool(np.multiply(np.reshape(np.matmul(weights.T, error_in), np.shape(d_sig(self.pooled_grids[g]))), d_sig(self.pooled_grids[g])), self.max_vals[g], np.shape(l_c.grids[g]))
            self.poolerror[g] = e

            for i in range(f.width):
                for j in range(f.height):
                    temp = 0

                    for n in range(f.width):
                        for m in range(f.height):

                            # i,j have been replaced with i+1,j+1. I think this fixes the weird behavior I was getting, and now the answers match the
                            # Mathematica calculation. However, I'm not totally sure, and the index conventions of the article I've been following 
                            # aren't consistent, so I can't understand exactly why this works. Using i,j is more sensible, but it completely breaks 
                            # d_relu(x) in Mathematica, so that makes me think that I'm really getting some sort of unannounced undefined behavior here 
                            # when I try it.
                            temp += e[i+1+m][j+1+n] * f.weights[m][n] * d_relu(self.grids[g][i][j])

                    self.error_mats[g][i][j] = temp
        
            # We need to rotate the final result. This is equivalent to taking the convolution with a flipped kernel.
            self.error_mats[g] = np.rot90(self.error_mats[g], 2)

        return self.error_mats

    def gradients(self):
        for k,f in enumerate(self.filters):

            for m in range(f.height):
                for n in range(f.width):
                    temp = 0

                    # I think using the dimensions of f here shouldn't cause any issues, but if issues come up, this is a place to look.
                    for i in range(f.height):
                        for j in range(f.width):
                            # There's another instance of adding 1 to i,j here. Again, not sure why this should be the case. 
                            # Another change from self.layer_input[0] to self.layer_input
                            temp += self.error_mats[k][i][j] * self.layer_input[i+1+m][j+1+n]

                    self.weight_sum[k][m][n] += temp 

            self.bias_sum += np.sum(self.error_mats[k])

        return self.weight_sum, self.bias_sum

    def update_parameters(self, rate, batch_size):
        for n,f in enumerate(self.filters):
            f.weights -= (rate/batch_size)* self.weight_sum[n]
            f.bias -= (rate/batch_size)* self.bias_sum[n]

    def reset_gradient_sums(self):
        for n,f in enumerate(self.filters):
            self.weight_sum[n] = np.zeros(np.shape(f.weights))
            self.bias_sum[n] = np.zeros(np.shape(f.bias))
        
class Network(object):
    def __init__(self, layers, first_layer_input):
        """
        "layers" is an array that holds layer objects of various types. The last element should be 10-dimensional, fully-connected layer. 
        """

        self.layers = layers

        # This populates various arrays with matrices of zeros. It needs to be done here because some of the matrix sizes depend on information that is
        # not available to the layer at the time of its creation. 
        self.layers[0].populate_arrays(first_layer_input)

        # Array that stores predicted digit for each input in test data
        self.test_outputs = []

    # def cost(self, x, lam, weights, n):
    #     # weights must include all weights in the network.
    #     # We use L2 regularization to help with overfitting.
    #     return -np.log(x)+np.square(weights).sum()/(2.0*n)

    def train(self, training_data, epochs, batch_size, rate, test_data=None, validation_data=None):
        
        # A subset of data that we can compare with after each batch to check progress. 
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)        

        # if validation_data:
        #     validation_data = list(validation_data)
        #     n_val = len(validation_data)
        
        # Data is list of tuples (x,y), where x is a 28x28-dimensional numpy array that holds the input image and y is a 10-dimensional
        # numpy array that indicates the correct number that corresponds to the image. 
        n_train = len(training_data)

        # Repeatedly train on shuffled dataset 
        for i in range(epochs):

            # Reset output vector for each epoch so that we can see how learning is improving
            self.test_outputs = []

            # Randomly shuffle training data and partition into subsets for batch training
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size] for j in range(0, n_train, batch_size)]
            
            #for batch in batches:
            for n,batch in enumerate(batches):
                print("Batch ", n+1, '/', len(batches))
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
            
            # Iterate over layers after the input
            for k in range(1,len(self.layers)):
                layer = self.layers[k]
                prev_layer = self.layers[k-1]
                layer.layer_input = prev_layer.output()

            return layer.output()

    # used to pass output error backwards through the network
    def backwards_pass(self, last_layer_error):

        self.layers[-1].error = last_layer_error

        # iterate over all layers except for the first. The error in the last layer is already
        # calculated in backpropagate. 
        for l in range(len(self.layers)-2,-1,-1):
            layer = self.layers[l]
            next_layer = self.layers[l+1]
            layer.error = layer.error_output(next_layer.error, next_layer.weights)

        return layer.error

    def backpropagate(self, batch, rate):
        m = len(batch)
        
        for image, answer in batch:    

            # pass an image through the network                
            result = self.forward_pass(image)
            layer = self.layers[-1]

            # The final layer's activation error needs to be calculated so we can compare to the desired result. 
            delta = layer.error_output(result, answer)

            # backpropagate error for each layer and caluclate gradient of cost function with respect to all weights and biases in the network
            self.backwards_pass(delta)               
            self.calculate_gradients()

        # update weights and biases using previously calculated gradients
        self.update_network(rate, m)

    def calculate_gradients(self):
        for l in self.layers:
            l.gradients()

    # update weights and biases and reset sums for next batch
    def update_network(self, rate, batch_size):
        for layer in self.layers:
            layer.update_parameters(rate, batch_size)
            layer.reset_gradient_sums()

[training_data, validation_data, test_data] = load_data()
training_data = list(training_data)

f = [Filter(5,5)]
pf = Filter(2,2)
l_c = ConvolutionalLayer(f, pf)

l_fc = Layer(10, 144)
l_s = Softmax_layer(10, 10)

net = Network([l_c, l_fc, l_s], training_data[0][0])
net.train(training_data, 30, 10, 0.1, test_data)