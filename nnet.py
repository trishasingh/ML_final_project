# CS 451 HW 3
# Neural Net with backpropagation

import random, math
import numpy as np

class Network(object):

    def __init__(self, sizes, debug=False):
        """
        Construct a new neural net with layer sizes given.  For
        example, if sizes = [2, 3, 1] then it would be a three-layer 
        network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized randomly.
        If debug=True then repeatable "random" values are used.
        biases and weights are lists of length sizes-1.
        biases[i] is a column vector for layer i+1.
        weights[i] is a matrix for layers [i] and [i+1].
        """
        self.sizes = sizes
        self.debug = debug
        
        ### FILL IN #4: 
        # use rand_mat(r, c, debug) to initialize these with
        # the right sizes r x c

        self.biases =  [rand_mat(sizes[i+1], 1, debug) for i in range (len(sizes)-1)]
        self.weights = [rand_mat(sizes[i+1], sizes[i], debug) for i in range (len(sizes)-1)]

    def feedforward(self, a):
        """Return the output of the network if a is input"""

        ### FILL IN #5: 
        # from one layer to next, compute z = w*a+b, a = g(z)
        for i in range(len(self.weights)):
            z = np.add(np.dot(self.weights[i],a),self.biases[i])
            a = sigmoid(z)
        return a

    def train(self, train_data, valid_data, epochs, mini_batch_size, alpha):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``train_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``valid_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        
        self.report_accuracy(0, train_data, valid_data)

        # vectorize outputs y in training data ("one hot" encoding)
        ny = self.sizes[-1]
        #train_data_vec = [(x, unit(y, ny)) for x, y in train_data]
        train_data_vec = [(x, y) for x, y in train_data]
        m = len(train_data)
        for j in range(epochs):
            if not self.debug:
                random.shuffle(train_data_vec)
            # make sure we can evenly divide in batches
            remainder = len(train_data_vec)%mini_batch_size
            for i in range(0, len(train_data_vec)- remainder, mini_batch_size):
                self.update_mini_batch(train_data_vec[i:i+mini_batch_size], alpha)
            if remainder > 0:
                self.update_mini_batch(train_data_vec[-remainder:])
            ### FILL IN #9:
            # divide train_data_vec into batches (lists of size
            # mini_batch_size), and call self.update_mini_batch on each
            
            self.report_accuracy(j+1, train_data, valid_data)

        
    def update_mini_batch(self, mini_batch, alpha):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        mini_batch is a list of tuples (x, y), and alpha
        is the learning rate.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        nL = len(self.sizes)
        m = len(mini_batch)
        for x, y in mini_batch:
            

            ### FILL IN #7:
            # call self.backprop(x, y) and call the results
            # delta_b, delta_w
            # then add each of their elements to grad_b, grad_w
            delta_b, delta_w = self.backprop(x,y)
            grad_b = np.add(delta_b, grad_b)
            grad_w = np.add(delta_w, grad_w)
            
        # once you have the sums over the mini batch,
        # adjust the biases and weights by 
        # subtracting (alpha/m) times the gradients
        self.biases = np.subtract(self.biases, (alpha/m)*grad_b)
        self.weights = np.subtract(self.weights, (alpha / m) * grad_w)


    def backprop(self, x, y):
        """
        Return (grad_b, grad_w) representing the gradient of the cost 
        function for a single training example (x, y).  grad_b and
        grad_w are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        n = len(self.sizes) # number of layers
        
        # forward pass
        ### #6:
        # same as feedforward, but store all a's and z's in lists
        a = [0] * n
        z = [0] * n
        a[0] = x  # initial activation (z[0] is not used)
        for i in range(1, n): # 1 .. n-1
            b = self.biases[i-1]
            w = self.weights[i-1]
            z[i] = np.add(np.dot(w, a[i-1]), b)
            a[i] = sigmoid(z[i])
            
        # backward pass
            
        delta = [0] * n
        i = n-1 # index of last layer
        delta[i] = (a[i]-y) * sigmoid_grad(z[i])
        print(np.shape(delta[i]))
        for i in range(n-2, 0, -1): # n-2 .. 1
            delta[i] = np.dot(self.weights[i].T, delta[i+1])*sigmoid_grad(z[i]) # FILL IN

        # compute gradients
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for i in range(0, n-1):
            grad_b[i] = delta[i+1]
            #print(np.shape(delta[i+1]))
            #print(np.shape(a[i].T))
            grad_w[i] = np.dot(delta[i+1], a[i].T)

        return (grad_b, grad_w)
        


    def evaluate(self, data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        ### FILL IN #8# :
        numCorrect = 0
        for i in range(len(data)):
            res = self.feedforward(data[i][0])
            # if the max index is y, then correct
            if np.argmax(res) == data[i][1]:
                numCorrect+=1
        return  numCorrect
        # loop over (x, y) in data
        # run feedforward(x)
        # find the index of the maximum element in the output vector
        # if it is equal to y, it is correct
        # hint: use np.argmax

    
    def report_accuracy(self, epoch, train_data, valid_data):
        """report current accuracy on training and validation data"""
        tr, ntr = self.evaluate(train_data), len(train_data)
        te, nte = self.evaluate(valid_data), len(valid_data)
        print("Epoch %d: " % epoch, end='')
        print("train %d/%d (%.2f%%) " % (tr, ntr, 100*tr/ntr), end='')
        print("valid %d/%d (%.2f%%) " % (te, nte, 100*te/nte))

#### Helper functions

def sigmoid(z):
    """vectorized sigmoid function"""
    return 1/(1+np.exp(-z))
    #return np.zeros(0)  ### FILL IN #1  (hint: use np.exp)

def sigmoid_grad(z):
    """vectorized gradient of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))
    #return np.zeros(0)  ### FILL IN #2

def unit(j, n):
    """return n x 1 unit vector with 1.0 at index j and zeros elsewhere"""

    x = np.zeros((n,1))  ### FILL IN #3
    print(j)
    print()
    x[j] = 1
    return x

def rand_mat(rows, cols, debug):
    """
    return random matrix of size rows x cols; if debug make repeatable
    """
    eps = 0.12 # random values are in -eps...eps
    if debug:
        # use math.sin instead of random numbers
        vals = np.array([eps * math.sin(x+1) for x in range(rows * cols)])
        return np.reshape(vals, (rows, cols))
    else:
        return 2 * eps * np.random.rand(rows, cols) - eps
