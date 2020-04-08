# coding=UTF-8

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random


# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) 
        #size是一个列表=[2,3,1],　size[1:]从index=1开始

        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #biases[0]: 3*1, biases[1]: 1*1
        # bias是一个列表，中括号里的代码对这个list的每一个元素进行赋值 
        # np.random.randn(y, 1)：从标准正态分布中产生一个 y*1矩阵

        self.weights = [np.random.randn(y, x) #size[:-1]不包含最后一个元素
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #zip(): 接受一系列可迭代的对象（例如，在这里是两个list）作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list。
        #zip([3, 4], [5, 9]) 返回值是：[(3, 5), (4, 9)]
        #weights[0]: 3*2的矩阵， weights[1]: 1*3的矩阵，见公式22

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    #stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs): #默认步长为１, [0，epochs)
            random.shuffle(training_data) #把list里的元素进行了随机打乱
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)] #步长由第３个参数给出,[0, n)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #mini_batch: a list of tuples of (x,y)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #产生一个zero matrix list
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #由一个样本的cost计算对每层的权值和bias偏导，不是由一个batch的cost计算对每层权值和bias偏导
            #一个样本计算出来的，从第2层到第L层，Cx对bias，相邻两层权重矩阵的偏导
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #累加偏导数
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #累加偏导数
        
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]#累加一个batch中的所有样本m

    def backprop(self, x, y): #一个训练样本(x, y)
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #从第L层到第2层，Cx对bias，相邻两层权重矩阵的偏导
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer, [第2层，第3层，...第L层]输出
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) #计算a^2，a^3
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #输出层$\delta^L$
        nabla_b[-1] = delta #Cx对输出层bias的偏导
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Cx对输出层w矩阵的偏导

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers): #[2, 3)
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #计算输出层的$\frac{\partial C_x}{\partial a}$
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) #$C_x= \frac{1}{2}(y - a)^2$


#函数作用于一个矩阵等于对矩阵中的每个元素都进行该函数，比如对某个矩阵进行numpy.exp()


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
