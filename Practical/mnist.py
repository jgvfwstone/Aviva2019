'''
In this script, we demonstrate how a single hidden layer feedforward network can solve MNIST with high accuracy
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# OPTIONAL: set random seed to get reproducible results
# np.random.seed(20)

class Layer(object):
    #dim is a numpy array with dim[0] the dimension of input and dim[1] the dimension of the output
    def __init__(self, dim, activation):
        super(Layer, self).__init__() #contructs the object instance
        self.w = np.array([np.random.normal() for _ in range(np.prod(dim))]).reshape(np.flip(dim)) #weights
        self.b = np.array([np.random.normal() for _ in range(dim[1])]) #biases
        self.a = activation #0 corresponds to RELU, 1 to softmax

    def show(self):
        return self.w, self.b

    def w(self):
        return self.w()

    def b(self):
        return self.b()

    def a(self):
        return self.a()

    def matmul(self,x):
        return np.matmul(self.w,x)+self.b #A*x+b

    def output(self,x):
        y = self.matmul(x)
        if self.a ==0:
            return np.maximum(0,y) # returns ReLU(y)
        elif self.a ==1:
            ey= np.exp(y-np.mean(y))
            return  ey/np.sum(ey) # returns Softmax(y)


class NeuralNet(object): #one hidden layer neural network
    def __init__(self, dims, activations):
        super(NeuralNet, self).__init__() #constructs the object instance
        self.Layer1 = Layer(dims[0:2], activations[0]) #initialises layer 1
        self.Layer2 = Layer(dims[1:3], activations[1]) #initialises layer 2
        self.learning_rate = 1.e-3

    def show(self):
        return self.Layer1.show(), self.Layer2.show()

    def output(self,x):
        y = self.Layer1.output(x)
        return self.Layer2.output(y)

    def gradient(self,x): #computes the gradient of the neural net at datapoint x
        y1 = self.Layer1.matmul(x)
        if self.Layer1.a() ==0:
            y2 np.maximum(0,y1) # ReLU
        elif self.a ==1:
            ey= np.exp(y1-np.mean(y1))
            y2 = ey/np.sum(ey) # softmax
        y3 = self.Layer2.matmul(y2)






'''
CODE
'''

dims = np.array([5,4,3])

activations = np.array([0,1])

NN =NeuralNet(dims, activations)

x = np.array([1,2,3,4,5])

z= NN.output(x)

