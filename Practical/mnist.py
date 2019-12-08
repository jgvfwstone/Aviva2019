'''
Classifying MNIST with a feedforward neural network and vanilla stochastic gradient descent using numpy
Author: LDC
'''

'''
Dependencies
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf #only used to load MNIST


'''
Activation functions
'''
def softmax(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(x-np.mean(x)) #substract mean to prevent over/underflow
    return ex/np.sum(ex)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(-x)
    return np.divide(1, ex+1)


'''
Jacobian matrices of activation functions
'''
# aka matrices of partial derivatives

def softmax_jacobian(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(x)
    shape = np.repeat(np.shape(x),2)
    ex2 = -np.kron(ex,ex).reshape(shape)
    ex2 = ex2 +np.diag(ex*np.sum(ex))
    return ex2/(sum(ex)**2)

def relu_jacobian(x):
    return np.diag(x>=0)

def sigmoid_jacobian(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(-x)
    return np.diag(np.divide(ex, (ex+1)**2))

'''
Loss functions
'''

'''
Jacobian of loss functions
'''

'''
Layer
'''
class Layer(object):

    def __init__(self, dim, activation = 'ReLU'):
        super(Layer, self).__init__() #contructs the object instance
        self.dim = dim    #dim is a numpy array with dim[0] the dimension of input and dim[1] the dimension of the output
        self.w = np.array([np.random.normal() for _ in range(np.prod(dim))]).reshape(np.flip(dim)) #weights initialised from random normal distribution
        self.b = np.array([np.random.normal() for _ in range(dim[1])]) #biases "--"
        if activation not in ['ReLU', 'Softmax', 'Sigmoid']: # currently supported activation functions
            raise TypeError('Invalid activation function!')
        else:
            self.a = activation
    '''
    Methods to show attributes
    '''
    def show(self):
        return self.w, self.b, self.a

    def weights(self):
        return self.w

    def biases(self):
        return self.b

    def show_activation(self):
        return self.a

    def dimension(self):
        return self.dim
    '''
    Methods to set attributes
    '''
    def set_weights(self, w):
        if np.shape(w)== np.shape(self.w):
            self.w = w
        else:
            raise TypeError('Input weight matrix has wrong dimension!')

    def set_biases(self,b):
        if np.shape(b) == np.shape(self.b):
            self.b = b
        else:
            raise TypeError('Input bias vector has wrong dimension!')

    '''
    Forward and backward pass
    '''
    def linear_map(self,x):
        return np.matmul(self.w,x) + self.b #A*x+b

    def activation(self,y):
        if self.a == 'ReLU':
            return relu(y)
        elif self.a == 'Softmax':
            return  softmax(y)
        elif self.a == 'Sigmoid':
            return sigmoid(y)
        else:
            raise TypeError('Invalid activation function!')

    def jacobian_activation(self, y):
        if self.a == 'ReLU':
            return relu_jacobian(y)
        elif self.a == 'Softmax':
            return  softmax_jacobian(y)
        elif self.a == 'Sigmoid':
            return sigmoid_jacobian(y)
        else:
            raise TypeError('Invalid activation function!')

'''
Neural network
'''
class NeuralNetwork(object): #neural network
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__() #constructs the object instance
        #need to check that layers can be composed with each other
        for i in range(len(layers)-1):
            if layers[i].dimension()[1] != layers[i+1].dimension()[0]:
                raise TypeError('Layers have not been initialised to compatible dimensions: %s' % i,i+1)
        self.layers = layers #list of layers from first to last layer

    def forwardpass(self, x):
        #returns a list of intermediate values of the neural network (before each activation function) and final value
        out = [None] * (len(layers)+1)
        out[0] = self.layers[0].linear_map(x)
        for i in range(1,len(layers)):
            layer_i_minus_1 = self.layers[i-1].activation(out[i-1])
            out[i] = self.layers[i].linear_map(layer_i_minus_1)
        out[i+1]= self.layers[i].activation(out[i])
        return out

    def output(self, x):
        #returns the final value of the neural network
        return self.forwardpass(x)[-1]

'''
Model
'''
class Model(object):

    def __init__(self, NN, loss = 'MSE', learning_rate = 1.e-3):
        super(Model, self).__init__()  # contructs the object instance
        self.NN = NN
        if loss not in ['MSE']:  # currently supported loss functions: 'MSE' mean squared loss
            raise TypeError('Invalid loss function!')
        else:
            self.loss = loss # set loss function
        #can add training method as attribute. For now just vanilla gradient descent
        if learning_rate <= 0:
            raise TypeError('Learning rate must be strictly positive!')
        self.learning_rate = learning_rate

    def train_model(self, data, label, epochs, batches):
        #updates weights of neural network given dataset, loss function and optimisation method
        #data is understood to be training data
        #ALGORITHM:
        #split the data into batches
        #while not converged and smaller than total number of epochs
            # for each epoch
                # for each batch
                    # for each datapoint in batch
                        #forwardpass the datapoint through neural net
                        #for each variable starting with the latest bias, then latest weight matrix and so on.
                            #compute gradient
                            #do gradient descent
        if len(data) != len(label):
            raise TypeError('Number of inputs and labels do not coincide!')

        if batches <= 0 or not isinstance(batches, int):
            raise TypeError('Number of batches must be a positive integer')

        converged = False
        epoch = 1

        while not converged and epoch <= epochs:
            print('=== Epoch === %s' % epoch)
            batch_order = np.random.permutation(len(data)) #shuffles the order in which we loop over the data (len(data) is the number of train datapoints)
            batch_size = int(len(data)/batches)
            dp = 0 #data point index
            for b in range(batches): #for each batch
                grad = [None]*batch_size #initialise gradients for backprop
                max_sum_grad = 0
                for l in range(len(self.NN.layers)-1, -1, -1):  # loop over layers backwards (for backprop)
                    for p in range(batch_size): #for each datapoint in batch
                        index = batch_order[b * batch_size + p]  # index of current datapoint in the data
                        x = data[index]  # takes datapoint in batch ------------------------------
                        y = label[index]  # takes label of datapoint
                        out = self.NN.forwardpass(x)  # forwardpass through the neural net
                        '''
                        Backpropagation
                        '''
                        if l == len(self.NN.layers)-1: #last layer of neural net
                            grad[p] = 2/batch_size * (out[-1] -y) # gradient of loss function (MSE) at x  -------- should be updated for more general loss functions
                            #need to check whether above is correct
                        grad[p] = np.matmul(grad[p], self.NN.layers[l].jacobian_activation(out[l]))

                    new_biases = self.NN.layers[l].biases() - self.learning_rate * sum(grad) #gradient descent step for biases
                    self.NN.layers[l].set_biases(new_biases) #set new biases
                    grad = np.matmul(grad, self.NN.layers[l].weights())
                    new_weights = self.NN.layers[l].weights() - self.learning_rate * sum(grad)
                    self.NN.layers[l].set_weights(new_weights)  # set new weights
                    # dp +=1
                max_sum_grad = np.maximum(max_sum_grad, np.linalg.norm(sum(grad))) #takes maximum of gradient norm over all batches
            if max_sum_grad < 0: #criterion for convergence: gradient for all batches has norm lower than 1.e-5
                converged = True
            print('--- Max sum grad --- %s' % max_sum_grad)
            print('--- Total loss --- %s' % self.total_loss(data, label))
            epoch +=1

    def total_loss(self, data, label):
        #computes total loss using MSE:
        total_loss = 0
        for i in range(len(data)):
            x = data[i]
            y = label[i]
            NNx = self.NN.output(x)
            total_loss += np.linalg.norm(NNx-y)**2
        return total_loss/len(data)






'''
Actual code
'''
# OPTIONAL: set random seed to get reproducible results
# np.random.seed(20)

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Process data
x_train = x_train.reshape([60000,784])
y_t = y_train
y_train =np.zeros([len(y_train),10])
for i in range(len(y_train)):
    y_train[i, y_t[i]] = 1


# Create layers
l1 = Layer(np.array([784,100]), 'ReLU') #first layer
l2 = Layer(np.array([100,10]), 'ReLU') #second layer

# Create neural network
layers =[l1,l2]
NN = NeuralNetwork(layers)

# Create model
M = Model(NN)

#Number of training epochs
epochs = 10

# Number of batches
batches = 10

# Train model
M.train_model(x_train, y_train, epochs, batches)