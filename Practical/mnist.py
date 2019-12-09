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
    sx = ex/np.sum(ex)
    if np.isnan(sx).any():
        raise TypeError('Over/underflow in softmax function!')
    return sx

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(-x)
    if np.isnan(ex).any():
        raise TypeError('Over/underflow in sigmoid function!')
    return np.divide(1, ex+1)

def lancelu(x):
    x=np.maximum(0, x)
    x=np.minimum(1000, x)
    return x


'''
Jacobian matrices of activation functions
'''
# aka matrices of partial derivatives

def softmax_jacobian(x):
    sx = softmax(x)
    dim = np.repeat(np.shape(sx), 2) #dimension of desired output
    return np.diag(sx)-np.kron(sx,sx).reshape(dim)

def relu_jacobian(x):
    return np.diag(x>=0)

def sigmoid_jacobian(x):
    x = np.float128(x) #to prevent over/underflow
    ex = np.exp(-x)
    if np.isnan(ex).any():
        raise TypeError('Over/underflow in jacobian of sigmoid function!')
    return np.diag(np.divide(ex, (ex+1)**2))

def lancelu_jacobian(x):
    return np.diag((x >= 0) * (x <= 1000)) #elementwise multiplication

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
        rand_w = (np.random.rand(np.prod(dim))-0.5)/10 #generate random weights
        self.w = rand_w.reshape(np.flip(dim)) #weights initialised randomly
        self.b =  (np.random.rand(np.prod(dim[1])) - 0.5) / 10  # biases initialised randomly
        if activation not in ['ReLU', 'Softmax', 'Sigmoid', 'LanceLU']: # currently supported activation functions
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
        if np.isnan(w).any():
            print(w)
            raise TypeError('Non number values in weight matrix!')
        if np.shape(w)== np.shape(self.w):
            self.w = w
        else:
            raise TypeError('Input weight matrix has wrong dimension!')

    def set_biases(self,b):
        if np.isnan(b).any():
            print(b)
            raise TypeError('Non number values in bias vector!')
        if np.shape(b) == np.shape(self.b):
            self.b = b
        else:
            raise TypeError('Input bias vector has wrong dimension!')

    '''
    Forward and backward pass
    '''
    def linear_map(self,x):
        return np.matmul(self.w,x) + self.b #W*x+b

    def activation(self,y):
        if self.a == 'ReLU':
            return relu(y)
        elif self.a == 'Softmax':
            return  softmax(y)
        elif self.a == 'Sigmoid':
            return sigmoid(y)
        elif self.a == 'LanceLU':
            return lancelu(y)
        else:
            raise TypeError('Invalid activation function!')

    def jacobian_activation(self, y):
        if self.a == 'ReLU':
            return relu_jacobian(y)
        elif self.a == 'Softmax':
            return  softmax_jacobian(y)
        elif self.a == 'Sigmoid':
            return sigmoid_jacobian(y)
        elif self.a == 'LanceLU':
            return lancelu_jacobian(y)
        else:
            raise TypeError('Invalid activation function!')

    '''
    Reinitialise weights
    '''
    def shuffle(self):
        epsilon =
        rand_w = np.random.normal(size = self.dim)
        self.w = self.w + rand_w.reshape(np.flip(self.dim)) #weights initialised randomly
        self.b = self.b + np.random.normal(size = self.dim[1])  # biases initialised randomly

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

    def show(self):
        for i in range(len(self.layers)):
            print('Layer %s' % i)
            print(self.layers[i].show())

    def shuffle(self): #reinitialise weights in case gradient during training is zero
        print('Everyday Im shuffling')
        for i in range(len(self.layers)):
            self.layers[i].shuffle()

'''
Model
'''
class Model(object):

    def __init__(self, NN, loss = 'MSE', learning_rate = 1.e-2):
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
        #split the data into batches of equal size
        #while not converged and within number of epochs
                # for each batch
                    # for each variable starting with the latest bias, then latest weight matrix and so on.
                        # for each datapoint in batch
                            #forwardpass the datapoint through neural net
                            #compute gradient using backprop
                        #do gradient descent

        if len(data) != len(label):
            raise TypeError('Number of inputs and labels do not coincide!')

        if batches <= 0 or not isinstance(batches, int):
            raise TypeError('Number of batches must be a positive integer')

        converged = False
        epoch = 1 #counter

        while not converged and epoch <= epochs:
            print('=== Epoch === %s' % epoch)
            batch_order = np.random.permutation(len(data)) #shuffles the order in which we loop over the data (len(data) is the number of train datapoints)
            batch_size = int(len(data)/batches)
            for b in range(batches): #for each batch
                grad = [None]*batch_size #initialise gradients for backprop
                sum_norm_grad = 0
                for l in range(len(self.NN.layers)-1, -1, -1):  # loop over layers backwards (for backprop)
                    for p in range(batch_size): #for each datapoint in batch
                        index = batch_order[b * batch_size + p]  # index of batch datapoint in overall dataset
                        if l == len(self.NN.layers)-1: #if last layer of neural net
                        #if l == 1:  # if last layer of neural net
                            x = data[index]  # takes datapoint in batch
                            y = label[index]  # takes label of datapoint
                            out = self.NN.forwardpass(x)  # forwardpass through the neural net
                            #out = NN.forwardpass(x)  # forwardpass through the neural net
                            if np.isnan(out[-1]).any():  # tests if there are any non-numbers resulting from this computation #suffices to check output
                                raise TypeError('Non-numerical value in output of neural net!')
                            grad[p] = 2/batch_size * (out[-1] - y) # gradient of loss function (MSE) at x  -------- should be updated for more general loss functions
                            #need to check whether above is correct
                        '''
                        Backpropagation
                        '''
                        jac_activ = self.NN.layers[l].jacobian_activation(out[l])
                        if np.isnan(jac_activ).any():
                            raise TypeError('Over/underflow in Jacobian computation')
                        grad[p] = np.matmul(grad[p], jac_activ)
                        #grad[p] = np.matmul(grad[p], NN.layers[l].jacobian_activation(out[l]))

                    '''
                    Stochastic gradient descent
                    '''
                    new_biases = self.NN.layers[l].biases() - self.learning_rate * sum(grad) #gradient descent step for biases
                    self.NN.layers[l].set_biases(new_biases) #set new biases
                    grad = np.matmul(grad, self.NN.layers[l].weights()) #backpropagation step again
                    new_weights = self.NN.layers[l].weights() - self.learning_rate * sum(grad) #gradient descent step for weights
                    self.NN.layers[l].set_weights(new_weights)  # set new weights

                sum_norm_grad += np.linalg.norm(sum(grad)) #takes sum of gradient norm over all batches
            print('--- Sum norm grad --- %s' % sum_norm_grad) # reports sum of gradient norm over all batches
            if sum_norm_grad < 0: #Currently not used: criterion for convergence: sum of norm of gradient for all batches has norm lower than threshold
                converged = True
            elif sum_norm_grad ==0:
                self.NN.shuffle()
            self.stats(data,label) # prints model accuracy statistics #can optimise this by leveraging the computations done here
            self.save() #save parameters
            epoch +=1

    def total_loss(self, data, label):         #computes total loss using MSE
        total_loss = 0
        for i in range(len(data)):
            x = data[i]
            y = label[i]
            NNx = self.NN.output(x)
            total_loss += np.linalg.norm(NNx-y)**2
        return total_loss/len(data)

    def correctly_classified(self, data, label): #computes number of correctly classified digits
        c_classified = 0
        for i in range(len(data)):
            x = data[i]
            y = label[i]
            NNx = self.NN.output(x)
            if np.argmax(NNx) == np.argmax(y):
                c_classified += 1
        return c_classified

    def model_accuracy(self, data, label): #computes classification accuracy of neural network
        return self.correctly_classified(data,label)/len(data)

    def stats(self, data, label): # a combination of the functions above, which runs faster than all separately
        total_loss = 0
        c_classified = 0
        for i in range(len(data)):
            x = data[i]
            y = label[i]
            NNx = self.NN.output(x)
            total_loss += np.linalg.norm(NNx-y)**2/len(data) #Mean squared error
            if np.argmax(NNx) == np.argmax(y):
                c_classified += 1
        accuracy = c_classified / len(data)
        print('--- Total loss --- %s' % total_loss)
        print('--- Correctly classified --- %s' % c_classified)
        print('--- Training accuracy --- %s' % accuracy)
        return total_loss, c_classified, accuracy


    def save(self): #saves model parameters so that training can be resumed another time
        for l in range(len(self.NN.layers)):
            layer = self.NN.layers[l]
            w_name = 'weight %s' % l
            np.save(w_name, layer.weights())
            b_name = 'bias %s' % l
            np.save(b_name, layer.biases())
        for l in range(len(self.NN.layers)):
            a_name = 'activation %s' % l
            np.save(a_name, layer.show_activation())



'''
MAIN
'''
# OPTIONAL: set random seed to get reproducible results
seed=100 # any number works
np.random.seed(seed)

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Process data
x_train = x_train.reshape([60000,784]) #set each datapoint as a vector instead of a matrix
y_t = y_train
y_train =np.zeros([len(y_train),10])
for i in range(len(y_train)):
    y_train[i, y_t[i]] = 1     #set each label as a vector of 10 entries (one for each digit) with one at the corresponding digit

# Neural network architecture
diml1 = np.array([784,200]) #dimension first layer (input, output)
diml2 = np.array([200,100]) #dimension second layer
diml3 = np.array([100,50])
diml4 = np.array([50,10])

# Create layers
l1 = Layer(diml1, 'LanceLU') #first layer
l2 = Layer(diml2, 'LanceLU') #second layer
l3 = Layer(diml3, 'LanceLU')
l4 = Layer(diml4, 'LanceLU')

# Create neural network
layers =[l1,l2,l3,l4]
NN = NeuralNetwork(layers)

# Show neural network weights
NN.show()

#Specify learning rate
r= 1

# Create model
M = Model(NN, 'MSE', r)

#Set number of training epochs
epochs = 100

# Set number of batches for training
batches = 200

# Train model
M.train_model(x_train, y_train, epochs, batches)

# Save model
M.save()