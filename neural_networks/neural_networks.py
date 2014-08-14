#!/usr/bin/python
# Neural network learning
# 
# Implementation of the feedforward and backpropagation algorithms for neural
# networks learning and the application of hand-written digit recoginition.

import time
import numpy as np
import scipy as sp
from sklearn import linear_model
from pylab import *
import random
import math

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoidGradient(z):
    g = np.multiply(sigmoid(z), (1. - sigmoid(z)))
    return g

def plotImage(X):
    size = np.shape(X)
    m = size[0]
    n = size[1]
    nrows = int(math.floor(sqrt(m)))
    ncols = int(math.ceil(m/nrows))
    npixels = int(sqrt(n)) # number of pixels per side
    #assert npixels*npixels == len(X[0]) 
    
    pad = 1 # between images padding
    display_array = np.ones((pad + nrows * (npixels + pad), pad+ncols*(npixels+pad)))
    
    curr_ex = 0
    for j in range(nrows):
        for i in range(ncols):
            nx = pad+j*(npixels+pad)
            ny = pad+i*(npixels+pad)
            
            #temp = np.reshape(X[curr_ex],(npixels,npixels),order='F') # order='F' for matlab style input file
            temp = np.reshape(X[curr_ex],(npixels,npixels))
            for k in range(npixels):
                display_array[nx+k][ny:(ny+npixels)] = temp[k]
            curr_ex += 1
            
    plt.ion()
    image = np.reshape(X[1],(npixels,npixels))
    plt.imshow(display_array, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.show()
    raw_input('Press enter to continue...')

#def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
def nnCostFunction(nn_params, *Z):
    X = Z[0]
    y = Z[1]
    n_input = Z[2] # input layer size
    n_hidden = Z[3] # hidden layer size
    num_labels = Z[4] # number of labels
    lamb = Z[5]
    J_or_G = Z[6] # return J or Gradient

    Theta1 = nn_params[:((n_input+1)*n_hidden)].reshape(n_hidden,n_input+1) # n_hidden x (n_input+1)
    Theta2 = nn_params[((n_input+1)*n_hidden):].reshape(num_labels,n_hidden+1) # n_labels x (n_hidden+1)

    m = len(X) # number of training examples
    
    J = 0
    Theta1_grad = zeros(Theta1.shape)
    Theta2_grad = zeros(Theta2.shape)
    
    # Feedforward

    X = np.c_[ np.ones(m), X ] # add a column of intercept. X: m x (n_input+1)
    a2 = sigmoid(X.dot(Theta1.T)) # a2: m x n_hidden
    
    a2 = np.c_[ np.ones(m), a2 ] # m x (n_hidden+1)
    h = sigmoid(a2.dot(Theta2.T)) # h: m x num_labels
    
    for c in range(num_labels):
        y2 = (y == c)
        J -= 1. / m * (np.log(h.T[c,:]).dot(y2) + np.log(1-h.T[c,:]).dot(1-y2))
    
    th1sq = np.multiply(Theta1, Theta1)
    th2sq = np.multiply(Theta2, Theta2)
    th1sq[:,0] = 0
    th2sq[:,0] = 0

    J += lamb / (2. * m) * (np.sum(th1sq)+np.sum(th2sq))
    
    # Backpropogation
    #start = time.time()
    for t in range(m):
        a1 = X[t,:] # a1 is a row vector, 1 x (n_input+1)
        z2 = a1.dot(Theta1.T) # z2: 1 x n_hidden
        a2 = sigmoid(z2) # row vector

        a2 = np.concatenate([np.array([1]), a2]) # a2: 1x26
        z3 = a2.dot(Theta2.T) # z3: 1x10
        a3 = sigmoid(z3)
        
        d3 = np.zeros(num_labels)
        for c in range(num_labels):
            d3[c] = a3[c] - (y[t] == c)
    
        d2 = d3.dot(Theta2) # d2: 1 x (n_hidden+1)
        d2 = d2[1:] # d2: 1 x n_hidden
        
        d2 = np.multiply(d2,sigmoidGradient(z2))

        d3 = d3[np.newaxis,:] # Numpy's way to allow 1xn array to behave like matrix
        d2 = d2[np.newaxis,:]
        a2 = a2[np.newaxis,:]
        a1 = a1[np.newaxis,:]

        Theta2_grad += d3.T.dot(a2) # num_labels x (1+n_hidden)
        Theta1_grad += d2.T.dot(a1) # n_hidden x (1+n_input)
    
    #end = time.time()
    #print "Elasped time:", end-start
    
    Theta1_grad /= m
    Theta2_grad /= m
    
    # regularization
    reg1 = lamb / m * Theta1
    reg2 = lamb / m * Theta2
    reg1[:,0] = 0
    reg2[:,0] = 0
    Theta1_grad += reg1
    Theta2_grad += reg2
    
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    
    if J_or_G == 'J':
        return J
    elif J_or_G == 'G':
        return grad
    
def J_func(nn_params, *Z):
    Y = Z + ('J',)
    J = nnCostFunction(nn_params, *Y)
    return J

def G_func(nn_params, *Z):
    Y = Z + ('G',)
    grad = nnCostFunction(nn_params, *Y)
    return grad

def randInitializeWeights(L_in, L_out):
    # Randomly initialize the weights of a layer with L_in incoming connections
    # and L_out outgoing connections to break the symmetry while training
    # the neural network
    
    epsilon_init = 0.12
    W = np.random.random_sample((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1+fan_in))
    W = W.flatten()
    nW = fan_out * (1+fan_in)
    for i in range(nW):
        W[i] = np.sin(i+1.)
    W = W.reshape(fan_out, 1+fan_in)
    W = W/10.
    return W

def computeNumericalGradient(nn_params, *Z):
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    
    e = 1e-4
    print nn_params.shape
    for p in range(len(nn_params)):
        perturb[p] = e
        loss1 = J_func(nn_params-perturb, *Z)
        loss2 = J_func(nn_params+perturb, *Z)
        numgrad[p] = (loss2-loss1) / (2*e)
        perturb[p] = 0
    return numgrad    

def checkNNGradients(lamb):
    # Create a small neural network to check the backpropagation gradients
    n_input = 3
    n_hidden = 5
    num_labels = 3
    m = 5
    
    # Generate random data for test
    Theta1 = debugInitializeWeights(n_hidden, n_input)
    Theta2 = debugInitializeWeights(num_labels, n_hidden)
    
    X = debugInitializeWeights(m, n_input-1)
    y = [0]*m
    y = np.array(y)
    for i in range(m):
        y[i] = (i+1) % num_labels + 1.
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    Z = (X, y, n_input, n_hidden, num_labels, lamb)

    grad = G_func(nn_params, *Z)

    numgrad = computeNumericalGradient(nn_params, *Z)
    print 'Numerical gradient, analytical gradient:'
    for i in range(len(grad)):
        print numgrad[i],grad[i]
        
    return

def nnPredict(Theta1, Theta2, X):
    # Predict the label of an input given a trained neural network
    m = len(X)
    num_labels = len(Theta2)
    
    X = np.c_[ np.ones(m), X ]
    h1 = sigmoid(X.dot(Theta1.T))
    h1 = np.c_[ np.ones(m), h1]
    h2 = sigmoid(h1.dot(Theta2.T))
    p = np.argmax(h2, axis=1) # p is 1-d array
    return p

def checkNNCost(*Z):
    # Loading pre-initialized parameters
    print "Loading neural network parameters..."
    params_data = scipy.io.loadmat('nn_weights.mat')
    
    # Let label 0, instead of label 10, represent digit 0
    Theta2 = params_data['Theta2']
    Theta2 = np.concatenate((Theta2[9,:],Theta2[:8,:]),axis=0)
    params_data['Theta2'] = Theta2
    temp_params = np.array([ params_data['Theta1'],params_data['Theta2'] ])

    nn_params = np.concatenate((temp_params[0].flatten(), temp_params[1].flatten())) # flatten to a 1d array
    
    print '\nFeedforward using neural network...\n'
    
    print 'Checking cost function without regularization...'
        
    J = J_func(nn_params, *Z)
    
    print 'Cost at parameters (loaded from ex4weights): ', J
    print '(this value should be about 0.287629)\n'
    
    # Add regularization
    print 'Checking cost function with regularization...'
    lamb = 1.

    Z = (X, y, input_layer_size, hidden_layer_size, num_labels, lamb)
    
    J = J_func(nn_params, *Z)
    
    print 'Cost at parameters (loaded from ex4weights): ', J
    print '(this value should be about 0.383770)\n'
    
    # Checking gradient
    print 'Checking gradient...'
    checkNNGradients(lamb)
    
   
def parse(train_file,test_file):
    # load training data and test data
    train_data = np.genfromtxt(train_file, delimiter=',',skip_header=1)
    test_data = np.genfromtxt(test_file, delimiter=',',skip_header=1)
    ncols = len(train_data[0])
    
    input_layer_size = ncols - 1
    scale = 255. # scaling factor for X to avoid exponential overflow
    
    np.random.shuffle(train_data)
    
    m = len(train_data)
    
    # Divide training data to training part and cross validation part
    m1 = int(m*0.7) # number of training samples, m-m1: number of cross validation samples
    
    # Train the classifier based on smaller samples
    #sample = np.array(random.sample(train_data[:m1],5000))
    #y_train = np.array(map(int,sample[:,0]))
    #X_train = sample[:,1:] / scale # scaling down to avoid
    
    # Train the classifier based on the first m1 examples
    y_train = np.array(map(int,train_data[:m1,0]))
    X_train = train_data[:m1,1:] / scale
    
    y_cross = np.array(map(int,train_data[m1:,0]))
    X_cross = train_data[m1:,1:] / scale
    
    # Train the classifier based on all training examples
    #y_train = np.array(map(int,train_data[:,0]))
    #X_train = train_data[:,1:] / scale # scaling down to avoid exponential overflow
    
    X_test = test_data / scale
    input_data = {}
    input_data['y'] = y_train
    input_data['X'] = X_train
    input_data['y_cross'] = y_cross
    input_data['X_cross'] = X_cross
    input_data['X_test'] = X_test
    
    return input_data

def sample(train_file):
    train_data = open(train_file, 'r')
    train = ''.join(train_data.readlines())
    train_data.close()

    lines = train.split('\n')
    m1 = len(train)
    train_sample = random.sample(lines[1:m1],5000)
    output_data = lines[0].rstrip() + '\n'
    for i in range(5000):
        output_data += ''.join(train_sample[i].rstrip()) + '\n'
        
    with open('test_sample.csv', 'w') as f:
        f.write(output_data)
    f.closed

def solve_it(input_data):
    ### Arrange data   
    X = input_data['X'] # training data
    y = input_data['y'] # training data
    #X_test = input_data['X_test'] # test data
    
    m = len(X) # number of training examples
    input_layer_size = len(X[0]) # number of pixels per image
    hidden_layer_size = 100
    num_labels = 10 # 10 labels, from 0 to 9
    
    print "m = ", m
    print "Size of input layer:", input_layer_size
    print "Size of hidden layer:", hidden_layer_size
    
    # Display images of 100 random samples from the training examples
    plotImage(random.sample(X,100))
    
    lamb = 0. # lambda for regularization  
    Z = (X, y, input_layer_size, hidden_layer_size, num_labels, lamb)
    
    # Check cost function with pre-initialized parameters
    #checkNNCost(*Z)
    
    # Implement a function to initialize the weights of the neural network
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    
    initial_nn_params = [initial_Theta1, initial_Theta2]
    initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))

    # Backpropagation
    lamb = 1.0
    Z = (X, y, input_layer_size, hidden_layer_size, num_labels, lamb)
    print 'lambda =', lamb
    
    print 'Minimizing cost function...'
    start = time.time()
    
    # Timing test for calling J_func and G_func
    #for i in range(1):
    #    J = J_func(initial_nn_params, *Z)
    #    grad = G_func(initial_nn_params, *Z)
    #end = time.time()
    #print 'J =', J
    #print 'Elasped time:',end-start
    #return
   
    # Use SciPy's fmin_l_bfgs_b for optimization. It is much faster than fmin_cg.
    # For some reason, fmin_bfgs is not able to get the result for this case,
    # probabaly because of the large training data.
    #
    # Default parameters:
    # scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(),
    # approx_grad=0, bounds=None, m=10, factr=10000000.0, pgtol=1e-05,
    # epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=None,
    # callback=None)
    
    nn_params, J = sp.optimize.fmin_l_bfgs_b(J_func, x0=initial_nn_params, fprime=G_func, args=Z, disp=1, maxiter=200)[:2]
    #nn_params, J = sp.optimize.fmin_cg(J_func, x0=initial_nn_params, fprime=G_func, args=Z, maxiter=500, full_output=True, disp=True)[:2]
    end = time.time()
    
    #print J
    
    print "Elapsed time:",end-start,'\n'
    
    Theta1 = nn_params[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[((input_layer_size+1)*hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)
    
    # Calculate training accuracy
    
    pred = nnPredict(Theta1, Theta2, X)
    print np.shape(pred)
    print np.shape(y)
    print pred
    print y
    print 'Training set accuracy:', np.mean((pred == y.flatten()).astype(int)) * 100,'%'

    if ('X_cross') in input_data:
        X_cross = input_data['X_cross']
        y_cross = input_data['y_cross']
        pred = nnPredict(Theta1, Theta2, X_cross)
        print np.shape(pred)
        print np.shape(y_cross)
        print pred
        print y_cross
    #print pred
    #print y.flatten()
        print 'Cross validation set accuracy:', np.mean((pred == y_cross.flatten()).astype(int)) * 100,'%'

    # visualize neural network
    plotImage(Theta1[:,1:])

    # Predict labels of test data
    if ('X_test') in input_data:
        X_test = input_data['X_test'] # test data
        pred = nnPredict(Theta1, Theta2, X_test)
        output_data = '\n'.join(map(str,pred))
  
    return output_data

import sys
import scipy.io

if __name__ == '__main__':
    if len(sys.argv) > 2:
        # load training and test data
        train_file = sys.argv[1].strip()
        test_file = sys.argv[2].strip()
        print 'Loading data...', train_file, test_file
        #sample(test_file)
        input_data = parse(train_file,test_file)
        output_data = solve_it(input_data)
        with open('y_test.txt', 'w') as f:
            f.write(output_data)
        f.closed
    elif len(sys.argv) >1:
        # load only training data
        file_location = sys.argv[1].strip()
        print 'Loading data...', file_location
        input_data = scipy.io.loadmat(file_location)
        y = input_data['y']
        # in nn_data.mat, label 10 represented digit 0
        for i in len(y):
            if y[i,0] == 10: y[i,0] = 0 # let label 0 represent digit 0
        input_data['y'] = y
        solve_it(input_data)
    else:
        print 'This example requires at lease one input files. (For example: python neural_networks.py train.csv test.csv)'

