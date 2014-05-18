#!/usr/bin/python
# Logistic regression

import time
import numpy as np
import scipy as sp
from sklearn import linear_model
from pylab import *

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def costFunction(theta, *Z):
    ### dim(X) = [m, n], m: number of examples, n: number of features
    ### dim(theta) = [n]
    X = Z[0]
    y = Z[1]
    m = len(y) # Number of training examples
    
    z = X.dot(theta) # dim(z) = [m, 1]
    h = sigmoid(z) # dim(h) = [m, 1]
    
    J = -1. / m * ((y.T.dot(np.log(h)) + (1 - y).dot(np.log(1-h))))
    #grad = 1. / m * (h - y).T.dot(X) # dim(grad) = [1, n+1]

    return J

def gradFunction(theta, *Z):
    ### dim(X) = [m, n], m: number of examples, n: number of features
    ### dim(theta) = [n]
    X = Z[0]
    y = Z[1]
    m = len(y) # Number of training examples

    z = X.dot(theta) # dim(z) = [m, 1]
    h = sigmoid(z) # dim(h) = [m, 1]
    #J = - (1. / m * (y.T.dot(log(h)) + (1 - y).dot(log(1-h))))
    grad = (1. / m * (h - y).T.dot(X)).T # dim(grad) = [1, n+1]
    return grad

def predict(theta, X):
    ### Predict whether the label is 0 or 1 using the learned theta values
    h = sigmoid(X.dot(theta))
    p = [ int (i) for i in (h >= 0.5) ]
    return p
    
def load(filename, delimiter=''):
    # This function read data from filename and returns a NumPy array
    input_data_file = open(filename, 'r')
    input_data = ''.join(input_data_file.readlines())
    input_data_file.close()
    lines = input_data.split('\n')
    lines.remove('')

    Z = []
    for line in lines:
        fields = [ float(field) for field in line.split(',') ]
        Z.append(fields)
    return np.array(Z)

def logistic(X, y, fit_intercept=True):
    ### Self-defined logistic regression function  
    theta = np.zeros(len(X[0]))
    X = np.c_[ np.ones(len(y)), X ] # Add a column of intercept
    theta = np.r_[0, theta] # Add intercept
    
    Z = (X, y)
    J = costFunction(theta, *Z)
    print 'Cost at initial theta (zeros): ', J, '\n'

    ### Minimize J using fmin_bfgs in SciPy
    print 'Minimizing cost function using scipy.optimize.fmin_bfgs...'
    
    theta, J = sp.optimize.fmin_bfgs(costFunction, x0=theta, fprime=gradFunction, args=(X,y), full_output=True)[0:2]
    #theta, J = sp.optimize.minimize(costFunction, x0=theta, args=(X,y), method='BFGS')
    print 'Minimized cost = ', J, '\n'
    
    return theta
    
def mapFeature(X1, X2, degree):
    ### Map the two input features to polynomial features for regularization
    for i in range(1, degree+1):
        for j in range(i+1):
            X_temp = np.power(X1, (i-j)) * (np.power(X2, j))
            if (i == 1) & (j == 0):
                X_new = X_temp  # initialize X_new
            else:
                X_new = np.c_[ X_new, X_temp ]  # add a new column
    return X_new
            
def solve_it(input_data):

    ### Arrange data
    X = input_data[:,:2]
    y = input_data[:,2]
    
    X[:,0] = (X[:,0] - np.mean(X[:,0]))/np.std(X[:,0])
    X[:,1] = (X[:,1] - np.mean(X[:,1]))/np.std(X[:,1])
    
    print "The first ten training examples: "
    print X[:10,:], '\n'
    
    ### Plot data
    print "Plotting data...", '\n'
    pos = np.where(y==1)
    neg = np.where(y==0)
    
    plt.ion() # interactive mode for plot, used with raw_input() 
    #subplot(311)
    scatter(X[pos,0], X[pos,1], marker='o')
    scatter(X[neg,0], X[neg,1], marker='x')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    plt.show()
    raw_input('Press enter to continue...')
    
    # Create additional features for regularization
    degree = 6
    X = mapFeature(X[:,0], X[:,1], degree)
    print X
    
    #return 0
    
    ### Do logistic regression using a cost function
    #start = time.time()
    theta = logistic(X, y, fit_intercept = True)
    #end = time.time()
    #print "elapsed time using self-defined logistic regression function: ", end - start
    
    print 'theta at minimum = ', theta, '\n'
    #print 'Minimized cost = ', J, '\n'
    
    ### Predict whether the label is 0 or 1 using learned theta values
    X1 = np.c_[ np.ones(len(y)), X ]
    print "Predicting using learned theta values on training examples..."
    p = predict(theta, X1)
    hits = [ int (i) for i in (p == y) ] # whether predicted value match y
    print "Mean accuracy = ", float(sum(hits))/len(y), '\n'
    
    ### Use sklearn's logistic regression module
    ### default: sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

    ### fit_intercept=True means adding a column of values 1 to X to calculate theta_0
    ### C is the inverse of lambda, which is the parameter for regularization
    ### default is l2 regularization
    
    print 'Solving with Scikit-learn (sklearn)... '
    
    #X = input_data[:,:2]
    
    #start = time.time()
    sol = linear_model.LogisticRegression(tol = 0.0001)
    sol.fit(X,y)
    
    #end = time.time()
    #print "elapsed time using Scikit-learn = ", end - start
    
    print 'Theta values from sklearn: (Intercept, theta_1, theta2... )'
    ### intercept_ is theta_0, coef_ is [ theta_1, theta_2, ...., theta_n ]
    print ' ', sol.intercept_
    #print sol.coef_
    for coef in sol.coef_: print ' ', coef
    
    ### Compute accuracy on the training set
    print '\n', "Mean accuracy using sklearn = ", sol.score(X,y), '\n'
    
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print "Loading data..."
        
        ### Self-defined load function, return a NumPy array ###
        #input_data = load(file_location, delimiter=',')
        
        ### Use genfromtxt() from numpy to load data ###
        input_data = np.genfromtxt(file_location, delimiter=',')
        
        print 'Solving: ', file_location
        solve_it(input_data)
    else:
        print 'This test requires an input file. Please select one from the directory. (i.e. python linear_regress.py *.txt)'

