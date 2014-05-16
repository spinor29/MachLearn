#!/usr/bin/python
# Logsitic regression

import numpy as np
from sklearn import linear_model
from pylab import *

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
    

def solve_it(input_data):
    
    ### Arrange data
    X = input_data[:,:2]
    y = input_data[:,2]
    
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
    
    ### Use sklearn's logistic regression module
    ### default: sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

    ### fit_intercept=True means adding a column of values 1 to X to calculate theta_0
    
    print 'Solving with Scikit-learn (sklearn)... '
    
    sol = linear_model.LogisticRegression(tol = 0.0001)
    sol.fit(X,y)
    
    print 'Theta values from sklearn: (Intercept, theta_1, theta2... )'
    ### intercept_ is theta_0, coef_ is [ theta_1, theta_2, ...., theta_n ]
    print ' ', sol.intercept_
    #print sol.coef_
    for coef in sol.coef_: print ' ', coef
    
    ### Compute accuracy on the training set
    print '\n', "Mean accuracy: "
    print sol.score(X,y), '\n'
    
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

