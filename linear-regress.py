#!/usr/bin/python

import numpy as np
from sklearn import linear_model

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
    
    # Use sklearn's linear regression module
    # default: linear_model.LinearRegression(fit_intercept=True, copy_X=True, normalize=False)
    # fit_intercept=True means adding a column of values 1 to X to calculate theta_0
    
    print
    print "Solving with Scikit-learn (sklearn)... "
    
    sol = linear_model.LinearRegression(normalize=True)
    sol.fit(X, y) # X is array containing m training examples and n features
    
    print "Theta values from sklearn: "
    # intercept_ is theta_0, coef_ is [ theta_1, theta_2, ...., theta_n ]
    print ' ', sol.intercept_
    for coef in sol.coef_: print ' ', coef
    print

    # Predict the price of an example
    x = [1650, 3]
    price = sol.predict(x)
  
    print 'Predicted price of a 1650 sq-ft, 3 br house: '
    print ' $'+str(price)
    
    #print X
    #print y
    
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

