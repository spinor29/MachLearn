#!/usr/bin/python

import numpy as np
from sklearn import linear_model

def solve_it(input_data):
    # Parse the input
    print 'Loading data...'
    lines = input_data.split('\n')
    lines.remove('')
    
    m = len(lines) # Number of data points in the training set
    X = []
    y = []
    for line in lines:
        fields = line.split(',')
        X.append([ float(fields[0]), float(fields[1]) ])
        y.append(float(fields[2]))

    X = np.array(X)
    y = np.array(y)
    
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
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print 'Solving: ', file_location
        solve_it(input_data)
    else:
        print 'This test requires an input file. Please select one from the directory. (i.e. python linear_regress.py *.txt)'

