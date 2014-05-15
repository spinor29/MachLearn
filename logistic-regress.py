#!/usr/bin/python

import numpy
from sklearn import linear_model
from pylab import *

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
        y.append(int(fields[2]))

    X = numpy.array(X)
    y = numpy.array(y)
    
    
    # Plot data
    print "Plotting data..."
    pos = numpy.where(y==1)
    neg = numpy.where(y==0)
    
    plt.ion() # interactive mode for plot, used with raw_input() 
    #subplot(311)
    scatter(X[pos,0], X[pos,1], marker='o')
    scatter(X[neg,0], X[neg,1], marker='x')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    plt.show()
    raw_input('Press enter to continue...')
    
    # Use sklearn's linear regression module
    # default: linear_model.LinearRegression(fit_intercept=True, copy_X=True, normalize=False)
    # fit_intercept=True means adding a column of values 1 to X to calculate theta_0
    
    print
    print "Solving with Scikit-learn (sklearn)... "
    
    sol = linear_model.LogisticRegression(tol = 0.0001)
    sol.fit(X,y)
    
    print "Theta values from sklearn: "
    # intercept_ is theta_0, coef_ is [ theta_1, theta_2, ...., theta_n ]
    print ' ', sol.intercept_
    #print sol.coef_
    for coef in sol.coef_: print ' ', coef
    print
    
    # Compute accuracy on the training set
    print "Mean accuracy: "
    print sol.score(X,y)
    
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

