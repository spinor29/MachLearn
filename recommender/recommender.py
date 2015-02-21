
import numpy as np
import scipy as sp
#from scipy import *
from sklearn import linear_model
from pylab import *
import time


def costFunction(params, *Z):
    [ X, Y, R, nm, nu, nf, lamb, J_or_G ] = Z[:8]

    X = params[:(nm*nf)].reshape(nm, nf)
    Theta = params[(nm*nf):].reshape(nu,nf)

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    Diff = X.dot(Theta.T) - Y
    J = 0.5 * np.sum(R * (Diff * Diff))

    X_grad = (R * Diff).dot(Theta)
    Theta_grad = (R * Diff).T.dot(X)

    #Add regularization
    reg1 = lamb / 2.0 * np.sum(Theta * Theta)
    reg2 = lamb / 2.0 * np.sum(X * X)

    J += reg1 + reg2

    X_grad += lamb * X
    Theta_grad += lamb * Theta

    grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))

    if J_or_G == 'J':
        return J
    elif J_or_G == 'G':
        return grad


def J_func(params, *Z):
    Y = Z + ('J',)
    J = costFunction(params, *Y)
    return J

def G_func(params, *Z):
    Y = Z + ('G',)
    grad = costFunction(params, *Y)
    return grad

def normalizeRatings(Y, R):
    m = len(Y)
    n = len(Y[0])
    Ymean = np.zeros(m)
    Ynorm = np.zeros((m,n))
    for  i in range(m):
        #for j in range(n):
        if long(np.sum(R[i,:])) > 0: 
            Ymean[i] = np.sum(Y[i,:]*R[i,:])/np.sum(R[i,:])
        else:
            Ymean[i] = 0.0

        for j in range(n):
            if int(R[i, j]) == 1:
                Ynorm[i,j] = Y[i,j] - Ymean[i]
    return Ynorm, Ymean

def loadMovieList():
    with open('movie_ids.txt', 'r') as f:
	    data = f.readlines()
    movies = []
    for line in data:
        movie = line.split(' ', 1)[1].rstrip()
        movies.append(movie)

    return movies


def loadMovieList2():
    with open('movies.dat', 'r') as f:
        data = f.readlines()
    movies = {}
    for line in data:
        fields = line.split('::')
        movies[long(fields[0])-1] = fields[1]

    n = 10
    print "\nThe first", n, "movies in data:"
    for i in range(n):
        print movies[i]
    print "......"

    return movies


def addNewRatings(Y,R,mlen):
    # mlen is the number of all movies
    #movieList = loadMovieList()
    my_ratings = np.zeros(mlen)
    my_R = np.zeros(mlen)

    with open('personalRatings.txt', 'r') as f:
        data = f.readlines()
    f.close()

    my_Y = np.zeros(mlen)
    my_R = np.zeros(mlen)
    for line in data:
        fields = line.split('::')
        iu, im, r, t = (long(fields[0]), long(fields[1])-1, float(fields[2]), long(fields[3]))
        my_Y[im], my_R[im] = r, 1

    Y = np.c_[ my_Y, Y ]
    R = np.c_[ my_R, R ]

    return my_Y, Y, R


def partition(input_data):
    lines = input_data.split('\n')
    lines.remove('')

    np.random.shuffle(lines)

    m1 = long(0.6*len(lines))
    m2 = long(0.8*len(lines))

    return lines[:m1], lines[m1:m2], lines[m2:]

def parseRatings(data, nm, nu):
    #lines = input_data.split('\n')
    #lines.remove('')

    Y = np.zeros((nm,nu))
    R = np.zeros((nm,nu))
    for line in data:
        fields = line.split('::')
        iu, im, r, t = (long(fields[0])-1, long(fields[1])-1, float(fields[2]), long(fields[3]))
        Y[im, iu] = r
        R[im, iu] = 1

    return Y, R

def solve_it(input_data, movieList):

    #Y = input_data['Y']
    #R = input_data['R']

    nu = 6040
    nm = 3952

    data_train, data_cross, data_test = partition(input_data)
    #print data_train[0]
    Y_train, R_train = parseRatings(data_train, nm, nu)
    Y_cross, R_cross = parseRatings(data_cross, nm, nu)
    Y_test, R_test = parseRatings(data_test, nm, nu)


    # Add ratings of a new user
    #my_ratings, Y, R = addNewRatings(Y, R, len(movieList))
    my_ratings, Y_train, R_train = addNewRatings(Y_train, R_train, nm)
    nu += 1
    Y_cross = np.c_[ np.zeros(nm), Y_cross ]
    R_cross = np.c_[ np.zeros(nm), R_cross ]
    Y_test = np.c_[ np.zeros(nm), Y_test ]
    R_test = np.c_[ np.zeros(nm), R_test ]

    print
    print "Add new ratings:"
    for i in range(len(my_ratings)):
        if int(my_ratings[i]) > 0:
            print my_ratings[i], movieList[i]

    # Normalize ratings
    Y = Y_train
    R = R_train

    nm = len(Y)
    nu = len(Y[0])
    nf = 10
    Ynorm, Ymean = normalizeRatings(Y, R)

    # Cross validation
    # Tune lamb, find the best value

    #lamb_values = [1.0, 5.0, 10.0, 20.0]
    lamb_values = [10.0]
    X_opt = []
    Theta_opt = []
    emin = 1e16
    lamb_opt = 1.0
    fw = open('crossValidate.log','w')

    start = time.time()
    for lamb in lamb_values:
        #lamb = 10.0
        X = np.random.randn(nm, nf)
        Theta = np.random.randn(nu, nf)
        initial_params = np.concatenate((X.flatten(), Theta.flatten()))
        Z = (X, Ynorm, R, nm, nu, nf, lamb)

        # Minimize the cost function
        params, J = sp.optimize.fmin_l_bfgs_b(J_func, x0=initial_params, fprime=G_func, args=Z, disp=1, maxiter=100)[:2]

        X = params[:nm*nf].reshape(nm, nf)
        Theta = params[nm*nf:].reshape(nu, nf)

        # Comparing predictions to cross validation set
        p = X.dot(Theta.T)

        for i in range(nu):
            p[:,i] += Ymean # predictions

        diff = (p - Y_cross) * R_cross
        err_cross = np.sqrt(np.sum(diff * diff)/np.sum(R_cross))

        diff = (p - Y_train) * R_train
        err_train = np.sqrt(np.sum(diff * diff)/np.sum(R_train))
        elog = ("lamba, err_train, err_cross = ", lamb, err_train, err_cross)
        print elog
        fw.write(str(elog))

        if err_cross < emin:
            emin, lamb_opt, X_opt, Theta_opt = err_cross, lamb, X, Theta
    
    fw.close()

    print "emin, lamb_opt = ", emin, lamb_opt

    print "Recommender system learning completed.\n"
    end = time.time()

    # Predictions
    X = X_opt
    Theta = Theta_opt

    p = X.dot(Theta.T)

    my_predictions = p[:,0] + Ymean

    id_sort = my_predictions.argsort()[::-1]

    print "Top recommendations for you:"

    ntop = 0
    for i in range(len(R_train)):
        if int(np.sum(R_train[id_sort[i],:])) > 5:
            print my_predictions[id_sort[i]], movieList[id_sort[i]]
            ntop += 1
        if ntop == 10: break

    print
    print "Your original ratings:"
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print my_ratings[i], movieList[i]

    print "Elapsed time:", end - start


import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print "Loading data..."

        input_data_file = open(file_location, 'r')
        ratingList = ''.join(input_data_file.readlines())
        input_data_file.close()

        # load matlab format file
        #ratingList = io.loadmat(file_location)
        #movieList = loadMovieList()

        movieList = loadMovieList2()

        print 'Solving: ', file_location
        solve_it(ratingList, movieList)
    else:
        print 'This test requires an input file. Please select one from the directory. (i.e. python recommender.py ratings.dat)'
