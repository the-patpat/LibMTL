import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from cvxopt import matrix, spmatrix, solvers as cvxsolvers, coneprog
from smcp.solvers import conelp

coneprog.options['show_progress'] = False
import time
def generate_vectors_nd(n=3):
    # Generate two random vectors in R3
    vector1 = np.random.rand(n)
    vector2 = np.random.rand(n)

    # Ensure the angle between the vectors is less than 90 degrees
    while np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))) >= np.pi / 2:
        vector2 = np.random.rand(n)

    return vector1, vector2

n = 3#2359296
 
for n in range(100, 10001, 500):
# for n in range(10000, 20000, 2000):

    vector1, vector2 = generate_vectors_nd(n)
    c = matrix(vector2)

    elem_v1_i = [0]*n
    elem_v1_j = list(range(n))
    elem_v2_i = [1]*n
    elem_v2_j = list(range(n))
    elem_zero_i = [2]*n
    elem_zero_j = list(range(n))
    elem_I_i = list(range(3, n+3))
    elem_I_j = list(range(n))

    inds_i = elem_v1_i + elem_v2_i + elem_zero_i + elem_I_i
    inds_j = elem_v1_j + elem_v2_j + elem_zero_j + elem_I_j

    G = spmatrix(np.concatenate([-vector1, vector2, np.zeros(n), np.ones(n)]), inds_i, inds_j)
    h = matrix([0, 0.0, np.linalg.norm(vector1), matrix(np.zeros(n))])

    dims = {'l' : 2, 'q':[n+1], 's': []}

    t1 = time.time()
    # sol = conelp(c, G, h, dims)
    t1 = time.time() - t1
    G = matrix([matrix(-vector1).T, matrix(vector2).T, matrix(np.zeros(n)).T, matrix(np.eye(n)).T])
    h = matrix([0, 0.0, np.linalg.norm(vector1), matrix(np.zeros(n))])
    t2 = time.time()
    sol = cvxsolvers.conelp(c,G,h, dims)
    t2 = time.time() - t2
    print(f'{n}\t{t1:.2f}\t{t2:.2f}')