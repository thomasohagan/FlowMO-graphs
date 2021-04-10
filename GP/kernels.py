# Author: Henry Moss, Ryan-Rhys Griffiths & Thomas O'Hagan
"""
Molecule graph kernels for Gaussian Process Regression implemented in GPflow using Graphkit-learn.
"""

import gpflow
from gpflow.utilities import positive
import tensorflow as tf
from pysmiles import read_smiles
import gklearn.kernels
import multiprocessing

#def K_diag(self, X):
 #   """
  #  Compute the diagonal of the N x N kernel matrix of X
   # :param X: N x D array
    #:return: N x 1 array
    #"""
    #return tf.fill((tf.shape(X)[:-1]), tf.squeeze(self.variance))
    # see grakel source code for their k_diag function

class PUTH(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        G1 = []
        if str(type(X[1])) == "<class 'numpy.ndarray'>":
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))
        else:
            X = X.numpy()
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))

        if X2 is None:
            G2 = G1

        else:
            G2 = []
            if str(type(X2[1])) == "<class 'numpy.ndarray'>":
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append(read_smiles(h))

            elif str(type(X2[1])) == "<class \'numpy.str_\'>":
                for string2 in X2:
                    G2.append(read_smiles(string2))

            else:
                X2 = X2.numpy()
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append((read_smiles(h)))

        kernel_options = {'directed': False, 'depth': 3, 'k_func': 'MinMax', 'compute_method': 'trie'}
        graph_kernel = gklearn.kernels.PathUpToH(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(G1)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class SP(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        G1 = []
        if str(type(X[1])) == "<class 'numpy.ndarray'>":
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))
        else:
            X = X.numpy()
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))

        if X2 is None:
            G2 = G1

        else:
            G2 = []
            if str(type(X2[1])) == "<class 'numpy.ndarray'>":
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append(read_smiles(h))

            elif str(type(X2[1])) == "<class \'numpy.str_\'>":
                for string2 in X2:
                    G2.append(read_smiles(string2))

            else:
                X2 = X2.numpy()
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append((read_smiles(h)))

        kernel_options = {'directed': False, 'depth': 3, 'k_func': 'MinMax', 'compute_method': 'trie'}
        graph_kernel = gklearn.kernels.ShortestPath(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(G1)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))


class CW(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        G1 = []
        if str(type(X[1])) == "<class 'numpy.ndarray'>":
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))
        else:
            X = X.numpy()
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))

        if X2 is None:
            G2 = G1

        else:
            G2 = []
            if str(type(X2[1])) == "<class 'numpy.ndarray'>":
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append(read_smiles(h))

            elif str(type(X2[1])) == "<class \'numpy.str_\'>":
                for string2 in X2:
                    G2.append(read_smiles(string2))

            else:
                X2 = X2.numpy()
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append((read_smiles(h)))

        kernel_options = {'directed': False, 'depth': 3, 'k_func': 'MinMax', 'compute_method': 'trie'}
        graph_kernel = gklearn.kernels.commonwalkkernel(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(G1)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class MK(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        G1 = []
        if str(type(X[1])) == "<class 'numpy.ndarray'>":
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))
        else:
            X = X.numpy()
            for string1 in X:
                h = string1.decode("utf-8")
                G1.append(read_smiles(h))

        if X2 is None:
            G2 = G1

        else:
            G2 = []
            if str(type(X2[1])) == "<class 'numpy.ndarray'>":
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append(read_smiles(h))

            elif str(type(X2[1])) == "<class \'numpy.str_\'>":
                for string2 in X2:
                    G2.append(read_smiles(string2))

            else:
                X2 = X2.numpy()
                for string2 in X2:
                    h = string2.decode("utf-8")
                    G2.append((read_smiles(h)))

        kernel_options = {'directed': False, 'depth': 3, 'k_func': 'MinMax', 'compute_method': 'trie'}
        graph_kernel = gklearn.kernels.marginalizedkernel(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(G1)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))