# Author: Henry Moss, Ryan-Rhys Griffiths & Thomas O'Hagan
"""
Molecule graph kernels for Gaussian Process Regression implemented in GPflow using Graphkit-learn.
"""

import gpflow
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from pysmiles import read_smiles
import gklearn.kernels
import multiprocessing
import time

#def K_diag(self, X):
 #   """
  #  Compute the diagonal of the N x N kernel matrix of X
   # :param X: N x D array
    #:return: N x 1 array
    #"""
    #return tf.fill((tf.shape(X)[:-1]), tf.squeeze(self.variance))
    # see grakel source code for their k_diag function

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
        for i in range(len(G2)):
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
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))


class RW(gpflow.kernels.Kernel):
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
        for i in range(len(G2)):
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
        graph_kernel = gklearn.kernels.ShortestPath(node_labels=[], edge_labels=[], **kernel_options, )
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered',
                                                         n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class SSP(gpflow.kernels.Kernel):
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
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class T(gpflow.kernels.Kernel):
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
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

### Complete
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
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print("\nLoop", i)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

        return self.variance * kernel

    #def K_diag(self, X):
        #return tf.fill((tf.shape(X)), tf.squeeze(self.variance))
    def K_diag(self, X):
        kernel_options = {'directed': False, 'depth': 3, 'k_func': 'MinMax', 'compute_method': 'trie'}
        graph_kernel = gklearn.kernels.marginalizedkernel(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(X)):
            kernel_list, run_time = graph_kernel.compute(X[i], X[i], parallel='imap_unordered',
                                                         n_jobs=multiprocessing.cpu_count(), verbose=2)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        return self.variance * kernel

class WL(gpflow.kernels.Kernel):
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
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        print('\nBeginning kernel loop...')
        kernel_time = time.time()

        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        print('\nFinished kernel loop after', time.time() - kernel_time)

        return self.variance * cross_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))