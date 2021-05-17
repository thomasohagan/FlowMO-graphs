# Author: Henry Moss, Ryan-Rhys Griffiths & Thomas O'Hagan
"""
Molecule graph kernels for Gaussian Process Regression implemented in GPflow using Graphkit-learn.
"""

import gpflow
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf
from pysmiles import read_smiles
import gklearn.kernels
import multiprocessing
import time
#from gklearn.dataset
#import os

#def K_diag(self, X):
 #   """
  #  Compute the diagonal of the N x N kernel matrix of X
   # :param X: N x D array
    #:return: N x 1 array
    #"""
    #return tf.fill((tf.shape(X)[:-1]), tf.squeeze(self.variance))
    # see grakel source code for their k_diag function

class CWgeo(gpflow.kernels.Kernel):
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


        graph_kernel = gklearn.kernels.CommonWalk(node_labels=[], edge_labels=[], weight=0.01, compute_method='geo', ds_infos={'directed': False, 'name' : 'MUTAG'})
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=True)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))


class CWexp(gpflow.kernels.Kernel):
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

        kernel_options = {'compute_method': 'exp'}
        graph_kernel = gklearn.kernels.CommonWalk(node_labels=[], edge_labels=[], ds_infos={'directed': False, 'name' : 'MUTAG'}, **kernel_options,)
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

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

        kernel_options = {'directed': False, 'remove_totters' : False}
        graph_kernel = gklearn.kernels.Marginalized(node_labels=[], edge_labels=[], p_quit=0.5, n_iteration=20, **kernel_options,)
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))


class RW(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        import functools
        from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct

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

        #mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        #sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
        #graph_kernel = gklearn.kernels.RandomWalk(node_labels=[],
#        						node_attrs=[],
#       						edge_labels=[],
#       						edge_attrs=[],
#        						directed=False,
#        						compute_method='sylvester',
#        						weight=1e-3,
#        						p=None,
#       						q=None,
#       						edge_weight=None,
#       						node_kernels=sub_kernels,
#        						edge_kernels=sub_kernels,
#        						sub_kernel='exp')

        gklearn.kernels.randomWalkKernel()

        kernel = []
        kernel_list = []
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        sub_kernels = [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]
        for i in range(len(G1)):
            kernel_j = []
            for j in range(len(G2)):
                kernel_ij, run_time = gklearn.kernels.randomwalkkernel(G1[i], G2[j], n_jobs=multiprocessing.cpu_count(), verbose=True, compute_method=compute_method,
                                                  weight=1e-3,
                                                  p=None,
                                                  q=None,
                                                  edge_weight=None,
                                                  node_kernels=sub_kernels,
                                                  edge_kernels=sub_kernels,
                                                  node_label=[],
                                                  edge_label=[],
                                                  sub_kernel='exp',)
                kernel_j.append(kernel_ij)
            kernel.append(kernel_j)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class SP(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
        import functools


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

        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

        graph_kernel = gklearn.kernels.ShortestPath(node_labels=[],
                                    node_attrs=[],
                                    directed=False,
                                    fcsp=None,
                                    node_kernels=sub_kernels)
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered',
                                                         n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class SSP(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
        import functools

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

        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}

        graph_kernel = gklearn.kernels.StructuralSP(node_labels=[],
						 edge_labels=[],
						 node_attrs=[],
						 edge_attrs=[],
						 ds_infos={'directed': False, 'name' : 'MUTAG'},
						 fcsp=None,
						 node_kernels=sub_kernels,
						 edge_kernels=sub_kernels)

        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

        return self.variance * kernel

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))

class T(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):

        from gklearn.utils.kernels import polynomialkernel
        import functools

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

        pkernel = functools.partial(polynomialkernel, d=2, c=1e5)
        graph_kernel = gklearn.kernels.Treelet(node_labels=[], edge_labels=[], sub_kernel=pkernel, ds_infos={'directed': False})
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            print(kernel_list)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        kernel = tf.transpose(kernel)

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

    def K_diag(self, X):
        return tf.fill((tf.shape(X)), tf.squeeze(self.variance))


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

        kernel_options = {'directed': False, 'height': 2}
        graph_kernel = gklearn.kernels.WeisfeilerLehman(node_labels=[], edge_labels=[], **kernel_options,)
        kernel = []
        for i in range(len(G2)):
            kernel_list, run_time = graph_kernel.compute(G1, G2[i], parallel='imap_unordered', n_jobs=multiprocessing.cpu_count(), verbose=2)
            kernel.append(kernel_list)

        kernel = tf.convert_to_tensor(kernel, dtype=tf.float64)
        #kernel = tf.transpose(kernel)

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