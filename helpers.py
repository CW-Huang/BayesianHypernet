#!/usr/bin/env python
    
import theano.tensor as T
import numpy
np = numpy

# from Hendrycks
def gelu_fast(x):
    return 0.5 * x * (1 + T.tanh(T.sqrt(2 / np.pi) * (x + 0.044715 * T.pow(x, 3))))
gelu = gelu_fast


  
######################
class SaveLoadMIXIN(object):
    """
    These could use set/get _all_param_values, if we're willing to use self.layer instead of self.params...
    (just based on https://github.com/Lasagne/Lasagne/blob/06e4ad666873bf9e5a0f914386a7f0bd80bb341a/lasagne/layers/helper.py)
    """
    def save(self, save_path):
        np.save(save_path, [p.get_value() for p in self.params])

    def load(self, save_path):
        # LOAD lasagne.layers.set_all_param_values([h_layer, layer], np.load(save_path + '_params_best.npy'))
        values = np.load(save_path)

        if len(self.params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                             (len(values), len(self.params)))

        for p, v in zip(self.params, values):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    # instead of saving/loading to disk, it may be faster to keep the reset params as attributes
    def add_reset(self, name):
        """
        store current params in self.reset_dict using name as a key
        """
        if not 'reset_dict' in self.__dict__.keys():
            self.reset_dict = {}
        current_params = [p.get_value() for p in self.params]#lasagne.layers.get_all_param_values(self.layer)
        updates = {p:p0 for p, p0 in zip(self.params,current_params)}
        reset_fn = theano.function([],None, updates=updates)
        # 
        self.reset_dict[name] = reset_fn

    def call_reset(self, name):
        self.reset_dict[name]()
         
  
######################

def flatten_list(plist):
    return T.concatenate([p.flatten() for p in plist])


def plot_dict(dd):
    from pylab import *
    figure()
    for kk, vv in dd.items():
        plot(vv, label=kk)
    legend()

######################

def get_mushrooms():
    from mushroom_data import X,Y
    from lasagne.objectives import squared_error
    return X, Y, None, squared_error

def get_mnist():
    pass

def get_task(task_name):
    """
    returns:
        X, Y, output_function, loss_function, {other}
    """
    pass


######################
# load_cifar10
# code repurposed from the tf-learn library
import sys
import os
import pickle
import numpy as np
from six.moves import urllib
import tarfile

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

# load training and testing data
def load_data10(randomize=True, return_val=False, one_hot=False, dirname="cifar-10-batches-py", mnistify=False):

    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            #d = pickle.load(f, encoding='latin1')
            d = pickle.load(f)
        data = d["data"]
        labels = d["labels"]
        return data, labels


    def maybe_download(filename, source_url, work_directory):
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            print("Downloading CIFAR 10...")
            filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                     filepath)
            statinfo = os.stat(filepath)
            print(('CIFAR 10 downloaded', filename, statinfo.st_size, 'bytes.'))
            untar(filepath)
        return filepath


    def untar(fname):
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname)
            tar.extractall()
            tar.close()
            print("File Extracted in Current Directory")
        else:
            print("Not a tar.gz file: '%s '" % sys.argv[0])

    tarpath = maybe_download("cifar-10-python.tar.gz",
                             "http://www.cs.toronto.edu/~kriz/", dirname)
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    X_test, Y_test = load_batch(os.path.join(dirname, 'test_batch'))

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if randomize is True:
        test_perm = np.array(np.random.permutation(X_test.shape[0]))
        X_test = X_test[test_perm]
        Y_test = np.asarray(Y_test)
        Y_test = Y_test[test_perm]

        perm = np.array(np.random.permutation(X_train.shape[0]))
        X_train = X_train[perm]
        Y_train = np.asarray(Y_train)
        Y_train = Y_train[perm]
    if return_val:
        X_train, X_val = np.split(X_train, [45000])     # 45000 for training, 5000 for validation
        Y_train, Y_val = np.split(Y_train, [45000])

        if one_hot:
            Y_train, Y_val, Y_test = to_categorical(Y_train, 10), to_categorical(Y_val, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        else:
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        if one_hot:
            Y_train, Y_test = to_categorical(Y_train, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_test, Y_test
        else:
            return X_train, Y_train, X_test, Y_test

