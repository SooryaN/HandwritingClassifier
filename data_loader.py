import cPickle
import numpy as np
import gzip

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    tr_inputs = [x.reshape(784, 1) for x in tr_d[0]]
    tr_results = [vectorized_result(n) for n in tr_d[1]]
    training_data = zip(tr_inputs, tr_results)
    validation_inputs = [x.reshape(784, 1) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [x.reshape(784, 1) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(n):
    v = np.zeros((10, 1))
    v[n] = 1
    return v
