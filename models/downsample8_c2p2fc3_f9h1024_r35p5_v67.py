import numpy as np
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from lasagne.layers import dnn
from batch_norm import BatchNormLayer
import metrics
import time

rs = T.shared_randomstreams.RandomStreams()
rs.seed(int(time.time()))

data_path = 'eeg_train.npy'
train_series = [0, 1, 2, 3, 4, 5]
valid_series = [6, 7]
test_series = [0, 1, 2, 3, 4, 5]
events = [0, 1, 2, 3, 4, 5]
num_events = len(events)

train_data_params = {'section': 'train',
                     'chunk_gen_fun': 'random_chunk_gen_fun',
                     'channels': 32,
                     'length': 4096,
                     'preprocess': 'per_sample_mean',
                     'chunk_size': 4096,
                     'num_chunks': 400,
                     'pos_ratio': 0.35,
                     'bootstrap': True,
                     'neg_pool_size': 81920,
                     'hard_ratio': 1,
                     'easy_mode': 'all',
                     #'resize': [0.7, 1.3],
                     }

valid_data_params = {'section': 'valid',
                     'chunk_gen_fun': 'fixed_chunk_gen_fun',
                     'channels': 32,
                     'length': 4096,
                     'preprocess': 'per_sample_mean',
                     'chunk_size': 4096,
                     'pos_interval': 100,
                     'neg_interval': 100,
                     }

bs_data_params = {'section': 'bootstrap',
                  'chunk_gen_fun': 'fixed_chunk_gen_fun',
                  'channels': 32,
                  'length': 4096,
                  'preprocess': 'per_sample_mean',
                  'chunk_size': 4096,
                  'pos_interval': 100,
                  'neg_interval': 100,
                  }

test_valid_params = {'section': 'valid',
                    'chunk_gen_fun': 'test_valid_chunk_gen_fun',
                    'channels': 32,
                    'length': 4096,
                    'preprocess': 'per_sample_mean',
                    'chunk_size': 4096,
                    'test_lens': [4096],
                    'interval': 10,
                    }

test_data_params = {'section': 'test',
                    'chunk_gen_fun': 'sequence_chunk_gen_fun',
                    'channels': 32,
                    'length': 4096,
                    'preprocess': 'per_sample_mean',
                    'chunk_size': 4096,
                    'test_lens': [4096],
                    'test_valid': True,
                    }


batch_size = 64
momentum = 0.9
wc = 0.001
display_freq = 10
valid_freq = 20
bs_freq = 20000
save_freq = 20

def lr_schedule(chunk_idx):
    base = 0.1
    if chunk_idx < 200:
        return base
    elif chunk_idx < 320:
        return 0.1 * base
    elif chunk_idx < 390:
        return 0.01 * base
    else:
        return 0.001 * base


std = 0.02
p = 0.5

metrics = [metrics.meanAUC]
metric_names = ['areas under the ROC curve']

Conv2DLayer = dnn.Conv2DDNNLayer
Pool2DLayer = dnn.Pool2DDNNLayer

input_dims = (batch_size,
              train_data_params['channels'],
              1,
              train_data_params['length'])
def build_model():
    l_in = nn.layers.InputLayer(input_dims)

    pool0 = Pool2DLayer(incoming = l_in, pool_size = (1, 8), stride = (1, 8), mode = 'average')
    print 'pool0', nn.layers.get_output_shape(pool0)

    conv1 = Conv2DLayer(incoming = l_in, num_filters = 8, filter_size = (1, 9),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv1', nn.layers.get_output_shape(conv1)

    bn1 = BatchNormLayer(incoming = conv1, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn1', nn.layers.get_output_shape(bn1)

    pool1 = Pool2DLayer(incoming = bn1, pool_size = (1, 2), stride = (1, 2))
    print 'pool1', nn.layers.get_output_shape(pool1)

    drop1 = nn.layers.DropoutLayer(incoming = pool1, p = p)
    print 'drop1', nn.layers.get_output_shape(drop1)

    conv2 = Conv2DLayer(incoming = drop1, num_filters = 16, filter_size = (1, 9),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv2', nn.layers.get_output_shape(conv2)

    bn2 = BatchNormLayer(incoming = conv2, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn2', nn.layers.get_output_shape(bn2)

    pool2 = Pool2DLayer(incoming = bn2, pool_size = (1, 2), stride = (1, 2))
    print 'pool2', nn.layers.get_output_shape(pool2)

    drop2 = nn.layers.DropoutLayer(incoming = pool2, p = p)
    print 'drop2', nn.layers.get_output_shape(drop2)

    fc3 = nn.layers.DenseLayer(incoming = drop2, num_units = 1024,
                                 W = nn.init.Normal(std = std),
                                 nonlinearity = None)

    bn3 = BatchNormLayer(incoming = fc3, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn3', nn.layers.get_output_shape(bn3)

    drop3 = nn.layers.DropoutLayer(incoming = bn3, p = p)

    fc4 = nn.layers.DenseLayer(incoming = drop3, num_units = 1024,
                                 W = nn.init.Normal(std = std),
                                 nonlinearity = None)

    bn4 = BatchNormLayer(incoming = fc4, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn4', nn.layers.get_output_shape(bn4)

    drop4 = nn.layers.DropoutLayer(incoming = bn4, p = p)

    fc5 = nn.layers.DenseLayer(incoming = drop4, num_units = 1024,
                                 W = nn.init.Normal(std = std),
                                 nonlinearity = None)

    bn5 = BatchNormLayer(incoming = fc5, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn5', nn.layers.get_output_shape(bn5)

    drop5 = nn.layers.DropoutLayer(incoming = bn5, p = p)

    l_out = nn.layers.DenseLayer(incoming = drop5, num_units = num_events,
                                 W = nn.init.Normal(std = std),
                                 nonlinearity = nn.nonlinearities.sigmoid)
    print 'l_out', nn.layers.get_output_shape(l_out)

    return l_out

def build_train_valid(l_out):
    params = nn.layers.get_all_params(l_out, regularizable = True)
    wc_term = 0.5 * sum(T.sum(param ** 2) for param in params)
    
    x_batch = T.tensor4('x', theano.config.floatX)
    y_batch = T.matrix('y', 'int32')
    train_output = nn.layers.get_output(l_out, x_batch)
    train_loss = nn.objectives.binary_crossentropy(train_output, y_batch)
    train_loss = nn.objectives.aggregate(train_loss, mode = 'mean')
    train_loss += wc * wc_term
    params = nn.layers.get_all_params(l_out, trainable = True)

    valid_output = nn.layers.get_output(l_out, x_batch, deterministic = True)

    lr = theano.shared(np.float32(lr_schedule(0)))
    updates = nn.updates.momentum(train_loss, params, lr, momentum)

    x_shared = nn.utils.shared_empty(dim = len(input_dims))
    y_shared = nn.utils.shared_empty(dim = 2, dtype = 'int32')
    idx = T.scalar('idx', 'int32')
    givens = {x_batch: x_shared[idx * batch_size:(idx + 1) * batch_size],
              y_batch: y_shared[idx * batch_size:(idx + 1) * batch_size]}

    iter_train = theano.function([idx], [train_loss, train_output],
                                 givens = givens,
                                 updates = updates)
    
    givens = {x_batch: x_shared[idx * batch_size:(idx + 1) * batch_size]}
    iter_valid = theano.function([idx], valid_output, givens = givens)
    
    return x_shared, y_shared, idx, lr, iter_train, iter_valid
