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
train_series = [0, 1, 2, 3, 4, 5, 6, 7]
valid_series = []
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
                    'test_valid': False,
                    }


batch_size = 64
momentum = 0.9
wc = 0.001
display_freq = 10
valid_freq = 20000
bs_freq = 20
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
p1 = 0
p2 = 0.2
p3 = 0.2
p4 = 0.2

metrics = [metrics.meanAccuracy, metrics.meanAUC]
metric_names = ['mean accuracy', 'areas under the ROC curve']

Conv2DLayer = dnn.Conv2DDNNLayer
Pool2DLayer = dnn.Pool2DDNNLayer
SumLayer = nn.layers.ElemwiseSumLayer

input_dims = (batch_size,
              train_data_params['channels'],
              1,
              train_data_params['length'])
def build_model():
    l_in = nn.layers.InputLayer(input_dims)

    pool0 = Pool2DLayer(incoming = l_in, pool_size = (1, 4), stride = (1, 4), mode = 'average')
    print 'pool0', nn.layers.get_output_shape(pool0)


    conv1 = Conv2DLayer(incoming = pool0, num_filters = 128, filter_size = (1, 9),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv1', nn.layers.get_output_shape(conv1)

    bn1 = BatchNormLayer(incoming = conv1, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.very_leaky_rectify)
    print 'bn1', nn.layers.get_output_shape(bn1)

    pool1 = Pool2DLayer(incoming = bn1, pool_size = (1, 4), stride = (1, 4))
    print 'pool1', nn.layers.get_output_shape(pool1)

    drop1 = nn.layers.DropoutLayer(incoming = pool1, p = p1)
    print 'drop1', nn.layers.get_output_shape(drop1)

    conv2 = Conv2DLayer(incoming = drop1, num_filters = 128, filter_size = (1, 1),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv2', nn.layers.get_output_shape(conv2)

    bn2 = BatchNormLayer(incoming = conv2, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.very_leaky_rectify)
    print 'bn2', nn.layers.get_output_shape(bn2)

    conv2a = Conv2DLayer(incoming = bn2, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = nn.init.Normal(std = std), b = None,
                         nonlinearity = None)
    print 'conv2a', nn.layers.get_output_shape(conv2a)

    sum2a = SumLayer(incomings = [conv2, conv2a], coeffs = 1)
    print 'sum2a', nn.layers.get_output_shape(sum2a)
    
    bn2a = BatchNormLayer(incoming = sum2a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn2a', nn.layers.get_output_shape(bn2a)

    conv2b = Conv2DLayer(incoming = bn2a, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv2a.W, b = None,
                         nonlinearity = None)
    print 'conv2b', nn.layers.get_output_shape(conv2b)

    sum2b = SumLayer(incomings = [conv2, conv2b], coeffs = 1)
    print 'sum2b', nn.layers.get_output_shape(sum2b)
    
    bn2b = BatchNormLayer(incoming = sum2b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn2b', nn.layers.get_output_shape(bn2b)

    conv2c = Conv2DLayer(incoming = bn2b, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv2a.W, b = None,
                         nonlinearity = None)
    print 'conv2c', nn.layers.get_output_shape(conv2c)

    sum2c = SumLayer(incomings = [conv2, conv2c], coeffs = 1)
    print 'sum2c', nn.layers.get_output_shape(sum2c)
    
    bn2c = BatchNormLayer(incoming = sum2c, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn2c', nn.layers.get_output_shape(bn2c)

    pool2 = Pool2DLayer(incoming = bn2c, pool_size = (1, 4), stride = (1, 4))
    print 'pool2', nn.layers.get_output_shape(pool2)

    drop2 = nn.layers.DropoutLayer(incoming = pool2, p = p2)
    print 'drop2', nn.layers.get_output_shape(drop2)

    conv3 = Conv2DLayer(incoming = drop2, num_filters = 128, filter_size = (1, 1),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv3', nn.layers.get_output_shape(conv3)

    bn3 = BatchNormLayer(incoming = conv3, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.very_leaky_rectify)
    print 'bn3', nn.layers.get_output_shape(bn3)

    conv3a = Conv2DLayer(incoming = bn3, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = nn.init.Normal(std = std), b = None,
                         nonlinearity = None)
    print 'conv3a', nn.layers.get_output_shape(conv3a)

    sum3a = SumLayer(incomings = [conv3, conv3a], coeffs = 1)
    print 'sum3a', nn.layers.get_output_shape(sum3a)
    
    bn3a = BatchNormLayer(incoming = sum3a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn3a', nn.layers.get_output_shape(bn3a)

    conv3b = Conv2DLayer(incoming = bn3a, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv3a.W, b = None,
                         nonlinearity = None)
    print 'conv3b', nn.layers.get_output_shape(conv3b)

    sum3b = SumLayer(incomings = [conv3, conv3b], coeffs = 1)
    print 'sum3b', nn.layers.get_output_shape(sum3b)
    
    bn3b = BatchNormLayer(incoming = sum3b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn3b', nn.layers.get_output_shape(bn3b)

    conv3c = Conv2DLayer(incoming = bn3b, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv3a.W, b = None,
                         nonlinearity = None)
    print 'conv3c', nn.layers.get_output_shape(conv3c)

    sum3c = SumLayer(incomings = [conv3, conv3c], coeffs = 1)
    print 'sum3c', nn.layers.get_output_shape(sum3c)
    
    bn3c = BatchNormLayer(incoming = sum3c, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn3c', nn.layers.get_output_shape(bn3c)

    pool3 = Pool2DLayer(incoming = bn3c, pool_size = (1, 2), stride = (1, 2))
    print 'pool3', nn.layers.get_output_shape(pool3)

    drop3 = nn.layers.DropoutLayer(incoming = pool3, p = p3)
    print 'drop3', nn.layers.get_output_shape(drop3)

    conv4 = Conv2DLayer(incoming = drop3, num_filters = 128, filter_size = (1, 1),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv4', nn.layers.get_output_shape(conv4)

    bn4 = BatchNormLayer(incoming = conv4, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.very_leaky_rectify)
    print 'bn4', nn.layers.get_output_shape(bn4)

    conv4a = Conv2DLayer(incoming = bn4, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = nn.init.Normal(std = std), b = None,
                         nonlinearity = None)
    print 'conv4a', nn.layers.get_output_shape(conv4a)

    sum4a = SumLayer(incomings = [conv4, conv4a], coeffs = 1)
    print 'sum4a', nn.layers.get_output_shape(sum4a)
    
    bn4a = BatchNormLayer(incoming = sum4a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn4a', nn.layers.get_output_shape(bn4a)

    conv4b = Conv2DLayer(incoming = bn4a, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv4a.W, b = None,
                         nonlinearity = None)
    print 'conv4b', nn.layers.get_output_shape(conv4b)

    sum4b = SumLayer(incomings = [conv4, conv4b], coeffs = 1)
    print 'sum4b', nn.layers.get_output_shape(sum4b)
    
    bn4b = BatchNormLayer(incoming = sum4b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn4b', nn.layers.get_output_shape(bn4b)

    conv4c = Conv2DLayer(incoming = bn4b, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv4a.W, b = None,
                         nonlinearity = None)
    print 'conv4c', nn.layers.get_output_shape(conv4c)

    sum4c = SumLayer(incomings = [conv4, conv4c], coeffs = 1)
    print 'sum4c', nn.layers.get_output_shape(sum4c)
    
    bn4c = BatchNormLayer(incoming = sum4c, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn4c', nn.layers.get_output_shape(bn4c)

    pool4 = Pool2DLayer(incoming = bn4c, pool_size = (1, 2), stride = (1, 2))
    print 'pool4', nn.layers.get_output_shape(pool4)

    drop4 = nn.layers.DropoutLayer(incoming = pool4, p = p4)
    print 'drop4', nn.layers.get_output_shape(drop4)

    conv5 = Conv2DLayer(incoming = drop4, num_filters = 128, filter_size = (1, 1),
                        stride = 1, border_mode = 'same',
                        W = nn.init.Normal(std = std),
                        nonlinearity = None)
    print 'conv5', nn.layers.get_output_shape(conv5)

    bn5 = BatchNormLayer(incoming = conv5, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.very_leaky_rectify)
    print 'bn5', nn.layers.get_output_shape(bn5)

    conv5a = Conv2DLayer(incoming = bn5, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = nn.init.Normal(std = std), b = None,
                         nonlinearity = None)
    print 'conv5a', nn.layers.get_output_shape(conv5a)

    sum5a = SumLayer(incomings = [conv5, conv5a], coeffs = 1)
    print 'sum5a', nn.layers.get_output_shape(sum5a)
    
    bn5a = BatchNormLayer(incoming = sum5a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn5a', nn.layers.get_output_shape(bn5a)

    conv5b = Conv2DLayer(incoming = bn5a, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv5a.W, b = None,
                         nonlinearity = None)
    print 'conv5b', nn.layers.get_output_shape(conv5b)

    sum5b = SumLayer(incomings = [conv5, conv5b], coeffs = 1)
    print 'sum5b', nn.layers.get_output_shape(sum5b)
    
    bn5b = BatchNormLayer(incoming = sum5b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn5b', nn.layers.get_output_shape(bn5b)

    conv5c = Conv2DLayer(incoming = bn5b, num_filters = 128, filter_size = (1, 9),
                         stride = 1, border_mode = 'same',
                         W = conv5a.W, b = None,
                         nonlinearity = None)
    print 'conv5c', nn.layers.get_output_shape(conv5c)

    sum5c = SumLayer(incomings = [conv5, conv5c], coeffs = 1)
    print 'sum5c', nn.layers.get_output_shape(sum5c)
    
    bn5c = BatchNormLayer(incoming = sum5c, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.very_leaky_rectify)    
    print 'bn5c', nn.layers.get_output_shape(bn5c)

    pool5 = Pool2DLayer(incoming = bn5c, pool_size = (1, 4), stride = (1, 4))
    print 'pool5', nn.layers.get_output_shape(pool5)

    l_out = nn.layers.DenseLayer(incoming = pool5, num_units = num_events,
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
    updates = nn.updates.nesterov_momentum(train_loss, params, lr, momentum)

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
