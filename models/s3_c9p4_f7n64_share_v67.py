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
p1 = 0.1
p2 = 0.1
p3 = 0.1
p4 = 0.1

metrics = [metrics.meanAUC]
metric_names = ['areas under the ROC curve']

Conv2DLayer = dnn.Conv2DDNNLayer
Pool2DLayer = dnn.Pool2DDNNLayer

input_dims = (batch_size,
              train_data_params['channels'],
              1,
              train_data_params['length'])

length_s = train_data_params['length'] / 4
l_in1 = nn.layers.InputLayer((batch_size,
                              train_data_params['channels'],
                              1,
                              length_s))
l_in2 = nn.layers.InputLayer((batch_size,
                              train_data_params['channels'],
                              1,
                              length_s * 2))
l_in3 = nn.layers.InputLayer((batch_size,
                              train_data_params['channels'],
                              1,
                              length_s * 4))
def build_model():
    l_in2_ = Pool2DLayer(incoming = l_in2, pool_size = (1, 2), stride = (1, 2))
    l_in3_ = Pool2DLayer(incoming = l_in3, pool_size = (1, 4), stride = (1, 4))
    
    s1, param = build_single_scale(l_in1)
    s2 = build_single_scale(l_in2_, param)
    s3 = build_single_scale(l_in3_, param)

    concat = nn.layers.ConcatLayer(incomings = [s1, s2, s3], axis = 1)
    l_out = nn.layers.DenseLayer(incoming = concat, num_units = num_events,
                                 W = nn.init.Normal(std = std),
                                 nonlinearity = nn.nonlinearities.sigmoid)
    print 'l_out', nn.layers.get_output_shape(l_out)

    return l_out

def build_single_scale(l_in, param=None):
    print

    if param is None:
        param = {}

    conv1 = Conv2DLayer(incoming = l_in, num_filters = 64, filter_size = (1, 7),
                        stride = 1, border_mode = 'same',
                        W = param['conv1_w'] if param.has_key('conv1_w') else nn.init.Normal(std = std),
                        b = param['conv1_b'] if param.has_key('conv1_b') else nn.init.Constant(0.),
                        nonlinearity = None)
    print 'conv1', nn.layers.get_output_shape(conv1)

    bn1 = BatchNormLayer(incoming = conv1, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn1', nn.layers.get_output_shape(bn1)

    pool1 = Pool2DLayer(incoming = bn1, pool_size = (1, 4), stride = (1, 4))
    print 'pool1', nn.layers.get_output_shape(pool1)

    drop1 = nn.layers.DropoutLayer(incoming = pool1, p = p1)
    print 'drop1', nn.layers.get_output_shape(drop1)

    conv2 = Conv2DLayer(incoming = drop1, num_filters = 64, filter_size = (1, 7),
                        stride = 1, border_mode = 'same',
                        W = param['conv2_w'] if param.has_key('conv2_w') else nn.init.Normal(std = std),
                        b = param['conv2_b'] if param.has_key('conv2_b') else nn.init.Constant(0.),
                        nonlinearity = None)
    print 'conv2', nn.layers.get_output_shape(conv2)

    bn2 = BatchNormLayer(incoming = conv2, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn2', nn.layers.get_output_shape(bn2)

    conv2a = Conv2DLayer(incoming = bn2, num_filters = 64, filter_size = (1, 7),
                         stride = 1, border_mode = 'same',
                         W = param['conv2a_w'] if param.has_key('conv2a_w') else nn.init.Normal(std = std),
                         b = param['conv2a_b'] if param.has_key('conv2a_b') else nn.init.Constant(0.),
                         nonlinearity = None)
    print 'conv2a', nn.layers.get_output_shape(conv2a)

    bn2a = BatchNormLayer(incoming = conv2a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn2a', nn.layers.get_output_shape(bn2a)

    pool2 = Pool2DLayer(incoming = bn2a, pool_size = (1, 4), stride = (1, 4))
    print 'pool2', nn.layers.get_output_shape(pool2)

    drop2 = nn.layers.DropoutLayer(incoming = pool2, p = p2)
    print 'drop2', nn.layers.get_output_shape(drop2)

    conv3 = Conv2DLayer(incoming = drop2, num_filters = 64, filter_size = (1, 7),
                        stride = 1, border_mode = 'same',
                        W = param['conv3_w'] if param.has_key('conv3_w') else nn.init.Normal(std = std),
                        b = param['conv3_b'] if param.has_key('conv3_b') else nn.init.Constant(0.),
                        nonlinearity = None)
    print 'conv3', nn.layers.get_output_shape(conv3)

    bn3 = BatchNormLayer(incoming = conv3, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn3', nn.layers.get_output_shape(bn3)

    conv3a = Conv2DLayer(incoming = bn3, num_filters = 64, filter_size = (1, 7),
                         stride = 1, border_mode = 'same',
                         W = param['conv3a_w'] if param.has_key('conv3a_w') else nn.init.Normal(std = std),
                         b = param['conv3a_b'] if param.has_key('conv3a_b') else nn.init.Constant(0.),
                         nonlinearity = None)
    print 'conv3a', nn.layers.get_output_shape(conv3a)

    bn3a = BatchNormLayer(incoming = conv3a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn3a', nn.layers.get_output_shape(bn3a)

    conv3b = Conv2DLayer(incoming = bn3a, num_filters = 64, filter_size = (1, 7),
                         stride = 1, border_mode = 'same',
                         W = param['conv3b_w'] if param.has_key('conv3b_w') else nn.init.Normal(std = std),
                         b = param['conv3b_b'] if param.has_key('conv3b_b') else nn.init.Constant(0.),
                         nonlinearity = None)
    print 'conv3b', nn.layers.get_output_shape(conv3b)

    bn3b = BatchNormLayer(incoming = conv3b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn3b', nn.layers.get_output_shape(bn3b)

    pool3 = Pool2DLayer(incoming = bn3b, pool_size = (1, 2), stride = (1, 2))
    print 'pool3', nn.layers.get_output_shape(pool3)

    drop3 = nn.layers.DropoutLayer(incoming = pool3, p = p3)
    print 'drop3', nn.layers.get_output_shape(drop3)

    conv4 = Conv2DLayer(incoming = drop3, num_filters = 64, filter_size = (1, 7),
                        stride = 1, border_mode = 'same',
                        W = param['conv4_w'] if param.has_key('conv4_w') else nn.init.Normal(std = std),
                        b = param['conv4_b'] if param.has_key('conv4_b') else nn.init.Constant(0.),
                        nonlinearity = None)
    print 'conv4', nn.layers.get_output_shape(conv4)

    bn4 = BatchNormLayer(incoming = conv4, epsilon = 0.0000000001,
                         nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn4', nn.layers.get_output_shape(bn4)

    conv4a = Conv2DLayer(incoming = bn4, num_filters = 64, filter_size = (1, 7),
                         stride = 1, border_mode = 'same',
                         W = param['conv4a_w'] if param.has_key('conv4a_w') else nn.init.Normal(std = std),
                         b = param['conv4a_b'] if param.has_key('conv4a_b') else nn.init.Constant(0.),
                         nonlinearity = None)
    print 'conv4a', nn.layers.get_output_shape(conv4a)

    bn4a = BatchNormLayer(incoming = conv4a, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn4a', nn.layers.get_output_shape(bn4a)

    conv4b = Conv2DLayer(incoming = bn4a, num_filters = 64, filter_size = (1, 7),
                         stride = 1, border_mode = 'same',
                         W = param['conv4b_w'] if param.has_key('conv4b_w') else nn.init.Normal(std = std),
                         b = param['conv4b_b'] if param.has_key('conv4b_b') else nn.init.Constant(0.),
                         nonlinearity = None)
    print 'conv4b', nn.layers.get_output_shape(conv4b)

    bn4b = BatchNormLayer(incoming = conv4b, epsilon = 0.0000000001,
                          nonlinearity = nn.nonlinearities.leaky_rectify)
    print 'bn4b', nn.layers.get_output_shape(bn4b)

    pool4 = Pool2DLayer(incoming = bn4b, pool_size = (1, 2), stride = (1, 2))
    print 'pool4', nn.layers.get_output_shape(pool4)

    drop4 = nn.layers.DropoutLayer(incoming = pool4, p = p4)
    print 'drop4', nn.layers.get_output_shape(drop4)

    print
    if not param :
        param['conv1_w']=conv1.W
        param['conv2_w']=conv2.W
        param['conv2a_w']=conv2a.W
        param['conv3_w']=conv3.W
        param['conv3a_w']=conv3a.W
        param['conv3b_w']=conv3b.W
        param['conv4_w']=conv4.W
        param['conv4a_w']=conv4a.W
        param['conv4b_w']=conv4b.W
        param['conv1_b']=conv1.b
        param['conv2_b']=conv2.b
        param['conv2a_b']=conv2a.b
        param['conv3_b']=conv3.b
        param['conv3a_b']=conv3a.b
        param['conv3b_b']=conv3b.b
        param['conv3_b']=conv3.b
        param['conv3a_b']=conv3a.b
        param['conv3b_b']=conv3b.b
        return drop4, param
    else :
        return drop4

def build_train_valid(l_out):
    params = nn.layers.get_all_params(l_out, regularizable = True)
    wc_term = 0.5 * sum(T.sum(param ** 2) for param in params)
    
    x_batch = T.tensor4('x', theano.config.floatX)
    y_batch = T.matrix('y', 'int32')
    train_output = nn.layers.get_output(l_out, {l_in1: x_batch[:, :, :, -length_s:],
                                                l_in2: x_batch[:, :, :, -2 * length_s:],
                                                l_in3: x_batch})
    train_loss = nn.objectives.binary_crossentropy(train_output, y_batch)
    train_loss = nn.objectives.aggregate(train_loss, mode = 'mean')
    train_loss += wc * wc_term
    params = nn.layers.get_all_params(l_out, trainable = True)

    valid_output = nn.layers.get_output(l_out, {l_in1: x_batch[:, :, :, -length_s:],
                                                l_in2: x_batch[:, :, :, -2 * length_s:],
                                                l_in3: x_batch},
                                        deterministic = True)

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
