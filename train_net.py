import numpy as np
import numpy.random as nr
import theano
import theano.tensor as T
import lasagne as nn
import data as data_
import metrics

import sys
from importlib import import_module
import time
import os
from itertools import izip
import cPickle

# set random seed
nr.seed(int(time.time()))

##########################################################################################
# parse the arguments
if len(sys.argv) < 2:
    sys.exit('Usage: train_net.py <model_name> [resume_file] [no_test]')

model_name = sys.argv[1]
resume_path = os.path.join('models', model_name)
if not os.path.exists(resume_path):
    os.mkdir(resume_path)

test_flag = True
if sys.argv[-1] == 'no_test' :
    test_flag = False
    print 'no_test'

# resume training if specified
if (test_flag is True and len(sys.argv) > 2) or (test_flag is False and len(sys.argv) > 3):
    print 'resume'
    resume_file = sys.argv[2]
    if not os.path.exists(resume_file):
        sys.exit('Resume_file does not exist')
    resume_data = np.load(resume_file)
    exp_id = resume_data['exp_id']
    resume_path, _ = os.path.split(resume_file)
else:
    exp_id = '%s' % time.strftime('%Y%m%d-%H%M%S', time.localtime())
    resume_path = os.path.join('models', model_name, exp_id)
    if not os.path.exists(resume_path):
        os.mkdir(resume_path)
    

##########################################################################################
# build the model
print
print "Experiment ID: %s" % exp_id
print

print "Building model"
model = import_module('models.%s' % model_name)
l_out = model.build_model()
x_shared, y_shared, idx, lr, iter_train, iter_valid=\
          model.build_train_valid(l_out)

chunk_idx = 0
metrics = model.metrics
metric_names = model.metric_names
losses_train = []
scores_train = []
scores_valid = []
best_record = [0, 0]
# resume history information if neccessary
if 'resume_data' in dir():
    print 'resume'
    nn.layers.set_all_param_values(l_out, resume_data['param_values'])
    chunk_idx = resume_data['chunk_idx']
    losses_train = resume_data['losses_train']
    scores_train = resume_data['scores_train']
    scores_valid = resume_data['scores_valid']
    best_record = resume_data['best_record']
    data_.neg_pool = resume_data['neg_pool']

chunk_idcs = np.arange(chunk_idx, model.train_data_params['num_chunks'])

##########################################################################################
# load data, and get the data generation functions for each section
data, labels = data_.load(model.data_path)

# exclude the second series of second subject
data[1,1]=np.zeros([0,32],'float32')
labels[1,1]=np.zeros([0,6],'float32')

# for subject-specific training
#data = data[model.subjects, :][:]
#labels = labels[model.subjects, :][:]

data_.init_sample_cells(labels, model.events, model.train_series, model.valid_series)

train_data_gen = lambda: data_.chunk_gen(
    getattr(data_, model.train_data_params['chunk_gen_fun'])(data[:, model.train_series],
                                                             labels[:, model.train_series],
                                                             model.events,
                                                             model.train_data_params))
valid_data_gen = lambda: data_.chunk_gen(
    getattr(data_, model.valid_data_params['chunk_gen_fun'])(data[:, model.valid_series],
                                                             labels[:, model.valid_series],
                                                             model.events,
                                                             model.valid_data_params))

bs_data_gen = lambda: data_.chunk_gen(
    getattr(data_, model.bs_data_params['chunk_gen_fun'])(data[:, model.train_series],
                                                          labels[:, model.train_series],
                                                          model.events,
                                                          model.bs_data_params))


##########################################################################################
do_validation = True
if 'test_valid' in model.test_data_params and model.test_data_params['test_valid'] == True:
    do_validation = True
else:
    do_validation = False

valid_result_folder = 'model_combine/all_valid_results/'
test_result_folder = 'model_combine/all_test_results/'

##########################################################################################
# start training
very_start = time.time()
for chunk_idx, (x_chunk, y_chunk, _) in izip(chunk_idcs, train_data_gen()):
    start_time = time.time()
    lr.set_value(model.lr_schedule(chunk_idx))
    x_shared.set_value(x_chunk)
    y_shared.set_value(y_chunk)

    if chunk_idx == chunk_idcs[0]:
        losses = []
        preds_train = np.zeros((0, model.num_events), 'float32')
        y_train = np.zeros((0, model.num_events), 'int32')        

    y_train = np.concatenate([y_train, y_chunk], axis = 0)
    num_batches_chunk = model.train_data_params['chunk_size'] / model.batch_size
    for b in np.arange(num_batches_chunk):
        loss, pred = iter_train(b)
        if np.isnan(loss):
            raise RuntimeError("NaN Detected.")
        losses.append(loss)
        preds_train = np.concatenate([preds_train, pred], axis = 0)
    
    if ((chunk_idx + 1) % model.display_freq) == 0:
        print
        print "Chunk %d/%d, lr = %.7f" % (chunk_idx + 1,
                                          model.train_data_params['num_chunks'],
                                          lr.get_value())
        
        mean_train_loss = np.mean(losses)
        print "  mean training loss:\t\t%.6f" % mean_train_loss
        losses_train.append(mean_train_loss)

        scores = [chunk_idx + 1]
        for i, metric in enumerate(metrics):
            scores.append(metric(y_train, preds_train))
            print "  %s:" % metric_names[i]
            print scores[-1]

        scores_train.append(scores)
        print "  The best score is %f, obtained in %d chunks" % (best_record[1], best_record[0])
        end_time = time.time()
        print "  elapsed time is %f seconds" % (end_time - start_time)
        print "  system time is ", time.strftime('%Y%m%d-%H%M%S', time.localtime())
        print "  elapsed time from the begining is %f seconds" % (end_time - very_start)
        losses = []
        preds_train = np.zeros((0, model.num_events), 'float32')
        y_train = np.zeros((0, model.num_events), 'int32')

    if ((chunk_idx + 1) % model.valid_freq) == 0 and do_validation is True:
        print
        print "Evaluating valid set"
        start_time = time.time()
        preds_valid = np.zeros((0, model.num_events), 'float32')
        y_valid = np.zeros((0, model.num_events), 'int32')
        for x_chunk, y_chunk, chunk_length in valid_data_gen():
            y_valid = np.concatenate([y_valid, y_chunk[:chunk_length, :]], axis = 0)    
            num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
            x_shared.set_value(x_chunk)
            chunk_output = np.zeros((0, model.num_events), 'float32')
            for b in np.arange(num_batches):
                pred = iter_valid(b)
                chunk_output = np.concatenate((chunk_output, pred), axis = 0)
            chunk_output = chunk_output[:chunk_length, :]
            preds_valid= np.concatenate((preds_valid, chunk_output), axis = 0)

        scores = [chunk_idx + 1]
        for i, metric in enumerate(metrics):
            scores.append(metric(y_valid, preds_valid))
            print "  %s:" % metric_names[i]
            print scores[-1]

        scores_valid.append(scores)
        if best_record[1] < scores[-1][-1]:
            best_record[0] = chunk_idx + 1
            best_record[1] = scores[-1][-1]
        print "  The best score is %f, obtained in %d chunks" % (best_record[1], best_record[0])
        end_time = time.time()
        print "  elapsed time is %f seconds" % (end_time - start_time)
        print

    if ((chunk_idx + 1) % model.bs_freq == 0) and (chunk_idx != chunk_idcs[-1]):
        print
        print "Bootstrap the training set"
        start_time = time.time()
        preds_bs = np.zeros((0, model.num_events), 'float32')
        y_bs = np.zeros((0, model.num_events), 'int32')
        for x_chunk, y_chunk, chunk_length in bs_data_gen():
            y_bs = np.concatenate([y_bs, y_chunk[:chunk_length, :]], axis = 0)
            num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
            x_shared.set_value(x_chunk)
            chunk_output = np.zeros((0, model.num_events), 'float32')
            for b in np.arange(num_batches):
                pred = iter_valid(b)
                chunk_output = np.concatenate((chunk_output, pred), axis = 0)
            chunk_output = chunk_output[:chunk_length, :]
            preds_bs = np.concatenate((preds_bs, chunk_output), axis = 0)

        scores = [chunk_idx + 1]
        for i, metric in enumerate(metrics):
            scores.append(metric(y_bs, preds_bs))
            print "  %s:" % metric_names[i]
            print scores[-1]
        data_.bootstrap(y_bs, preds_bs)

        end_time = time.time()
        print "  elapsed time is %f seconds" % (end_time - start_time)
        print

    if ((chunk_idx + 1) % model.save_freq) == 0:
        print
        print "Saving model"
        save_path = os.path.join(resume_path, '%d' % (chunk_idx + 1))
        with open(save_path, 'w') as f:
            cPickle.dump({
                'model': model_name,
                'exp_id': exp_id,
                'chunk_idx': chunk_idx + 1,
                'losses_train': losses_train,
                'scores_train': scores_train,
                'scores_valid': scores_valid,
                'best_record': best_record,
                'param_values': nn.layers.get_all_param_values(l_out),
                'neg_pool': data_.neg_pool
                }, f, cPickle.HIGHEST_PROTOCOL)



##########################################################################################
# test all valid and save results
if 'test_valid' in model.test_data_params and model.test_data_params['test_valid'] == True:
    data, labels = data_.load(model.data_path)
    test_valid_data_gen = lambda: data_.chunk_gen(
        getattr(data_, model.test_valid_params['chunk_gen_fun'])(data[:, model.valid_series],
                                                                labels[:, model.valid_series],
                                                                model.events,
                                                                model.test_valid_params))
    print
    print "Testing all valid samples"
    start_time = time.time()
    preds_test = np.zeros((0, model.num_events), 'float32')
    y_test = np.zeros((0, model.num_events), 'int32')
    idx = 1
    for x_chunk, y_chunk, chunk_length in test_valid_data_gen():
        t1 = time.time()
        y_test = np.concatenate([y_test, y_chunk[:chunk_length, :]], axis = 0)
        num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
        x_shared.set_value(x_chunk)
        chunk_output = np.zeros((0, model.num_events), 'float32')
        for b in np.arange(num_batches):
            pred = iter_valid(b)
            chunk_output = np.concatenate((chunk_output, pred), axis = 0)
        chunk_output = chunk_output[:chunk_length, :]
        preds_test= np.concatenate((preds_test, chunk_output), axis = 0)
        t2 = time.time()
        idx += 1
    save_valid_name = model_name[:-2] + str(model.valid_series[0]) + str(model.valid_series[1])
    save_path = os.path.join(valid_result_folder, 'test_valid_' + save_valid_name + '.npy')
    np.save(save_path, [y_test, preds_test])
    end_time = time.time()
    print "  elapsed time is %f seconds" % (end_time - start_time)


##########################################################################################
# test
if ((not model.test_data_params.has_key('test')) or model.test_data_params['test'] == True) and test_flag is True:
    data, labels = data_.load('eeg_test.npy')
    model.test_data_params['section'] = 'test'
    test_data_gen = lambda: data_.chunk_gen(
        getattr(data_, model.test_data_params['chunk_gen_fun'])(data,
                                                                labels,
                                                                model.events,
                                                                model.test_data_params))
    print
    print "Testing" 
    start_time = time.time()
    preds_test = np.zeros((0, model.num_events), 'float32')
    y_test = np.zeros((0, model.num_events), 'int32')
    idx = 1
    for x_chunk, y_chunk, chunk_length in test_data_gen():
        t1 = time.time()
        y_test = np.concatenate([y_test, y_chunk[:chunk_length, :]], axis = 0)
        num_batches = int(np.ceil(chunk_length / float(model.batch_size)))
        x_shared.set_value(x_chunk)
        chunk_output = np.zeros((0, model.num_events), 'float32')
        for b in np.arange(num_batches):
            pred = iter_valid(b)
            chunk_output = np.concatenate((chunk_output, pred), axis = 0)
        chunk_output = chunk_output[:chunk_length, :]
        preds_test= np.concatenate((preds_test, chunk_output), axis = 0)
        t2 = time.time()
        idx += 1
    
    save_path = os.path.join(test_result_folder, 'test_' + model_name + '.npy')
    np.save(save_path, [y_test, preds_test])

end_time = time.time()
print "  elapsed time is %f seconds" % (end_time - start_time)
