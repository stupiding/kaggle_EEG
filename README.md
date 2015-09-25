This project provides the solution of team daheimao for the Kaggle Grasp-and-Lift EEG Detection Competition. It is implemented using Python and mainly based on [Lasagne](http://lasagne.readthedocs.org/en/latest/). Many thanks to its authors.

This introduction is written by @daheimao.

#Overview
Before going to our solution, I want to mention the contribution of my team mate @stupiding. We prepared and completed this competition together. But unfortunately we missed the team merging deadline, for we are both new to kaggle and not familiar with the rules.

We are graduate students majored in neural networks and computer vision. My intention of participating in this competition is to evaluate the performance of recurrent convolutional neural network (RCNN) in processing time series data. RCNN is firstly proposed by I for image classification ([CVPR 2015](http://www.xlhu.cn/papers/Liang15-cvpr.pdf)), and then used for scene labeling (will appear in [NIPS 2015](https://nips.cc/Conferences/2015/AcceptedPapers) soon). In these two tasks input data is static image, and RCNN performs well. It turns out that RCNN also performs well for EEG data. Without any domain knowledge related modification, our best single model achieves 0.97652/0.97661 public/private LB scores. 

The pipeline of our solution is simple:

1. A set of candidate models (mostly RCNN) are trained in a 4-fold cross validation (CV) manner over the training set.

2. Greedy forward selection (GFS) is applied to the candidate model set, so that some models are selected for combination and submission.

More detailes can be found in the **Single model** and **Model selection** sections. The selected models are re-train over all the training data, and their predictions over the test data are averaged and submitted. 

#Single model
##RCNN
The key module of a RCNN is the recurrent convolutional layer (RCL), which can be seen as a specific form of RNN. A generic form of RNN is:

![rnn_equation](https://github.com/stupiding/kaggle_EEG/blob/master/image_folder/rnn.png)

In the RCL, the feed-forward and recurrent computation both take the form of convolution:

![rcl_equation](https://github.com/stupiding/kaggle_EEG/blob/master/image_folder/rcl.png)

During training or test phase, the RCL is unfolded through discrete time steps into a feed-forward subnetwork. The number of time steps (or recurrent iterations) is pre-fixed as a hyper-parameter. A subnetwork with 3 iterations is shown in the following figure:

![rcl_unfolding](https://github.com/stupiding/kaggle_EEG/blob/master/image_folder/rcl_unfolding.png)

RCL is a natural integration of RNN and CNN. Through recurrent iterations, arbitrarily deep subnetwork can be obtained while the number of free paramters is fixed. Furthermore, each unfolded RCL has several paths of different depths from input to output. The combination of these paths make it less prone to over-fitting.

RCNN is in principle a stack of RCLs (optionally interleaved with pooling layers). Convolutional layers are sometimes used in the first few layers in order to save time.

##Model for EEG signal
A input sample is treated as height-1-images, so that the efficient CUDNN library can be used. The input sample at time t is composed of the n 32-dimensiontal data at times t - n + 1, t - n + 2, ..., t (n is the length), so that future data is not used (more details in **No future data** section). If t < n zeros will be padded in the left. The input samples is organized in mini-batches of size 64, whose shape is denoted in the format of (batch size, channels, 1, n).

The pre-processing is very simple, just removing the mean of each input sample, which is the common practice in image classification. The model is neither subject-specific nor event-specific. A model receives the data of any subject as input, and outputs six predictions.

The structure of a typical RCNN is given below:

| Layer type      | Size                                                        | Output shape       |
| --------------- |:-----------------------------------------------------------:| ------------------ |
| Convolutional   | 128 1×9 filters                                             | (64, 128, 1, 3584) |
| Max pooling     | Pool size 4, stride 4                                       | (64, 128, 1, 896)  |
| RCL             | 128 1×1 feed-forward filters, 128 1×9 filters, 3 iterations | (64, 128, 1, 896)  |
| Max pooling     | Pool size 4, stride 4                                       | (64, 128, 1, 224)  |
| RCL             | 128 1×1 feed-forward filters, 128 1×9 filters, 3 iterations | (64, 128, 1, 224)  |
| Max pooling     | Pool size 4, stride 4                                       | (64, 128, 1, 56)   |
| RCL             | 128 1×1 feed-forward filters, 128 1×9 filters, 3 iterations | (64, 128, 1, 56)   |
| Max pooling     | Pool size 4, stride 4                                       | (64, 128, 1, 14)   |
| RCL             | 128 1×1 feed-forward filters, 128 1×9 filters, 3 iterations | (64, 128, 1, 14)   |
| Max pooling     | Pool size 2, stride 2                                       | (64, 128, 1, 7)    |
| Fully connected | 896×6                                                       | (64, 6)            |

This net is composed of one convolutional layer (for speed) and four RCLs, and all these layers are followed by a max pooling layer. Relu or leaky Relu activation is used in the convolutional layer and RCLs, and sigmoid activation is used in the fully connected layer for classification. The loss is the average of six binary cross entropy functions, corresponding to the six events.

Weight decay, dropout and batch normalization are used to regularize the model. For some models, data augmentation is also used: variable-length inputs are cropped and resized to a fixed length.

We also tried CNN, and the best CNN model achieves 0.97136/0.97297 public/priviate LB scores, which are significantly lower than those of the RCNN models.

#Model ensemble
A set of candidate models are trained in 4-fold cross validation manner, and then GFS is used to select an optimal subset.  Thus for each model its predictions over all the training data can be obtained by concatenating the validation results of all 4 CV splits. The models are combined in an average manner, that is, the predicted probabilities of different models are simply averaged to obtain the final prediction.

Initially, the selected subset is empty. In the first trial of GFS, the model with the highest AUC is selected and moved into the selected subset. In each of the following trials, the model which brings the largest AUC improvement when moved into the selected subset is chosen. GFS stops when the AUC stops improving. GFS is used for each event, so six subsets are selected.

GFS works well and its result keeps good consistency with the LB score. Unfortunately, this consistency is destroyed by a mistake. At first, we use fixed cv splits for all models. When there are three days before the deadline, we found some models are wrongly trained and decide to run a set of new models. To increase the variation of the new models,random cv splits are used for the new models. After this change, the LB scores always decrease with more new models. We did not found the answer until the last submission is made. By using random cv split, some models which do not complement each other may become “complementary” because they are trained by different splits, which means different training data.

The six subsets of the final submission contains 36 models in total, and achieves 0.98049/0.98029 public/private LB scores.

# No future data
For each time t, only the historical data is used as input. Zeros are padded in the left when the data has not enough length. Because no filtering pre-processing is used, it is easy for our models to statisfy the rule of no future data.

#Code
##Code overview
The code is written in Python (version 2.7.6), and the main dependencies include Lasagne (0.1.dev), Numpy, Sklearn, Scipy,  Skimage and Theano (0.7.0). Each model is run on a single Titan black GPU. Note that the code is **GPU-only** for the use of CUDNN.

##How to use the code

1. Generate the eeg_train.npy and eeg_test.npy with read_data.py
 
2. Prepare the model files in the folder **models/**. Some models (one validation model file and the non-cross-validation model file for each network structure)  have been put into this folder. The name of the model file follows certain rules. For example, the model **len3584_resize3_bs_c1r4p5_f9n256r35p1_v67.py** means:
 1. **shuffle**: the series used in this model are shuffled 
 2. **len3584**: the input has a length of 3584
 3. **resize3**: the resizing range (data augmentation) is 0.7 to 1.3
 4. **bs**: bootstrap is used
 5. **c1r4p5**: the model has 1 convolutional layer, 4 RCLs, and 5 pooling layers
 6. **f9**: the size of recurrent filters is 1×9
 7. **n256**: 256 filters are used for convolutional layer and RCLs
 8. **r35**: positive samples have a ratio of 0.35 over all inputs
 9. **v67**: this model uses series 0-5 for training, and 6, 7 for validation. When the series are shuffled, **v67** just means the 6th and 7th series of the shuffled list are used for validation.

3. Train the models using **train_net.py** with the command below:    
**THEANO_FLAGS=device=gpu0,floatX=float32 python train_net.py name_of_your_model**     
Note: step 3 and step 4 can be skipped, and use our model selection results file in **/model_combination/final_ffs.npy**.)

4. Over the validation results, use GFS to select a subset of models with the following steps:
 1. change into folder **model_combine/**
 2. copy all the vilid results files generated by **train_net.py** into **all_valid_results/** (which can be modified together with the parameter **result_dir** in **combine_valid.py**(line 7) )
 3. modify the parameter **results** in **combine_valid.py**(line 8). **results** contains the names of models to do gfs
 4. run **combine_valid.py**, and this step will generate **valid_conbined.npy**
 5. run **per_events_gfs.py** for selection, which requires the file **valid_conbined.npy**, and will generate **xx_per_event_gfs.npy**

5. Re-train the selected models over the entire training set, and average their output to obtain the submission file.
 1. change into folder **model_combine/**
 2. After the results over the test set are obtained, copy all the test results into **all_test_results/**
 3. run **combine_test.py**, this command need an argument as the following:    
    python combine_test.py xx_per_event_gfs.npy
 4. generate the submission file with the command below, in which the **submit_test.npy** is generated by **combine_test.npy** in the above step:    
    python submit.py submit_test.npy    
    and this command will generate the final submission file: **submit_test.npy.csv**
  
##Parameters in model file
1. **data_path**: path of the training data which is generated by **read_data.py**
2. **train_series**: series list used for training. In the final re-train stage, this list should contain all the 8 series(start from 0 to 7)
3. **valid_series**: series list used for validation. In the final re-train stage, this list should be empty.
4. **test_series**: series list used for test. This parameter is temporariy not used.
5. **events**: events list used for training, validation and test. This parameter can be set with a portion of all the 6 events for event-specific model.
6. **num_events**: length of the **events** list
7. **xx_data_params**: parameters used to generated chunks. 
 1. **section**: specify in which section to used the generate function. Can be set with **train/valid/test**
 2. **chunk_gen_fun**: specify which chunk-generation function to use. Can be set with
  1. **random_chunk_gen_fun**: generate chunks randomly
  2. **fixed_chunk_gen_fun**: generate chunks with fixed order for every model
  3. **test_valid_chunk_gen_fun**: generate chunks for validation set with series-major sequencial order
  4. **sequence_chunk_gen_fun**: generate chunks for test set with sequencial order of the input data
 3. **channels**: channels of data (32 in this project)
 4. **length**: input length of the model
 5. **preprocess**: method to preprocess the input data. Selected from the following:
  1. **per_sample_mean**: subtract each channel of input data with the mean of the sample's corresponding channel 
  2. **per_sample_mean_variance**: subtract each channel of input data with the mean of the sample's corresponding channel, and divide by the correspoing sample channel's variance
  3. **mean**: substract input data with the mean of all training data
  4. **mean_variance**: substract input data with the mean of all training data, and divide it with variance of all training data
 6. **chunk_size**: number of samples in each chunk
 7. **num_chunks**: number of chunks for training (used only in **train_data_params**)
 8. **pos_ratio**: the ratio of positive samples generated by function in each chunk
 9. **bootstrap**: specify whether to do bootstrap or not
 10. **neg_pool_size**: size of the negative pool in bootstrap
 11. **hard_ratio**: hard negative samples' ratio in the negative pool
 12. **easy_mode**: method to generate non-hard negative samples. Selected from:
  1. **random**: randomly select from the data samples used in bootstap section
  2. **easy**: select the negative samples that are easy to classify for the model
  3. **all**: select the samples randomly from neative pool
 13. **resize**: range of resize ratio (used in training section only)
 14: **pos_interval**: time interval to generate positive samples (used only for **fixed_chunk_gen_fun**)
 15: **pos_interval**: time interval to generate negative samples (used only for **fixed_chunk_gen_fun**)
 16. **test_lens**: list of sample length for test or validation
 17. **interval**: time interval to generate samples (used only in **test_valid_params**)
 18. **test_valid**: specify whether to do validation or not
8. **batch_size**
9. **momentum**
10. **wc**: weight decay
11. **display_freq**: frequency to display the training results
12. **valid_freq**: frequency to test the validation set
13. **bs_freq**: frequency to do bootstrap (to stop bootstap, this parameter should set with a number larger than **tarin_data_params**'s **num_chunks**)
14. **save_freq**: frequency to save models

#Acknowledgements
To summarize, this competition is very exciting although we made some mistakes. We want to give thanks to the organizers and all the other teams!
