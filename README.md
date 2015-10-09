This project provides the solution of team daheimao for the Kaggle Grasp-and-Lift EEG Detection Competition. It is implemented using Python and mainly based on [Lasagne](http://lasagne.readthedocs.org/en/latest/). Many thanks to its authors.

#Overview
I am a Ph.D student majored in neural networks and computer vision. My intention of participating in this competition is to evaluate the performance of recurrent convolutional neural network (RCNN) in processing time series data. RCNN is firstly proposed by I for image classification ([CVPR 2015](http://www.xlhu.cn/papers/Liang15-cvpr.pdf)), and then used for scene labeling (will appear in [NIPS 2015](https://nips.cc/Conferences/2015/AcceptedPapers) soon). In these two tasks input data is static image, and RCNN performs well. It turns out that RCNN also performs well for EEG data. Without any domain knowledge related modification, the best single model achieves 0.97652/0.97661 public/private LB scores. 

The pipeline of the solution is simple:

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
| Convolutional   | 256 1×9 filters                                             | (64, 256, 1, 3584) |
| Max pooling     | Pool size 4, stride 4                                       | (64, 256, 1, 896)  |
| RCL             | 256 1×1 feed-forward filters, 256 1×9 filters, 3 iterations | (64, 256, 1, 896)  |
| Max pooling     | Pool size 4, stride 4                                       | (64, 256, 1, 224)  |
| RCL             | 256 1×1 feed-forward filters, 256 1×9 filters, 3 iterations | (64, 256, 1, 224)  |
| Max pooling     | Pool size 4, stride 4                                       | (64, 256, 1, 56)   |
| RCL             | 256 1×1 feed-forward filters, 256 1×9 filters, 3 iterations | (64, 256, 1, 56)   |
| Max pooling     | Pool size 4, stride 4                                       | (64, 256, 1, 14)   |
| RCL             | 256 1×1 feed-forward filters, 256 1×9 filters, 3 iterations | (64, 256, 1, 14)   |
| Max pooling     | Pool size 2, stride 2                                       | (64, 256, 1, 7)    |
| Fully connected | 1792×6                                                      | (64, 6)            |

This net is composed of one convolutional layer (for speed) and four RCLs, and all these layers are followed by a max pooling layer. Relu or leaky Relu activation is used in the convolutional layer and RCLs, and sigmoid activation is used in the fully connected layer for classification. The loss is the average of six binary cross entropy functions, corresponding to the six events.

Weight decay, dropout and batch normalization are used to regularize the model. For some models, data augmentation is also used: variable-length inputs are cropped and resized to a fixed length.

I also tried CNN, and the best CNN model achieves 0.97136/0.97297 public/priviate LB scores, which are significantly lower than those of the RCNN models.

#Model ensemble
A set of candidate models are trained in 4-fold cross validation manner, and then GFS is used to select an optimal subset.  Thus for each model its predictions over all the training data can be obtained by concatenating the validation results of all 4 CV splits. The models are combined in an average manner, that is, the predicted probabilities of different models are simply averaged to obtain the final prediction.

Initially, the selected subset is empty. In the first trial of GFS, the model with the highest AUC is selected and moved into the selected subset. In each of the following trials, the model which brings the largest AUC improvement when moved into the selected subset is chosen. GFS stops when the AUC stops improving. GFS is used for each event, so six subsets are selected.

GFS works well and its result keeps good consistency with the LB score. Unfortunately, this consistency is destroyed by a mistake. At first, I use fixed cv splits for all models. When there are three days before the deadline, I found some models are wrongly trained and decide to run a set of new models. To increase the variation of the new models,random cv splits are used for the new models. After this change, the LB scores always decrease with more new models. I did not found the answer until the last submission is made. By using random cv split, some models which do not complement each other may become “complementary” because they are trained by different splits, which means different training data.

The six subsets of the final submission contains 36 models in total, and achieves 0.98049/0.98029 public/private LB scores.

# No future data
For each time t, only the historical data is used as input. Zeros are padded in the left when the data has not enough length. Because no filtering pre-processing is used, it is easy for the models to statisfy the rule of no future data.

#Code
##Code overview
The code is written in Python (version 2.7.6), and the main dependencies include Lasagne (0.1.dev), Numpy, Sklearn, Scipy,  Skimage and Theano (0.7.0). Each model is run on a single Titan black GPU. Note that the code is **GPU-only** for the use of CUDNN.

##How to use the code
Note: there are 5 steps to use the code. Step 1 and step 2 are preparations that one needs to do before training the models. And step 3 and step 4 are used to generate cross-validation results and then do model selection. They are time-consuming and can be skipped. I have prepared the result of model selection. (The file **selection_result.npy** in folder **model_selection/**. After step 5, the **selection_result.npy** can be directly used by **make_submission.py** to generate the submission. The list of selected model names can be found in models/README.md)

1. Generate the eeg_train.npy and eeg_test.npy with read_data.py
 
2. Prepare the model files in the folder **models/** according to **models/README.md**. Some models (**only validation v67 model file and the non-cross-validation model file for each network structure**)  have been put into this folder. The name of the model file follows certain rules. For example, the model **len3584_resize3_bs_c1r4p5_f9n256r35p1_v67.py** means:
 1. **len3584**: the input has a length of 3584
 2. **resize3**: the resizing range (data augmentation) is 0.7 to 1.3
 3. **bs**: bootstrap is used
 4. **c1r4p5**: the model has 1 convolutional layer, 4 RCLs, and 5 pooling layers
 5. **f9**: the size of recurrent filters is 1×9
 6. **n256**: 256 filters are used for convolutional layer and RCLs
 7. **r35**: positive samples have a ratio of 0.35 over all inputs
 8. **v67**: this model uses series split 01, 23, 45 for training, and 67 for validation. When the model name has no **vxx**, it means this model uses all 8 series for training and no validation.    

3. Train the models using **train_net.py** with the command below:
**THEANO_FLAGS=device=gpu0,floatX=float32 python train_net.py name_of_your_model**     
This commmand will only process one validation at a time. To do cross validation, you need to execute the command for every validation set, namely **xx_v01**, **xx_v23**, **xx_v45** and **xx_v67**.  

4. Over the validation results, use GFS to select a subset of models with the following steps:
 1. change into folder **model_combine/**
 2. modify the parameter **results** in **combine_valid.py**(line 8). **results** contains the names of models to do gfs
 3. run **combine_valid.py**, and this step will generate **valid_conbined.npy**
 4. run **per_events_gfs.py** for selection, which requires the file **valid_conbined.npy**, and will generate **selection_result.npy**

5. Re-train the selected models over the entire training set
6. Combine models' prediction and generate the **submission.csv** file
 1. change into folder **model_combine/**
 2. generate the submission file with the command below, in which the **selection_result.npy** is generated by **per_events_gfs.py** in the above step:    
    **python make_submission.py**    
    and this command will generate the final submission file: **submission.csv**
  

#Acknowledgements
To summarize, this competition is very exciting although I made some mistakes. I want to give thanks to the organizers and all the other teams!
