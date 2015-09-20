This project provides the solution of team daheimao for the Kaggle Grasp-and-Lift EEG Detection Competition. It is implemented using Python and mainly based on [Lasagne](http://lasagne.readthedocs.org/en/latest/). Many thanks to its authors.

This introduction is written by daheimao.

#Overview
Before going to our solution, I want to mention the contribution of my team mate @stupiding. We prepared and completed this competition together. But unfortunately we missed the team merging deadline, for we are both new to kaggle and not familiar with the rules.

I am a Ph.D student majored in neural networks and computer vision. My intention of participating in this competition is to evaluate the performance of recurrent convolutional neural network (RCNN) in processing time series data. RCNN is firstly proposed by I for image classification (http://www.xlhu.cn/papers/Liang15-cvpr.pdf), and then used for scene labeling (will appear in NIPS 2015 soon). In both of these two tasks the data is static image, and RCNN performs well. It turns out that RCNN also performs well for EEG data. Without any domain knowledge related modification, our best single model achieves 0.97652/0.97661 public/private LB scores. 

The pipeline of our solution is simple:
1. A set of candidate models (mostly RCNN) are trained in a 4-fold cross validation (CV) manner over the training set.
2. Greed forward selection (GFS) is applied to the candidate model set, and some models are selected for combination and submission.
More detailes can be found in Single model and Model selection sections. The selected models are re-train over all the training data, and their predictions over the test data are averaged and submitted. Note that for each model, its predictions of all training data can be obtained by concatenating the validation results of all 4 CV splits.

#Single model
The structure of a typical RCNN is given below:
------------------------------
|Layer type|Size|Output shape|
------------------------------
Convolutional|128 1×9 filters|(64, 128, 1, 3584)|
-------------------------------------------------

Max pooling                   Pool size 4, stride 4	              (64, 128, 1, 896)

RCL                           Feed-forward 128 1×1 filters,         (64, 128, 1, 896)
                              Recurrent 128 1×9 filters,
                              3 iterations
                              
Max pooling	                  Pool size 4, stride 4	              (64, 128, 1, 448)

RCL                           Feed-forward 128 1×1 filters,         (64, 128, 1, 448)
                              Recurrent 128 1×9 filters,
                              3 iterations
                              
Max pooling	                  Pool size 4, stride 4
(64, 128, 1, 112)
RCL                           Feed-forward 128 1×1 filters,         (64, 128, 1, 112)
                              Recurrent 128 1×9 filters,
                              3 iterations
                              
Max pooling	                  Pool size 4, stride 4
(64, 128, 1, 28)
RCL                           Feed-forward 128 1×1 filters,         (64, 128, 1, 28)
                              Recurrent 128 1×9 filters,
                              3 iterations
                              
Max pooling	                  Pool size 4, stride 4	              (64, 128, 1, 7)

Fully connected	            896×6	                                (64, 6)

The net is composed of one convolutional layer (for speed) and four RCLs, and all these layers are followed by a max pooling layer with size 4 and stride 4. Relu or leaky Relu is used in the convolutional layer and RCLs, and sigmoid is used in the fully connected layer for classification.

Weight decay, dropout and batch normalization are used to regularize the model. For some models, data augmentation is also used: the input is randomly resized. But this operation seems to bring only slight improvement.

We also tried CNN, and the best CNN achieves 0.97136/0.97297 LB/PB scores, which is lower than all our RCNN models.

Model ensemble
We use greedy forward selection (GFS) method to select models for combining the ensemble. A set of models are trained in 4-fold cross validation manner. Thus for each model its predictions over all the training data is obtained and GFS is done based on all these predictions. In the first trial, we choose the model with the best AUC, and this model is moved to the ensemble set. In each following trial, the model which brings the largest AUC improvement when moved in the ensemble is chosen. GFS stops when the AUC stops improving.

GFS works well and its result keeps very good consistency with the LB score. Unfortunately, this consistency is destroyed by a mistake. At first, we use fixed cv splits for all models. When there are three days before the deadline, we found some models are wrongly trained and decide to run as many as new models. To increase the variation of the new models, we use random cv split for the new models. After this change, our LB scores always decrease with more new models. We did not found the answer until I made the last submission… By using random cv split, some models which do not complement each other may become “complementary” because they are trained by different splits, which means different training data.

To summarize, this competition is exciting although we made some mistakes. We want to give our thanks to the organizers and all the other teams!



######################################################################################################
To use this code, you should do the following:
1. Install Lasagne

2. Generate the eeg_train.npy and eeg_test.npy with read_data.py

3. Write your own model file in the folder "models/" with Lasagne grammar like the example models in there, e.g. 
   resize3_c1r4p5_f9n128r35p1_v67.py, len2560p1_resize3_bs_c1r4p5_f9n128r35p1_v67.py and 
   len4096_downsample4_resize3_bs_c7p7_f9n128_r35p1_v67.py

4. Start to train your model with the command below (take the resize3_c1r4p5_f9n128r35p1_v67 model for example):
   THEANO_FLAGS=device=gpu0,floatX=float32 python train_net.py resize3_c1r4p5_f9n128r35p1_v67
     # Note: this is for GPU-available device only
     # For details of model parameters, go to models/resize3_c1r4p5_f9n128r35p1_v67.py



######################################################################################################
To combine diffent models and get the submition files:
1. Change the python files in "model_combine/" according to your own environment
2. Execute the following python files in order:
   a. combine_valid.py 
   b. per_events_ffs.py
   c. combine_test.py
   d. submit.py
