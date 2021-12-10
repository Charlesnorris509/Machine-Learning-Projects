#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.display import display, HTML, Javascript

notebook_theme = 'carrot'
color_maps = {'turquoise': ['#1abc9c', '#e8f8f5', '#d1f2eb', '#a3e4d7', '#76d7c4', '#48c9b0', '#1abc9c', '#17a589', '#148f77', '#117864', '#0e6251'], 'green': ['#16a085', '#e8f6f3', '#d0ece7', '#a2d9ce', '#73c6b6', '#45b39d', '#16a085', '#138d75', '#117a65', '#0e6655', '#0b5345'], 'emerald': ['#2ecc71', '#eafaf1', '#d5f5e3', '#abebc6', '#82e0aa', '#58d68d', '#2ecc71', '#28b463', '#239b56', '#1d8348', '#186a3b'], 'nephritis': ['#27ae60', '#e9f7ef', '#d4efdf', '#a9dfbf', '#7dcea0', '#52be80', '#27ae60', '#229954', '#1e8449', '#196f3d', '#145a32'], 'peter': ['#3498db', '#ebf5fb', '#d6eaf8', '#aed6f1', '#85c1e9', '#5dade2', '#3498db', '#2e86c1', '#2874a6', '#21618c', '#1b4f72'], 'belize': ['#2980b9', '#eaf2f8', '#d4e6f1', '#a9cce3', '#7fb3d5', '#5499c7', '#2980b9', '#2471a3', '#1f618d', '#1a5276', '#154360'], 'amethyst': ['#9b59b6', '#f5eef8', '#ebdef0', '#d7bde2', '#c39bd3', '#af7ac5', '#9b59b6', '#884ea0', '#76448a', '#633974', '#512e5f'], 'wisteria': ['#8e44ad', '#f4ecf7', '#e8daef', '#d2b4de', '#bb8fce', '#a569bd', '#8e44ad', '#7d3c98', '#6c3483', '#5b2c6f', '#4a235a'], 'wet': ['#34495e', '#ebedef', '#d6dbdf', '#aeb6bf', '#85929e', '#5d6d7e', '#34495e', '#2e4053', '#283747', '#212f3c', '#1b2631'], 'midnight': ['#2c3e50', '#eaecee', '#d5d8dc', '#abb2b9', '#808b96', '#566573', '#2c3e50', '#273746', '#212f3d', '#1c2833', '#17202a'], 'sunflower': ['#f1c40f', '#fef9e7', '#fcf3cf', '#f9e79f', '#f7dc6f', '#f4d03f', '#f1c40f', '#d4ac0d', '#b7950b', '#9a7d0a', '#7d6608'], 'orange': ['#f39c12', '#fef5e7', '#fdebd0', '#fad7a0', '#f8c471', '#f5b041', '#f39c12', '#d68910', '#b9770e', '#9c640c', '#7e5109'], 'carrot': ['#e67e22', '#fdf2e9', '#fae5d3', '#f5cba7', '#f0b27a', '#eb984e', '#e67e22', '#ca6f1e', '#af601a', '#935116', '#784212'], 'pumpkin': ['#d35400', '#fbeee6', '#f6ddcc', '#edbb99', '#e59866', '#dc7633', '#d35400', '#ba4a00', '#a04000', '#873600', '#6e2c00'], 'alizarin': ['#e74c3c', '#fdedec', '#fadbd8', '#f5b7b1', '#f1948a', '#ec7063', '#e74c3c', '#cb4335', '#b03a2e', '#943126', '#78281f'], 'pomegranate': ['#c0392b', '#f9ebea', '#f2d7d5', '#e6b0aa', '#d98880', '#cd6155', '#c0392b', '#a93226', '#922b21', '#7b241c', '#641e16'], 'clouds': ['#ecf0f1', '#fdfefe', '#fbfcfc', '#f7f9f9', '#f4f6f7', '#f0f3f4', '#ecf0f1', '#d0d3d4', '#b3b6b7', '#979a9a', '#7b7d7d'], 'silver': ['#bdc3c7', '#f8f9f9', '#f2f3f4', '#e5e7e9', '#d7dbdd', '#cacfd2', '#bdc3c7', '#a6acaf', '#909497', '#797d7f', '#626567'], 'concrete': ['#95a5a6', '#f4f6f6', '#eaeded', '#d5dbdb', '#bfc9ca', '#aab7b8', '#95a5a6', '#839192', '#717d7e', '#5f6a6a', '#4d5656'], 'asbestos': ['#7f8c8d', '#f2f4f4', '#e5e8e8', '#ccd1d1', '#b2babb', '#99a3a4', '#7f8c8d', '#707b7c', '#616a6b', '#515a5a', '#424949']}

color_maps = {i: color_maps[i] for i in color_maps if i not in ['clouds', 'silver', 'concrete', 'asbestos', 'wet asphalt', 'midnight blue', 'wet']}

CMAP = 'Oranges'
prompt = '#1DBCCD'
main_color = '#E58F65' # color_maps[notebook_theme]
strong_main_color = '#EB9514' # = color_maps[notebook_theme] 
custom_colors = [strong_main_color, main_color]

# ----- Notebook Theme -----

html_contents ="""
<!DOCTYPE html>
<html lang="en">
    <head>
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Raleway">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Oswald">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Open Sans">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
        .title-section{
            font-family: "Oswald", Arial, sans-serif;
            font-weight: bold;
            color: "#6A8CAF";
            letter-spacing: 6px;
        }
        hr { border: 1px solid #E58F65 !important;
             color: #E58F65 !important;
             background: #E58F65 !important;
           }
        body {
            font-family: "Open Sans", sans-serif;
            }        
        </style>
    </head>    
</html>
"""

import os
if not os.path.exists("../input/g-research-crypto-forecasting/"): os.chdir('/t/Datasets/kaggle_crypto/internal')
HTML(html_contents)    


# <hr>

# >### 🔥🔥 Crypto Prediction: Xgboost Regressor 🔥🔥
# >
# > **Just a simple pipeline going from zero to a valid submission**
# >
# > We train one `XGBRegressor` for each asset over a very very naive set of features (the input dataframe `['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']`), we get the predictions correctly using the iterator and we submit. No validation for now, no cross validation... nothing at all lol: just the bare pipeline!
# >
# >
# >This notebook follows the ideas presented in my "Initial Thoughts" [here][1].
# 
# [1]: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/284903
# [2]: https://www.kaggle.com/yamqwe/let-s-talk-validation-grouptimeseriessplit
# 
# <div class="alert alert-block alert-warning">
# <b>References:</b>
# <ul>
#     <li><a href = "https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285726">Dataset Thread</a></li>
#     <li><a href = "https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/284903">Initial Thoughts Thread
# </a></li>
#     <li><a href = "https://www.kaggle.com/yamqwe/let-s-talk-validation-grouptimeseriessplit">Validation Thread
# </a></li>
# </ul>
# </div>

# ____
# 
# #### <center>All notebooks in the series 👇</center>
# 
# | CV + Model | Hyperparam Optimization  | Time Series Models | Feature Engineering |
# | --- | --- | --- | --- |
# | [Neural Network Starter](https://www.kaggle.com/yamqwe/purgedgrouptimeseries-cv-with-extra-data-nn) | [MLP + AE](https://www.kaggle.com/yamqwe/bottleneck-encoder-mlp-keras-tuner)        | [LSTM](https://www.kaggle.com/yamqwe/time-series-modeling-lstm) | [Technical Analysis #1](https://www.kaggle.com/yamqwe/crypto-prediction-technical-analysis-features) |
# | [LightGBM Starter](https://www.kaggle.com/yamqwe/purgedgrouptimeseries-cv-with-extra-data-lgbm)     | [LightGBM](https://www.kaggle.com/yamqwe/purged-time-series-cv-lightgbm-optuna)     | [Wavenet](https://www.kaggle.com/yamqwe/time-series-modeling-wavenet)  | [Technical Analysis #2](https://www.kaggle.com/yamqwe/crypto-prediction-technical-analysis-feats-2) |
# | [Catboost Starter](https://www.kaggle.com/yamqwe/purgedgrouptimeseries-cv-extra-data-catboost)      | [Catboost](https://www.kaggle.com/yamqwe/purged-time-series-cv-catboost-gpu-optuna) | [Multivariate-Transformer [written from scratch]](https://www.kaggle.com/yamqwe/time-series-modeling-multivariate-transformer) | [Time Series Agg](https://www.kaggle.com/yamqwe/features-all-time-series-aggregations-ever) | 
# | [XGBoost Starter](https://www.kaggle.com/yamqwe/xgb-extra-data)                                            | [XGboost](https://www.kaggle.com/yamqwe/purged-time-series-cv-xgboost-gpu-optuna) | [N-BEATS](https://www.kaggle.com/yamqwe/crypto-forecasting-n-beats) |  [Neutralization](https://www.kaggle.com/yamqwe/g-research-avoid-overfit-feature-neutralization/) |
# | [Supervised AE [Janestreet 1st]](https://www.kaggle.com/yamqwe/1st-place-of-jane-street-adapted-to-crypto) | [Supervised AE [Janestreet 1st]](https://www.kaggle.com/yamqwe/1st-place-of-jane-street-keras-tuner) | [DeepAR](https://www.kaggle.com/yamqwe/probabilistic-forecasting-deepar/) | [Quant's Volatility Features](https://www.kaggle.com/yamqwe/crypto-prediction-volatility-features) |
# | [Transformer)](https://www.kaggle.com/yamqwe/let-s-test-a-transformer)                                     | [Transformer](https://www.kaggle.com/yamqwe/sh-tcoins-transformer-baseline)  |  | ⏳Target Engineering |
# | [TabNet Starter](https://www.kaggle.com/yamqwe/tabnet-cv-extra-data)                                       |  |  |⏳Fourier Analysis | 
# | [Reinforcement Learning (PPO) Starter](https://www.kaggle.com/yamqwe/g-research-reinforcement-learning-starter) | | | ⏳Wavelets | 
# 
# ____

# 
# #### **<span>Dataset Structure</span>**
# 
# > **train.csv** - The training set
# > 
# > 1.  timestamp - A timestamp for the minute covered by the row.
# > 2.  Asset_ID - An ID code for the cryptoasset.
# > 3.  Count - The number of trades that took place this minute.
# > 4.  Open - The USD price at the beginning of the minute.
# > 5.  High - The highest USD price during the minute.
# > 6.  Low - The lowest USD price during the minute.
# > 7.  Close - The USD price at the end of the minute.
# > 8.  Volume - The number of cryptoasset u units traded during the minute.
# > 9.  VWAP - The volume-weighted average price for the minute.
# > 10. Target - 15 minute residualized returns. See the 'Prediction and Evaluation section of this notebook for details of how the target is calculated.
# > 11. Weight - Weight, defined by the competition hosts [here](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition)
# > 12. Asset_Name - Human readable Asset name.
# > 
# >
# > **example_test.csv** - An example of the data that will be delivered by the time series API.
# > 
# > **example_sample_submission.csv** - An example of the data that will be delivered by the time series API. The data is just copied from train.csv.
# > 
# > **asset_details.csv** - Provides the real name and of the cryptoasset for each Asset_ID and the weight each cryptoasset receives in the metric.
# > 
# > **supplemental_train.csv** - After the submission period is over this file's data will be replaced with cryptoasset prices from the submission period. In the Evaluation phase, the train, train supplement, and test set will be contiguous in time, apart from any missing data. The current copy, which is just filled approximately the right amount of data from train.csv is provided as a placeholder.
# >
# > - 📌 There are 14 coins in the dataset
# >
# > - 📌 There are 4 years  in the [full] dataset

# In[3]:


css_file = '''
div #notebook {
background-color: white;
font-family: 'Open Sans', Helvetica, sans-serif;
line-height: 20px;
}

#notebook-container {
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s; /* light orange */
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s; /* light orange */
font-weight: 600;
}

.rendered_html code {
    background-color: #efefef; /* light gray */
}

.CodeMirror {
color: #8c8c8c; /* dark gray */
padding: 0.7em;
}

div.input_area {
border: none;
    background-color: %s; /* rgba(229, 143, 101, 0.1); light orange [exactly #E58F65] */
    border-top: 2px solid %s; /* light orange */
}

div.input_prompt {
color: %s; /* light blue */
}

div.output_prompt {
color: %s; /* strong orange */
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s; /* light orange */
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s; /* light orange */
}

.edit_mode div.cell.selected:before {
background: %s; /* light orange */
}

.edit_mode div.cell.selected {
border-color: %s; /* light orange */

}
'''
def to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:])[0], to_rgb(main_color[1:])[1], to_rgb(main_color[1:])[2])
open('notebook.css', 'w').write(css_file % (main_color, main_color, main_color_rgba, main_color,  prompt, strong_main_color, main_color, main_color, main_color, main_color))
from IPython.core.display import display, HTML, Javascript
def nb(): return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# In[4]:


from IPython.core.display import display, HTML, Javascript
# def nb(): return HTML("<style>" + open("../input/starter-utils/css_oranges.css", "r").read() + "</style>")
def nb(): return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# <center><img src="https://i.ibb.co/8bvJY8B/xgboost-logo.png" height=250 width=250></center>
# <hr>
# <center>XGBoost🏻</center>

# XGBoost is a favorite choice on kaggle and it doesn't look like it is going anywhere! 
# It is basiclly the a version of gradient boosting machines framework that made the approach so popular.
# 
# **It is usually included in winning ensembles on Kaggle when solving a tabular problem**
# 
# XGBoost algorithm provides large range of hyperparameters. In order to get the best performance out of it, we need to know to tune them.
# 
# ><h4>TL;DR: What makes XGBoost great:</h4>
# >
# >1. XGBoost was the first wide-spread GBM framework so it has "more mileage" then all other frameworks.
# >2. Easy to use 
# >3. When using GPU it is usually faster than nearly all other gradient boosting algorithms that use GPU.
# >4. A very powerful gradient boosting. 
# 
# <h4>Leaf growth in XGBoost</h4>
# 
# XGboost splits up to the specified max_depth hyperparameter and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. It uses this approach since sometimes a split of no loss reduction may be followed by a split with loss reduction. XGBoost can also perform leaf-wise tree growth (as LightGBM).
# 
# Normally it is impossible to enumerate all the possible tree structures q. A greedy algorithm that starts from a single leaf and iteratively adds branches to the tree is used instead. Assume that I_L and I_R are the instance sets of left and right nodes after the split. Then the loss reduction after the split is given by,
# 
# ![](https://i.imgur.com/jzyLh81.png)
# 
# <h4>XGBoost vs LightGBM</h4>
# 
# LightGBM uses a novel technique of Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value while XGBoost uses pre-sorted algorithm & Histogram-based algorithm for computing the best split. Here instances mean observations/samples.
# 
# Let's see how pre-sorting splitting works-
# 
# - For each node, enumerate over all features
# 
# - For each feature, sort the instances by feature value
# 
# - Use a linear scan to decide the best split along that feature basis information gain
# 
# - Take the best split solution along all the features
# 
# In simple terms, Histogram-based algorithm splits all the data points for a feature into discrete bins and uses these bins to find the split value of histogram. While, it is efficient than pre-sorted algorithm in training speed which enumerates all possible split points on the pre-sorted feature values, it is still behind GOSS in terms of speed.
# 
# <h4>XGBoost Model Parameters</h4>
# 
# >For an exhaustive overview of all parameters [see here](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# 
# **objective [default=reg:linear]**
# 
# This defines the loss function to be minimized. Mostly used values are:
# 
# - binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
# 
# - multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
# you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
# 
# - multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
# 
# **eval_metric [ default according to objective ]**
# 
# The metric to be used for validation data. The default values are rmse for regression and error for classification.
# Typical values are:
# 
# - rmse – root mean square error
# 
# - mae – mean absolute error
# 
# - logloss – negative log-likelihood
# 
# - error – Binary classification error rate (0.5 threshold)
# 
# - merror – Multiclass classification error rate
# 
# - mlogloss – Multiclass logloss
# 
# - auc: Area under the curve
# 
# 
# **eta [default=0.3]**
# 
# - Analogous to learning rate in GBM.
# - Makes the model more robust by shrinking the weights on each step.
# - Typical final values to be used: 0.01-0.2
# 
# **colsample_bytree**:  We can create a random sample of the features (or columns) to use prior to creating each decision tree in the boosted model. That is, tuning Column Sub-sampling in XGBoost By Tree. This is controlled by the colsample_bytree parameter. The default value is 1.0 meaning that all columns are used in each decision tree. A fraction (e.g. 0.6) means a fraction of columns to be subsampled. We can evaluate values for colsample_bytree between 0.1 and 1.0 incrementing by 0.1.
# 
# <h3>Regularization in XGBoost</h3>
# 
# XGBoost adds built-in regularization to achieve accuracy gains beyond gradient boosting. Regularization is the process of adding information to reduce variance and prevent overfitting.
# 
# Although data may be regularized through hyperparameter fine-tuning, regularized algorithms may also be attempted. For example, Ridge and Lasso are regularized machine learning alternatives to LinearRegression.
# 
# XGBoost includes regularization as part of the learning objective, as contrasted with gradient boosting and random forests. The regularized parameters penalize complexity and smooth out the final weights to prevent overfitting. XGBoost is a regularized version of gradient boosting.
# 
# Mathematically, XGBoost's learning objective may be defined as follows:
# 
# $$obj(θ) = l(θ) + Ω (θ)$$
# 
# Here, **l(θ)**  is the loss function, which is the Mean Squared Error (MSE) for regression, or the log loss for classification, and **Ω (θ)** is the regularization function, a penalty term to prevent over-fitting. Including a regularization term as part of the objective function distinguishes XGBoost from most tree ensembles.
# 
# The learning objective for the th boosted tree can now be rewritten as follows:
# 
# ![img](https://i.imgur.com/IRNCrvM.png)
# 
# **reg_alpha and reg_lambda** : First note the loss function is defined as
# 
# ![img](https://i.imgur.com/aw1Hod9.png)
# 
# >So the above is how the regularized objective function looks like if you want to allow for the inclusion of a L1 and a L2 parameter in the same model
# 
# `reg_alpha` and `reg_lambda` control the L1 and L2 regularization terms, which in this case limit how extreme the weights at the leaves can become. Higher values of alpha mean more L1 regularization. See the documentation [here](http://xgboost.readthedocs.io/en/latest///parameter.html#parameters-for-tree-booster).
# 
# Since L1 regularization in GBDTs is applied to leaf scores rather than directly to features as in logistic regression, it actually serves to reduce the depth of trees. This in turn will tend to reduce the impact of less-predictive features. We might think of L1 regularization as more aggressive against less-predictive features than L2 regularization.
# 
# These two regularization terms have different effects on the weights; L2 regularization (controlled by the lambda term) encourages the weights to be small, whereas L1 regularization (controlled by the alpha term) encourages sparsity — so it encourages weights to go to 0. This is helpful in models such as logistic regression, where you want some feature selection, but in decision trees we’ve already selected our features, so zeroing their weights isn’t super helpful. For this reason, I found setting a high lambda value and a low (or 0) alpha value to be the most effective when regularizing.
# 
# >[From this Paper](https://arxiv.org/pdf/1603.02754.pdf)
# 
# 
# <div class="alert alert-block alert-info">
# <b>Read More:</b>
# <ul>
#     <li><a href = "https://github.com/dmlc/xgboost">XGBoost Github Documentation</a></li>
#     <li><a href = "https://xgboost.readthedocs.io/en/stable/parameter.html">XGBoost Parameters</a></li>
#     <li><a href = "https://xgboost.readthedocs.io/en/stable/">Official Documentation</a></li>    
# </ul>
# </div>
# 
# ____
# 
# <h3>All Parameters Overview</h3>
# 
# ____
# 
# Before diving into the actual parameters of XGBoost, Let's define three types of parameters: General Parameters, Booster Parameters and Task Parameters.
# 
# 1. **General parameters**  Relate to chosing which booster algorithm we will be using, usually tree or linear model.
# 
# 2. **Booster parameters**  Are the actual parameters of the booster you have chosen.
# 
# 3. **Task parameters**  Tells the framework what problem are we trying to solve. For example, regression tasks may use different parameters with ranking tasks.
# 
# ____
# 
# 
# <h3>How to tune XGBoost like a boss?</h3>
# 
# Hyperparameters tuning guide:
# 
# <h4>General Parameters</h4>
# 
# 1. **booster**  [default= gbtree ] 
#   - Which booster to use. Can be gbtree, gblinear or dart; 
#   - gbtree and dart use tree based models while gblinear uses linear functions.
# 
# 2. **verbosity** [default=1]
#   - Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). 
#   - Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. 
#   - If there’s unexpected behaviour, please try to increase value of verbosity.
# 
# 3. **nthread**  [default to maximum number of threads available if not set]
#   - Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.
# 
# ____
# 
# 
# <h4>Tree Booster Parameters</h4>
# 
# 1. **eta [default=0.3, ]**
#   - alias: learning_rate
#   - Step size shrinkage used in update to prevents overfitting.
#   - After each boosting step, we can directly get the weights of new features
#   - It makes the model more robust by shrinking the weights on each step.
#   - range: [0,1]
# 
# 2. **gamma [default=0]**
#   - Minimum loss reduction required to make a further partition on a leaf node of the tree. 
#   - The larger gamma is, the more conservative the algorithm will be.
#   - range: [0,∞]
# 
# 3. **max_depth [default=6]**
# 
#   - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
#   - 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. 
#   - Beware that XGBoost aggressively consumes memory when training a deep tree.
#   - range: [0,∞] )
# 
# 4. **min_child_weight [default=1]**
# 
#   - its Minimum sum of instance weight (hessian) needed in a child. I
#   - if the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. 
#   - In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. 
#   - The larger min_child_weight is, the more conservative the algorithm will be.
#   - range: [0,∞]
# 
# 5. **max_delta_step [default=0]**
# 
#   - Maximum delta step we allow each leaf output to be. 
#   - If the value is set to 0, it means there is no constraint. 
#   - If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
#   - range: [0,∞]
# 
# 6. **subsample [default=1]**
# 
#   - It denotes the fraction of observations to be randomly samples for each tree.
#   - Subsample ratio of the training instances.
#   - Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. - This will prevent overfitting.
#   - Subsampling will occur once in every boosting iteration.
#   - Lower values make the algorithm more conservative and prevents overfitting but too small alues might lead to under-fitting.
#   - typical values: 0.5-1
#   - range: (0,1]
# 
# 7. **sampling_method [default= uniform]**
# 
#   - The method to use to sample the training instances.
#   - **uniform:** each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
#   - **gradient_based:**  the selection probability for each training instance is proportional to the regularized absolute value of gradients 
# 
# 8. **colsample_bytree, colsample_bylevel, colsample_bynode [default=1]**
# 
# ><h4>This is a family of parameters for subsampling of columns.</h4>
# >
# >**All colsample_by**  parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.
# >
# >**lsample_bytree**s the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
# >
# >**colsample_bylevel**  is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
# >
# >**colsample_bynode**  is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
# >
# >**colsample_by**  parameters work cumulatively. For instance, the combination **{'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}** with 64 features will leave 8 features to choose from at each split.
# >
# 
# 9. **lambda [default=1]**
#   - alias: reg_lambda
#   - L2 regularization term on weights. 
#   - Increasing this value will make model more conservative.
# 
# 10. **alpha [default=0]**
#   - alias: reg_alpha
#   - L1 regularization term on weights.
#   - Increasing this value will make model more conservative.
# 
# 11. **grow_policy [default= depthwise]**
#   - Controls a way new nodes are added to the tree.
#   - Currently supported only if tree_method is set to hist or gpu_hist.
#   - **Choices:*  - depthwise, lossguide
#   - **depthwise:*  - split at nodes closest to the root.
#   - **lossguide:*  - split at nodes with highest loss change.
# 
# 12. **max_leaves [default=0]**
#   - Maximum number of nodes to be added. 
#   - Only relevant when grow_policy=lossguide is set.
# 
# [Read More](https://xgboost.readthedocs.io/en/latest/parameter.html)
# 
# ____
# 
# 
# 
# <h4>Task Parameters</h4>
# 
# 1. **objective [default=reg:squarederror]**
# 
# It defines the loss function to be minimized. Most commonly used values are given below -
# 
#   - reg:squarederror : regression with squared loss.
# 
#   - reg:squaredlogerror: regression with squared log loss 1/2[log(pred+1)−log(label+1)]2. - All input labels are required to be greater than -1.
# 
#   - reg:logistic : logistic regression
# 
#   - binary:logistic : logistic regression for binary classification, output probability
# 
#   - binary:logitraw: logistic regression for binary classification, output score before logistic transformation
# 
#   - binary:hinge : hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
# 
#   - multi:softmax : set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
# 
#   - multi:softprob : same as softmax, but output a vector of ndata nclass, which can be further reshaped to ndata nclass matrix. The result contains predicted probability of each data point belonging to each class.
# 
# 2. **eval_metric [default according to objective]**
#   - The metric to be used for validation data.
#   - The default values are rmse for regression, error for classification and mean average precision for ranking.
#   - We can add multiple evaluation metrics.
#   - Python users must pass the metrices as list of parameters pairs instead of map.
#   - The most common values are given below -
# 
#    - rmse : root mean square error
#    - mae : mean absolute error
#    - logloss : negative log-likelihood
#    - error : Binary classification error rate (0.5 threshold). 
#    - merror : Multiclass classification error rate.
#    - mlogloss : Multiclass logloss
#    - auc: Area under the curve
#    - aucpr : Area under the PR curve
#    

# # <span class="title-section w3-xxlarge" id="outline">Libraries 📚</span>
# <hr>

# #### Code starts here ⬇

# In[5]:


import traceback
import numpy as np
import pandas as pd
import datatable as dt
import gresearch_crypto
from lightgbm import LGBMRegressor

TRAIN_JAY = '../input/cryptocurrency-extra-data-binance-coin/orig_train.jay'
ASSET_DETAILS_JAY = '../input/cryptocurrency-extra-data-binance-coin/orig_asset_details.jay'


# In[6]:


df_train = dt.fread('../input/cryptocurrency-extra-data-binance-coin/orig_train.jay').to_pandas()
df_train.head()


# In[7]:


df_asset_details = dt.fread('../input/cryptocurrency-extra-data-binance-coin/orig_asset_details.jay').to_pandas().sort_values("Asset_ID")
df_asset_details


# # <span class="title-section w3-xxlarge" id="training">Training 🏋️</span>
# <hr>

# ## Utility functions to train a model for one asset

# **Feature Extraction**

# In[8]:


import xgboost as xgb

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat


# **Main Training Function**

# In[9]:


import xgboost as xgb
def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=11,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
    )
    model.fit(X, y)

    return X, y, model


# ## Loop over all assets

# In[ ]:


Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    try:
        X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
        Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model
    except:         
        Xs[asset_id], ys[asset_id], models[asset_id] = None, None, None    


# In[ ]:


# Check the model interface
x = get_features(df_train.iloc[1])
y_pred = models[0].predict(pd.DataFrame([x]))
y_pred[0]


# In[ ]:


env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        if models[row['Asset_ID']] is not None:
            try:
                model = models[row['Asset_ID']]
                x_test = get_features(row)
                y_pred = model.predict(pd.DataFrame([x_test]))[0]
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
            except:
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
                traceback.print_exc()
        else: 
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
        
    env.predict(df_pred)

