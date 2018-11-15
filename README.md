# A stacking approach based on Convnets and boosting methods for Statoil-C-CORE-Iceberg-Classifier-Challenge

<p align="justify">This project is a solution for the Statoil/C-CORE Iceberg Classifier Challenge (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge). Stacking is a form of ensemble learning that combine multiple classification or regression models via a meta-classifier or a meta-regressor. In the first layer, models which are called base models consume the original features as input, while meta model consumes the predictions of the base models as its inputs (https://blog.statsbot.co/ensemble-learning-d1dcd548e936). 

<p align="justify">For the base models different models including VGG16, Resnet50, Mobilenet, some self-defined CNNs, and some tree boosting models are used. The meta model is a lightGBM model. For each base model 5-fold cross validation is used.

## Data
<p align="justify">Put train.jason and test.jason files into "Input" directory. (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)

## Requirements

- keras 2.2.2 with Tensorflow backend
- sklearn
- numpy
- pandas
- cv2

## How to run

1. Train base models
2. Train meta model

### 1. Train base models
<p align="justify">From the "Base Models" directory run each model separately. Doing so, for each model two files will be written in the "Preds" directory. One file is the prediction of a trained model on train data the other will be the prediction on test data. So, if you train 5 base models you will obtain 10 files in the "Preds" directory.

### 2. Train meta model
From the "Meta Model" directory run meta.py. 
