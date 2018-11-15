import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
import keras

# from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_dir = '/home/peyman/All files/Kaggle Competitions/Iceberg/Data'

train = pd.read_json('{0}/train.json'.format(input_dir))
test = pd.read_json('{0}/test.json'.format(input_dir))

target_train = train['is_iceberg']
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')  # We have only 133 NAs.
train['inc_angle'] = train['inc_angle'].fillna(method='pad')
X_angle = train['inc_angle']
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
X_test_angle = test['inc_angle']

# Generate the training data
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
# X_band_3=(X_band_1+X_band_2)/2
X_band_3 = np.fabs(np.subtract(X_band_1, X_band_2))
X_band_4 = np.maximum(X_band_1, X_band_2)
X_band_5 = np.minimum(X_band_1, X_band_2)
# X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
X_train = np.concatenate([X_band_3[:, :, :, np.newaxis], X_band_4[:, :, :, np.newaxis], X_band_5[:, :, :, np.newaxis]],
                         axis=-1)

X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
# X_band_test_3=(X_band_test_1+X_band_test_2)/2
X_band_test_3 = np.fabs(np.subtract(X_band_test_1, X_band_test_2))
X_band_test_4 = np.maximum(X_band_test_1, X_band_test_2)
X_band_test_5 = np.minimum(X_band_test_1, X_band_test_2)
# X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
X_test = np.concatenate(
    [X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis], X_band_test_5[:, :, :, np.newaxis]],
    axis=-1)

# Import Keras.
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

# Data Aug for multi-input
from keras.preprocessing.image import ImageDataGenerator

batch_size = 64
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.,
                         height_shift_range=0.,
                         channel_shift_range=0,
                         zoom_range=0.5,
                         rotation_range=10)


# Here is the function that merges the two generators (images and angles)
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]


# Finally create generator
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def getVggAngleModel():
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)

    merge_one = Dropout(0.6)(x)
    predictions = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(merge_one)

    model = Model(input=[base_model.input, input_2], output=predictions)

    sgd = Adam(lr=1e-4)  # SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


# Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test):
    K = 5
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=20).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        # Angle
        X_angle_cv = X_angle[train_idx]
        X_angle_hold = X_angle[test_idx]

        # define file path and get callbacks
        file_path = "%s_aug_model_weights.hdf5" % j
        callbacks = get_callbacks(filepath=file_path, patience=10)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        pModel = getVggAngleModel()
        pModel.fit_generator(
            gen_flow,
            steps_per_epoch=24,
            epochs=1,
            shuffle=True,
            verbose=1,
            validation_data=([X_holdout, X_angle_hold], Y_holdout),
            callbacks=callbacks)

        # Getting the Best Model
        pModel.load_weights(filepath=file_path)
        # Getting Training Score
        score = pModel.evaluate([X_train_cv, X_angle_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        # Getting Test Score
        score = pModel.evaluate([X_holdout, X_angle_hold], Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Getting validation Score.
        pred_valid = pModel.predict([X_holdout, X_angle_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        # Getting Test Scores
        temp_test = pModel.predict([X_test, X_test_angle])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])


    y_test_pred_log = y_test_pred_log / K


    # print('\n Train Log Loss Validation= ', log_loss(target_train, y_train_pred_log))
    print(' Test Log Loss Validation= ', log_loss(target_train, y_valid_pred_log))
    return y_valid_pred_log,y_test_pred_log


valid_preds,test_preds = myAngleCV(X_train, X_angle, X_test)

valid_pred_df = pd.DataFrame()
valid_pred_df['id'] = train['id']
valid_pred_df['is_iceberg'] = valid_preds
valid_pred_df.to_csv('../Preds/valid.VGG16.v1.csv',index=False)

submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = test_preds
submission.to_csv('../Preds/sub.VGG16.v1.csv', index=False)