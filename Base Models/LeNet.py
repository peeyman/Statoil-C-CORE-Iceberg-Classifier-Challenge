import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.metrics import log_loss
plt.rcParams['figure.figsize'] = 10, 10

#---------------Functions-----------------
def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])
def size(arr):
    return float(np.sum(arr<-5))#/(75*75)

input_dir = '/home/peyman/All files/Kaggle Competitions/Iceberg/Data'


train = pd.read_json('{0}/train.json'.format(input_dir))
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
target_train=train['is_iceberg']
train_ids = train['id']
test = pd.read_json('{0}/test.json'.format(input_dir))


train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)
test['iso1'] = test.iloc[:, 0].apply(iso)
test['iso2'] = test.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
train['s1'] = train.iloc[:,5].apply(size)
train['s2'] = train.iloc[:,6].apply(size)
test['s1'] = test.iloc[:,5].apply(size)
test['s2'] = test.iloc[:,6].apply(size)
#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
X_train_angle = np.array(train.inc_angle)
X_train_size = np.array(train['s1'])

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3=(X_band_test_1+X_band_test_2)/2
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)
X_test_angle=test['inc_angle']
X_test_size = np.array(test['s1'])

# Import Keras.
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model,load_model
from keras import initializers,regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
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
                         rotation_range=20)


# Here is the function that merges the three generators (Images, angles,and size)
# The exact same generator with the same random seed was used
def gen_flow_for_two_inputs(X1, X2,X3, y):
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)
    genX3 = gen.flow(X1, X3, batch_size=batch_size, seed=55)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[1],X3i[1]], X1i[1]


# Finally create generator
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def getVggAngleModel():
    #Building the model
    pic_input = Input(shape=(75, 75, 3))
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    input_3 = Input(shape=[1], name="size")
    size_layer = Dense(1, )(input_3)

    cnn = BatchNormalization()(pic_input)
    #Conv Layer 1
    cnn = Conv2D(64, kernel_size=(3, 3),activation='relu',\
                 kernel_regularizer=regularizers.l2(.2))(cnn)
    cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn)
    # cnn = Dropout(0.5)(cnn)

    #Conv Layer 2
    cnn = Conv2D(128, kernel_size=(3, 3), activation='relu' )(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)

    #Conv Layer 3
    cnn = Conv2D(128, kernel_size=(3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)

    #Conv Layer 4
    cnn = Conv2D(64, kernel_size=(3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)

    #Flatten the data for upcoming dense layers
    cnn = Flatten()(cnn)
    cnn = concatenate([cnn, angle_layer, size_layer])
    #Dense Layers
    cnn = Dense(512)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(0.2)(cnn)

    #Dense Layer 2
    cnn = Dense(256)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(0.2)(cnn)

    #Sigmoid Layer
    cnn = Dense(1)(cnn)
    cnn = Activation('sigmoid')(cnn)
    gmodel = Model(inputs=[pic_input,input_2,input_3],outputs=cnn)

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel

def myAngleCV(X_train, X_train_angle,X_train_size, X_test):
    K = 5
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=20).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    fold_cv = []
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        # Angle
        X_angle_cv = X_train_angle[train_idx]
        X_angle_hold = X_train_angle[test_idx]

        #Size
        X_size_cv = X_train_size[train_idx]
        X_size_hold = X_train_size[test_idx]

        # define file path and get callbacks
        file_path = "%s_aug_model_weights.hdf5" % j
        callbacks = get_callbacks(filepath=file_path, patience=10)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv,X_size_cv, y_train_cv)
        pModel = getVggAngleModel()
        pModel.fit_generator(
            gen_flow,
            steps_per_epoch=24,
            epochs=100,
            shuffle=True,
            verbose=1,
            validation_data=([X_holdout, X_angle_hold,X_size_hold], Y_holdout),
            callbacks=callbacks)

        # Getting the Best Model
        pModel.load_weights(filepath=file_path)
        # Getting Training Score
        score = pModel.evaluate([X_train_cv, X_angle_cv,X_size_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        # Getting Test Score
        score = pModel.evaluate([X_holdout, X_angle_hold,X_size_hold], Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        fold_cv.append(score[0])
        # Getting validation Score.
        pred_valid = pModel.predict([X_holdout, X_angle_hold,X_size_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        # Getting Test Scores
        temp_test = pModel.predict([X_test, X_test_angle,X_test_size])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])

        # Getting Train Scores
        temp_train = v.predict([X_train, X_train_angle,X_train_size])
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])

    y_test_pred_log = y_test_pred_log / K
    y_train_pred_log = y_train_pred_log / K
    print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
    print(' Valid Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    return y_train_pred_log,y_valid_pred_log,y_test_pred_log,fold_cv

train_preds,valid_preds,test_preds,fold_cv = \
    myAngleCV(X_train,X_train_angle,X_train_size,X_test)

valid_pred_df = pd.DataFrame()
valid_pred_df['id'] = train['id']
valid_pred_df['is_iceberg'] = valid_preds
valid_pred_df.to_csv('../Preds/valid.lenet.size.v1.csv',index=False)
#Submission for each day.
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_preds
submission.to_csv('../Preds/sub.lenet.size.v1.csv', index=False)

print(fold_cv)
