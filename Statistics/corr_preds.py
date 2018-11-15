import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.metrics import log_loss,accuracy_score
from skimage.feature import hog
from skimage import io
import lightgbm as lgb
import seaborn as sns
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


plt.rcParams['figure.figsize'] = 10, 10

np.random.seed(6)

input_dir = '/home/peyman/All files/Kaggle Competitions/Iceberg/Data'

temp = '{0}/train.json'.format(input_dir)
train = pd.read_json('{0}/train.json'.format(input_dir))
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
X_angle=np.array(train['inc_angle']).reshape((1604,1))
target_train=train['is_iceberg']
train_ids = train['id']
test = pd.read_json('{0}/test.json'.format(input_dir))
# X_test_angle=test['inc_angle'].reshape(len(test),1)
#Generate the training data
#Create 3 bands having HH, HV and avg of both
# X_band_1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_1"]])
# X_band_2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_2"]])
# X_angle_train = np.array(train.inc_angle).reshape(1604,1)

# X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_1"]])
# X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_2"]])

# data_num = X_band_1.shape[0]

#-----------------Read data-------------------------------
pred_dir = '/home/peyman/All files/Kaggle Competitions/Iceberg/Last week/Cross Validation/pred_dir'
train1 = pd.read_csv('{0}/valid.Jirka.v2.csv'.format(pred_dir))
test1 = pd.read_csv('{0}/sub.Jirka.v2.csv'.format(pred_dir))

train2 = pd.read_csv('{0}/valid.2018.vgg16.size.v2.csv'.format(pred_dir))
test2 = pd.read_csv('{0}/sub.2018.vgg16.size.v2.csv'.format(pred_dir))

train3 = pd.read_csv('{0}/valid.vgg16.hog.v1'.format(pred_dir))
test3 = pd.read_csv('{0}/sub.vgg16.hog.v1.csv'.format(pred_dir))

train4 = pd.read_csv('{0}/valid.2018.vgg16.v1.csv'.format(pred_dir))
test4 = pd.read_csv('{0}/sub.2018.vgg16.v1.csv'.format(pred_dir))

train5 = pd.read_csv('{0}/valid.lightgbm.v1.csv'.format(pred_dir))
test5 = pd.read_csv('{0}/sub.lightgbm.v1.csv'.format(pred_dir))

train6 = pd.read_csv('{0}/valid.ensemble.v1.csv'.format(pred_dir))
test6 = pd.read_csv('{0}/subm_blend009_2018-01-18-17-20.csv'.format(pred_dir))

train7 = pd.read_csv('{0}/valid.mobilenet.v1.csv'.format(pred_dir))
test7 = pd.read_csv('{0}/sub.mobilenet.v1.csv'.format(pred_dir))

train8 = pd.read_csv('{0}/valid.falk.resnet.v1.csv'.format(pred_dir))
test8 = pd.read_csv('{0}/sub.falk.resnet.v1.csv'.format(pred_dir))

train9 = pd.read_csv('{0}/valid.lightgbm.size.v1'.format(pred_dir))
test9 = pd.read_csv('{0}/sub.lightgbm.size.v1.csv'.format(pred_dir))

train10 = pd.read_csv('{0}/valid.peyman.chia.v1'.format(pred_dir))
test10 = pd.read_csv('{0}/sub.peyman.chia.v1.csv'.format(pred_dir))

train11 = pd.read_csv('{0}/valid.vgg19.size.v1'.format(pred_dir))
test11 = pd.read_csv('{0}/sub.vgg19.size.v1.csv'.format(pred_dir))

train12 = pd.read_csv('{0}/valid.peyman.cnn.size.v1'.format(pred_dir))
test12 = pd.read_csv('{0}/sub.peyman.cnn.size.v1.csv'.format(pred_dir))

train13 = pd.read_csv('{0}/valid.lenet.size.v1.csv'.format(pred_dir))
test13 = pd.read_csv('{0}/sub.lenet.size.v2.csv'.format(pred_dir))

# train3.iloc[:,1],train10.iloc[:, 1],
X_train = pd.concat((train1.iloc[:,1],train2.iloc[:,1], \
                     train3.iloc[:,1],
                     train4.iloc[:, 1],train5.iloc[:, 1],train6.iloc[:, 1],
                     train7.iloc[:, 1],train8.iloc[:, 1], \
                     train9.iloc[:, 1], train10.iloc[:,1],\
                     train11.iloc[:, 1],train12.iloc[:, 1], \
                     train13.iloc[:, 1]),axis=1)
# test3.iloc[:,1],test10.iloc[:, 1],
X_test = pd.concat((test1.iloc[:,1],test2.iloc[:,1], \
                    test4.iloc[:, 1],test5.iloc[:, 1],test6.iloc[:, 1], \
                    test7.iloc[:, 1],test8.iloc[:, 1], \
                    test9.iloc[:, 1], \
                    test11.iloc[:, 1],train12.iloc[:, 1], \
                    train13.iloc[:, 1]),axis=1)

# X_train = X_train.iloc[:,:8]
# X_train = np.array(X_train)

X_train.columns =['Jirka','vgg16_size','vgg16_hog','vgg16','lightgbm','selfDesignCNN',\
                        'mobilenet','resnet50','lightgbm_size','lightgbm_stats',\
                        'vgg19','selfDesignCNN_size','lenet']

order_cols = ['vgg16','vgg16_size','vgg16_hog','selfDesignCNN','selfDesignCNN_size','vgg19',\
              'resnet50','mobilenet','lenet','Jirka','lightgbm','lightgbm_size','lightgbm_stats']

X_train = X_train[order_cols]

corr = X_train.corr()
print(corr.round(2))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(corr,annot=True)

plt.show()
temp = 1

