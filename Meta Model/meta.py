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
plt.rcParams['figure.figsize'] = 10, 10

np.random.seed(6)

input_dir = '/home/peeyman/All Files/Kaggle Competitions/Iceberg/Data'

train = pd.read_json('{0}/train.json'.format(input_dir))
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
X_angle=train['inc_angle'].reshape(1604,1)
target_train=train['is_iceberg']
train_ids = train['id']
test = pd.read_json('{0}/test.json'.format(input_dir))

#-----------------Read data-------------------------------
pred_dir = '/home/peyman/All files/Kaggle Competitions/IcebergGit/Preds'
train1 = pd.read_csv('{0}/valid.lenet.size.v1.csv'.format(pred_dir))
test1 = pd.read_csv('{0}/sub.lenet.size.v1.csv'.format(pred_dir))

train2 = pd.read_csv('{0}/valid.lenet.size.v1.csv'.format(pred_dir))
test2 = pd.read_csv('{0}/sub.lenet.size.v1.csv'.format(pred_dir))

train3 = pd.read_csv('{0}/valid.mobilenet.v1.csv'.format(pred_dir))
test3 = pd.read_csv('{0}/sub.mobilenet.v1.csv'.format(pred_dir))

train4 = pd.read_csv('{0}/valid.peyman.cnn.size.v1.csv'.format(pred_dir))
test4 = pd.read_csv('{0}/sub.peyman.cnn.size.v1.csv'.format(pred_dir))

train5 = pd.read_csv('{0}/valid.peyman.chia.v1.csv'.format(pred_dir))
test5 = pd.read_csv('{0}/sub.peyman.chia.v1.csv'.format(pred_dir))

train6 = pd.read_csv('{0}/valid.lightgbm.size.v1.csvs'.format(pred_dir))
test6 = pd.read_csv('{0}/sub.lightgbm.size.v1.csv'.format(pred_dir))

train7 = pd.read_csv('{0}/valid.resnet50.size.v1.csv'.format(pred_dir))
test7 = pd.read_csv('{0}/sub.resnet50.size.v1.csv'.format(pred_dir))

train8 = pd.read_csv('{0}/valid.VGG16.v1.csv'.format(pred_dir))
test8 = pd.read_csv('{0}/sub.VGG16.v1.csv'.format(pred_dir))

train9 = pd.read_csv('{0}/valid.inception.size.v1.csv'.format(pred_dir))
test9 = pd.read_csv('{0}/sub.inception.size.v1.csv'.format(pred_dir))


X_train = pd.concat((train1.iloc[:,1],train2.iloc[:,1],train3.iloc[:,1], \
                     train4.iloc[:, 1],train5.iloc[:, 1],train6.iloc[:, 1],
                     train7.iloc[:, 1],train8.iloc[:, 1],train9.iloc[:,1]),axis=1)


X_test = pd.concat((test1.iloc[:,1],test2.iloc[:,1],test3.iloc[:,1], \
                    test4.iloc[:, 1],test5.iloc[:, 1],test6.iloc[:, 1], \
                    test7.iloc[:, 1],test8.iloc[:, 1],test9.iloc[:,1]),axis=1)

# X_train = X_train.iloc[:,:8]
X_train = np.array(X_train)
#-----------------Get hog features------------------------
params = {}
params['learning_rate'] = 0.005
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = .6
params['num_leaves'] = 20
params['min_data'] = 1
params['min_hessian'] = 1e-3
params['feature_fraction_seed'] = 1#np.random.randint(0, 10000)
params['lambda_l1'] = 0
params['bagging_fraction'] = .3
params['bagging_freq'] = 1
# params['bagging_seed'] = 1


def myModelCV(X_train):
    clf_list = []
    K = 20
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=20).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    fold_cv = []
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        # y_train_cv = y_train_cv.reshape(len(y_train_cv),1)
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]
        # Y_holdout = Y_holdout.reshape(len(Y_holdout),1)

        d_train = lgb.Dataset(X_train_cv, label=y_train_cv, free_raw_data=True)
        d_valid = lgb.Dataset(X_holdout, Y_holdout, free_raw_data=False)
        pred_valid = np.zeros(len(X_holdout))
        for j in range(1):
            # params['sub_feature'] = .3*np.random.rand()+.7
            params['seed']=j
            clf = lgb.train(params, d_train, 15000, valid_sets=d_valid,\
                            early_stopping_rounds=40)
            pred_valid += clf.predict(X_holdout,num_iteration=clf.best_iteration)
        pred_valid /=1
        clf_list.append(clf)
        # pred_valid = np.clip(pred_valid,.001,.999)
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        y_test_pred_log += clf.predict(X_test)
        # print(accuracy_score(Y_holdout,pred_valid))
        fold_cv.append(log_loss(Y_holdout, pred_valid))
        print(log_loss(Y_holdout, pred_valid))
        # raw_input('press key')

    y_test_pred_log = y_test_pred_log / K
    return clf_list,fold_cv,y_valid_pred_log,y_test_pred_log


clf_list,fold_cv,valid_preds,test_preds = myModelCV(X_train)
print(fold_cv)
print(np.mean(fold_cv))


valid_pred_df = pd.DataFrame()
valid_pred_df['id'] = train['id']
valid_pred_df['is_iceberg'] = valid_preds
#valid_pred_df.to_csv('valid.meta.0.1276.csv',index=False)
# #Submission for each day.
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_preds
submission.to_csv('sub.meta', index=False)
