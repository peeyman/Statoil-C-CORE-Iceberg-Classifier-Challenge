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
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
from itertools import combinations
from math import exp, expm1, log1p, log10, sqrt, ceil, floor, isnan
import gc
plt.rcParams['figure.figsize'] = 10, 10

#---------------Functions-----------------
def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+1*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])
def size(arr):
    return float(np.sum(arr<-5))#/(75*75)

def read_jason(file='', loc='/home/peeyman/All Files/Kaggle Competitions/Iceburg/Iceberg/Data'):
    df = pd.read_json('{}{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    # print(df['inc_angle'].value_counts())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2, 0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands

# Stats are borrowed from the following kernel
# https://www.kaggle.com/cttsai/ensembling-gbms-lb-203


def img_to_stats(paths):
    img_id, img = paths[0], paths[1]

    # ignored error
    np.seterr(divide='ignore', invalid='ignore')

    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    # opt_poly = False

    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])

            # median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])]
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]),
                       (sub_st[-1] / sub_st[1])]  # normalized by stdev
            st += sub_st
            # Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()]  # blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]

            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]

                # hist
            # hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))]  # only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                st += [hist_diff.index(max(hist_diff))]  # only the smallest index w/ max value would be incl
                st += [np.std(hist_diff), np.max(hist_diff), np.median(hist_diff),
                       (np.max(hist_diff) - np.median(hist_diff))]

        # correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan

    except:
        print('except: ')

    return [img_id, st]


def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8)  # (cpu_count())
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)


def process(df, bands):
    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]);
    gc.collect()
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1);
    gc.collect()

    print(data.shape)
    return data


input_dir = '/home/peeyman/All Files/Kaggle Competitions/Iceburg/Iceberg/Data/'
# Load data
train = pd.read_json('{0}/train.json'.format(input_dir))
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
target_train=train['is_iceberg']
train_ids = train['id']

test = pd.read_json('{0}/test.json'.format(input_dir))
test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
X_test_angle=test['inc_angle'].reshape(len(test),1)

train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)
test['iso1'] = test.iloc[:, 0].apply(iso)
test['iso2'] = test.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
train['s1'] = train.iloc[:,5].apply(size)
train['s2'] = train.iloc[:,6].apply(size)
test['s1'] = test.iloc[:,5].apply(size)
test['s2'] = test.iloc[:,6].apply(size)

X_train_size = np.array(train['s1'])#+train['s2'])
X_train_size = np.reshape(X_train_size,(len(X_train_size),1))
X_test_size = np.array(test['s1'])#+test['s2'])
X_test_size = np.reshape(X_test_size,(len(X_test_size),1))
#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_2"]])
X_train_angle = np.array(train.inc_angle).reshape(1604,1)
X_total_1 = np.array([np.array(band).astype(np.float32).reshape(75*75) for band in train["band_1"]])
X_total_2 = np.array([np.array(band).astype(np.float32).reshape(75*75) for band in train["band_2"]])
X_train_total = np.concatenate((X_total_1,X_total_2),axis=1)


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_2"]])
X_total_1 = np.array([np.array(band).astype(np.float32).reshape(75*75) for band in test["band_1"]])
X_total_2 = np.array([np.array(band).astype(np.float32).reshape(75*75) for band in test["band_2"]])
X_test_total = np.concatenate((X_total_1,X_total_2),axis=1)


train_num = X_band_1.shape[0]
test_num = X_band_test_1.shape[0]
#-----------------Get hog features------------------------
params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 1
params['num_leaves'] = 12
params['min_data'] = 10
params['min_hessian'] = 1e-3
params['feature_fraction_seed'] = 1#np.random.randint(0, 10000)
# params['lambda_l1'] = .1
# params['bagging_fraction'] = .9
# params['bagging_freq'] = 10
# params['bagging_seed'] = 1

ppc = 16
hog_images = []
hog_features_band1 = []
hog_features_band2 = []
for i in range(test_num):
    print(i)
    image = X_band_test_1[i,:,:]
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features_band1.append(fd)

    image = X_band_test_2[i,:,:]
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features_band2.append(fd)

hog_features_band1 = np.array(hog_features_band1)
hog_features_band2 = np.array(hog_features_band2)
X_test = np.concatenate((hog_features_band1,hog_features_band2,X_test_angle),axis=1)
# X_train = np.concatenate((hog_features_band1,hog_features_band2),axis=1)

hog_images = []
hog_features_band1 = []
hog_features_band2 = []
for i in range(train_num):
    print(i)
    image = X_band_1[i,:,:]
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features_band1.append(fd)

    image = X_band_2[i,:,:]
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_images.append(hog_image)
    hog_features_band2.append(fd)

hog_features_band1 = np.array(hog_features_band1)
hog_features_band2 = np.array(hog_features_band2)
X_train = np.concatenate((hog_features_band1,hog_features_band2,X_train_angle),axis=1)

# joblib.dump([X_train,X_test],'hog_feats')
# [X_train,X_test] = joblib.load('hog_feats')
X_train = np.concatenate((X_train,X_train_size),axis=1)
X_test = np.concatenate((X_test,X_test_size),axis=1)

from sklearn.decomposition import PCA,KernelPCA
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# pca = PCA(n_components=100)
# X_phase = pca.fit_transform(X_phase)

pca = PCA(n_components=15)
X_total_pca = pca.fit_transform(X_train_total)
X_test_total_pca = pca.transform(X_test_total)

kpca = KernelPCA(kernel='poly',n_components=2,degree=3)
X_kernel_total = kpca.fit_transform(X_train_total)
X_test_kernel_total = kpca.transform(X_test_total)

kpca = KernelPCA(kernel='poly',n_components=2,degree=3)
X_kernel_pca1 = kpca.fit_transform(X_train)
X_test_kernel_pca1 = kpca.transform(X_test)
#
# kpca = KernelPCA(kernel='sigmoid',n_components=5)
# X_kernel_pca2 = kpca.fit_transform(X_train)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()

[X_chia_train,X_chia_test] = joblib.load('Chia.feats')

X_train = np.concatenate((X_train_angle,X_pca,X_kernel_pca1,X_total_pca,\
                          X_kernel_total,X_chia_train),axis=1)
X_test = np.concatenate((X_test_angle,X_test_pca,X_test_kernel_pca1,X_test_total_pca,\
                         X_test_kernel_total,X_chia_test),axis=1)


def myAngleCV(X_train):
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
        # y_train_cv = y_train_cv.reshape(len(y_train_cv),1)
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]
        # Y_holdout = Y_holdout.reshape(len(Y_holdout),1)

        d_train = lgb.Dataset(X_train_cv, label=y_train_cv, free_raw_data=True)
        d_valid = lgb.Dataset(X_holdout, Y_holdout, free_raw_data=False)
        pred_valid = np.zeros(len(X_holdout))
        for j in range(1):
            params['sub_feature'] = .3*np.random.rand()+.7
            clf = lgb.train(params, d_train, 15000, valid_sets=d_valid,\
                            early_stopping_rounds=20)
            pred_valid += clf.predict(X_holdout)
        pred_valid /=1
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        y_test_pred_log += clf.predict(X_test)
        # print(accuracy_score(Y_holdout,pred_valid))
        fold_cv.append(log_loss(Y_holdout, pred_valid))
        print(log_loss(Y_holdout, pred_valid))
        # raw_input('press key')

    y_test_pred_log = y_test_pred_log / K
    return fold_cv,y_valid_pred_log,y_test_pred_log

fold_cv,valid_preds,test_preds = myAngleCV(X_train)
print(fold_cv)
print(np.mean(fold_cv))
#
valid_pred_df = pd.DataFrame()
valid_pred_df['id'] = train['id']
valid_pred_df['is_iceberg'] = valid_preds
valid_pred_df.to_csv('../Preds/valid.peyman.chia.v1.csv',index=False)
#Submission for each day.
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_preds
submission.to_csv('../Preds/sub.peyman.chia.v1.csv', index=False)
