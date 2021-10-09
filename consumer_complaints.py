#!/usr/bin/env python
# coding: utf-8

import os
import random
import math
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (16, 14)
plt.style.use('ggplot')
np.set_printoptions(precision=3)
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None)
sns.set(font_scale=1.2)

warnings.filterwarnings('ignore')

import optuna
import lightgbm
import xgboost
import re

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.decomposition import PCA, LatentDirichletAllocation

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score

from tqdm import tqdm
from tqdm.keras import TqdmCallback

import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn import over_sampling
from imblearn import combine

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def get_top_k_frequent_percentage(col, k=10):
    return df[col].value_counts().head(k).sum()*100/df.shape[0]
def get_top_k_frequent_items(col, k=10):
    return set(df[col].value_counts().head(k).index.tolist())

result_dir_path = './results/'
if not os.path.isdir(result_dir_path):
    os.mkdir(result_dir_path)
feature_engineered_data_dir_path = './feature_engineered_data/'
if not os.path.isdir(feature_engineered_data_dir_path):
    os.mkdir(feature_engineered_data_dir_path)

feature_engineered_data_dir = os.listdir(feature_engineered_data_dir_path)

if len(feature_engineered_data_dir) == 0:

    print('x_train, x_test, y_train, y_test file creation start ............')
    consumer_complaints = pd.read_csv('consumer_complaints.csv')
    df = consumer_complaints.copy()
    rename_cols = {'Date received': 'date_received', 'Product':'product', 'Sub-product':'sub_product', 
                   'Sub-issue':'sub_issue', 'Consumer complaint narrative': 'complaint', 'Issue':'issue',
                   'Company public response': 'response_to_public', 'Company':'company', 'State':'state',
                  'ZIP code':'zip', 'Tags':'tags', 'Consumer consent provided?': 'consent_provided',
                  'Submitted via': 'submitted_via', 'Date sent to company':'date_sent',
                  'Company response to consumer':'response_to_consumer', 'Timely response?':'timely_response',
                  'Consumer disputed?':'consumer_disputed', 'Complaint ID':'complaint_id'}
    df.rename(columns=rename_cols, inplace=True)

    df['date_received'] = pd.to_datetime(df['date_received'], format="%m/%d/%Y")
    df['date_sent'] = pd.to_datetime(df['date_sent'], format="%m/%d/%Y")

    df.drop('complaint_id', axis=1, inplace=True)
    target = 'response_to_consumer'

    """ Fill Null Values """
    print('filling null values ................')
    s = df.isna().sum()*100/df.shape[0] > 0
    high_null_col = s[s].index.tolist()

    df['complaint'] = 1 - df['complaint'].isna().astype(int)
    df['response_to_public'], index_response_to_public = pd.factorize(df['response_to_public'])
    df['tags'], index_tags = pd.factorize(df['tags'])
    df['consent_provided'], index_consent_provided = pd.factorize(df['consent_provided'])

    # df.groupby('sub_issue')[target].value_counts().to_frame().head(300).head(50)
    ### checked to make sure that there is no particular relevance of 'sub_issue' to 'response_to_consumer'

    sub_issue_top = df['sub_issue'].value_counts().head(6).index.tolist()
    df['sub_issue'] = df['sub_issue'].apply(lambda x: 'others' if (not x is np.nan) and (x not in sub_issue_top) else x)
    df['sub_issue'], index_sub_issue = pd.factorize(df['sub_issue'])

    cols = ['state', 'zip', 'consumer_disputed']
    df.loc[:, cols] = df[cols].fillna(method='ffill') # used ffill since % na is less and data is a bit sequential

    col = 'sub_product'
    df_sub_prod_fill = df[~df['sub_product'].isna()].groupby(['product'])[col].apply(lambda x: x.mode().iloc[0]).to_frame().reset_index()

    df['sub_product'].fillna(value='credit', inplace=True)

    """ Categorical Encoding """
    print('categorical encoding is on progress ................')
    yes_no_dct = {'Yes':1, 'No':0}
    df['timely_response'] = df['timely_response'].apply(lambda x: yes_no_dct[x])
    df['consumer_disputed'] = df['consumer_disputed'].apply(lambda x: yes_no_dct[x])

    df['time_delta'] = (df['date_sent'] - df['date_received']).dt.days

    for col in ['date_received', 'date_sent']:
        df[col+'_day'] = df[col].dt.day
        df[col+'_month'] = df[col].dt.month
        df[col+'_quarter'] = df[col].dt.quarter
        df[col+'_year'] = df[col].dt.year

    """After rounding `zip` code to last 3 places, and filling null values using `groupby` and `median`,
    we check unique values of `zip` grouping by `state` and find that most of the zip codes are concentrated
    on 1-3 zip codes within a state, which is not accounting for much variance given `state` info.
    Even if we pick up top few `zip` codes and replace the rest by the median values using groupby on state,
    it will not add any info besides what we have from `state` info, but will create many more columns after
    one hot encoding. Hence, we will drop `zip` feature."""

    cols_to_drop = ['date_received', 'date_sent', 'zip']
    cat_cols = df.select_dtypes(['object', 'datetime64']).columns.tolist()
    cat_cols = list(set(cat_cols).difference(set(cols_to_drop)))

    freq_company = get_top_k_frequent_items('company', 12)
    df['company'] = df['company'].apply(lambda x: x if x in freq_company else 'Others')

    freq_issue = get_top_k_frequent_items('issue', 8)
    df['issue'] = df['issue'].apply(lambda x: x if x in freq_issue else 'others')

    print('label encoding target ........... ')
    df['response'] = df['response_to_consumer']
    label_encode_dct = {'Closed':3, 'Closed with explanation':0, 'Closed with monetary relief':2, 'Closed with non-monetary relief':1, 'Closed with relief':4, 'Closed without relief':5, 'In progress':6, 'Untimely response':7}
    df['response_to_consumer'] = df['response'].apply(lambda x: label_encode_dct[x])

    label_encode_dct_inv = {val:key for (key, val) in label_encode_dct.items()}
    keys = list(label_encode_dct_inv.keys())
    keys.sort()
    encoded_labels = [label_encode_dct_inv[key] for key in keys]
    np.save(result_dir_path+'encoded_labels.npy', encoded_labels)

    cols_to_drop = ['date_received', 'date_sent', 'zip', 'sub_product', 'response'] # , 'issue' , 'company', 'state']
    dfc = df.copy()
    dfc.drop(columns=cols_to_drop, inplace=True)

    dfc = pd.get_dummies(dfc)
    X = dfc.drop(columns='response_to_consumer')
    y = dfc['response_to_consumer']

    """ Divide dataset into train, test """
    print('dividing into train, test data ........... ')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    """ Target Encoding feature engineering """
    # if not os.path.exists('x_train.csv'):
    print('target encoding is taking place ............ ')
    cols = ['date_received_year', 'timely_response', 'sub_issue']
    df_train = pd.concat([x_train, y_train], axis=1)
    df_target_mean = df_train.groupby(cols)[target].apply(lambda x: x.mean()).to_frame().reset_index()
    df_target_mean.rename(columns={target : target+'_mean'}, inplace=True)
    df_target_count_percent = df_train.groupby(cols)[target].apply(lambda x: len(x)*100/df_train.shape[0]).to_frame().reset_index()
    df_target_count_percent.rename(columns={target : target+'_count_percent'}, inplace=True)

    x_train = pd.merge(x_train, df_target_mean, how='left', left_on=cols, right_on=cols)
    x_test = pd.merge(x_test, df_target_mean, how='left', left_on=cols, right_on=cols)
    x_train = pd.merge(x_train, df_target_count_percent, how='left', left_on=cols, right_on=cols)
    x_test = pd.merge(x_test, df_target_count_percent, how='left', left_on=cols, right_on=cols)
    print('x_train, x_test created ....')

    # if not os.path.exists('x_train.csv'):
    x_train.to_csv(feature_engineered_data_dir_path + 'x_train.csv')
    y_train.to_csv(feature_engineered_data_dir_path + 'y_train.csv')
    x_test.to_csv(feature_engineered_data_dir_path + 'x_test.csv')
    y_test.to_csv(feature_engineered_data_dir_path + 'y_test.csv')
    print('x_train, x_test written ....')
    for df in [x_train, x_test]:
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

else:
    print('x_train, x_test read ....')
    encoded_labels = np.load(result_dir_path+'encoded_labels.npy')
    x_train = pd.read_csv(feature_engineered_data_dir_path + 'x_train.csv', index_col=0)
    y_train = pd.read_csv(feature_engineered_data_dir_path + 'y_train.csv', index_col=0)
    x_test = pd.read_csv(feature_engineered_data_dir_path + 'x_test.csv', index_col=0)
    y_test = pd.read_csv(feature_engineered_data_dir_path + 'y_test.csv', index_col=0)
    for df in [x_train, x_test]:
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


""" Calculation of mutual information score """
# mi_score_path = result_dir_path + 'mi_scores.csv'
# if not os.path.isfile(mi_score_path):
#     print('calculating mutual information scores ......... ')
#     start_time = time.time()
#     mi_scores = mutual_info_classif(x_train, y_train)
#     print(f'time-taken to compute mi_scores: {time.time()-start_time : 0.4f} s')

#     mi_scores = pd.Series(data=mi_scores, index=x_train.columns.to_list())
#     mi_scores.sort_values(ascending=False, inplace=True)
#     mi_scores.to_csv(mi_score_path)
#     k = 10
#     bar_plot = sns.barplot(x=mi_scores.head(k).values, y=mi_scores.head(k).index)
#     bar_plot.get_figure().savefig('mi_scores.png')


""" Other Functions """
Input_shape = x_train.shape[1]

def get_ann_model():
    model = Sequential([
    Dense(128, activation='relu', input_shape=(Input_shape,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_value_counts(y):
    y_classes, y_counts = np.unique(y, return_counts=True)
    return {key:val for (key, val) in zip(y_classes, y_counts)}

def get_sns_heatmap(conf_mat, name, balance_algo, f1, encoded_labels=encoded_labels):
    heat_dir = './heat_plots/'
    if not os.path.isdir(heat_dir):
        os.mkdir(heat_dir)
    conf_mat = pd.DataFrame(conf_mat, index=encoded_labels, columns=encoded_labels)
    conf_mat_path = heat_dir+name+'_'+balance_algo+'.csv'
    conf_mat.to_csv(conf_mat_path)
    heat_plot = sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
    fig = heat_plot.get_figure()
    fig.suptitle(f'f1-macro: {f1: 0.3f}')
    fig_path = heat_dir+name+'_'+balance_algo+'.png'
    fig.savefig(fig_path)
    print(f'{fig_path} saved')
    return None

def model_name(model):
    return model.__class__.__name__

def get_balance_algo(get_func):
    if get_func == get_f1_score:
        return 'baseline'
    elif get_func == get_f1_confusion_mat_SMOTE:
        return 'SMOTE'
    elif get_func == get_f1_confusion_mat_custom_SMOTETomek:
        return 'custom_SMOTETomek'
    elif get_func == get_f1_confusion_mat_hyperparameter_class_wt:
        return 'class_weight'
    else:
        raise KeyError(f'{get_func} does not exist')

def model_f1_conf_mat(model, x_test=x_test, y_test=y_test):
    if model_name(model) != 'Sequential':
        y_test_pred = model.predict(x_test)
    else:
        y_test_pred = np.argmax(model.predict(x_test), axis=1)
    f1 = f1_score(y_test, y_test_pred, average='macro')
    conf_mat = confusion_matrix(y_test, y_test_pred)
    return f1, conf_mat


# ## create df_result to save results     

# models = [lightgbm.LGBMClassifier(), xgboost.XGBClassifier(eval_metric='mlogloss'), get_ann_model(), KNeighborsClassifier()]  #  SVC(),
models = [get_ann_model(), KNeighborsClassifier()]  #  SVC(),
index = [model_name(model) for model in models]
columns = ['baseline', 'SMOTE', 'custom_SMOTETomek', 'class_weight']

file_result_path = result_dir_path + 'df_result.csv'
if not os.path.exists(file_result_path):
    nrows = len(index)
    ncols = len(columns)
    data = [[np.nan]*ncols]*nrows
    df_result = pd.DataFrame(data=data, index=index, columns=columns)
    df_result.to_csv(file_result_path)
else:
    df_result = pd.read_csv(file_result_path, index_col=0)

file_train_time_path = result_dir_path + 'df_train_time.csv'
if not os.path.exists(file_train_time_path):
    nrows = len(index)
    ncols = len(columns)
    data = [[np.nan]*ncols]*nrows
    df_time = pd.DataFrame(data=data, index=index, columns=columns)
    df_time.to_csv(file_train_time_path)
else:
    df_time = pd.read_csv(file_train_time_path, index_col=0)


def get_write_result(model, get_func):
    name = model_name(model)
    balance_algo = get_balance_algo(get_func)
    if math.isnan(df_result.loc[name, balance_algo]):
        start_time = time.time()
        print(name)
        f1, conf_mat = get_func(model)
        train_time = (time.time()-start_time)/60
        print(f'train-time: {train_time : 0.2f} min')
        print('f1_macro:', f1)
        get_sns_heatmap(conf_mat, name, balance_algo, f1)
        df_result.loc[name, balance_algo] = f1
        df_result.to_csv(file_result_path)
        df_time.loc[name, balance_algo] = train_time
        df_time.to_csv(file_train_time_path)
    else:
        print(f'result calculated for model: {name} ---')
        print(f'f1 score: {df_result.loc[name, balance_algo]}')

""" 1. F1 score without using over-sampling SMOTE (baseline) """
def get_f1_score(model = lightgbm.LGBMClassifier(), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
    if model_name(model) != 'Sequential':
        model.fit(x_train, y_train)
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-5, patience=5)
        model.fit(x_train, y_train, validation_split=0.3, epochs=80, 
                  callbacks=[early_stopping, TqdmCallback(verbose=0)], verbose=0)
    return model_f1_conf_mat(model)

""" 2. F1 score using hyperparameter tuning inside algorithm to handle imbalance """
def get_f1_confusion_mat_hyperparameter_class_wt(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
    cls_wts = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    if model_name(model) != 'Sequential':
        model.fit(x_train, y_train, sample_weight=cls_wts)
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-5, patience=5)
        model.fit(x_train, y_train, class_weight=cls_wts, validation_split=0.3, epochs=80, callbacks=[early_stopping, TqdmCallback(verbose=0)], verbose=0)
    return model_f1_conf_mat(model)

""" 3. F1 score using over-sampling SMOTETomek """
def get_f1_confusion_mat_custom_SMOTETomek(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
    sampling_strategy = {3:100000, 0:395185, 2:100000, 1:100000, 4:50000, 5:100000, 6:50000, 7:50000}
    sm = combine.SMOTETomek(sampling_strategy=sampling_strategy, random_state=7)
    X_sm, y_sm = sm.fit_resample(x_train, y_train)
    if model_name(model) != 'Sequential':
        model.fit(X_sm, y_sm)
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-5, patience=5)
        model.fit(X_sm, y_sm, validation_split=0.3, epochs=80, 
                  callbacks=[early_stopping, TqdmCallback(verbose=0)], verbose=0)
    return model_f1_conf_mat(model)

""" 4. F1 score using over-sampling SMOTE """
def get_f1_confusion_mat_SMOTE(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
    smt = over_sampling.SMOTE(random_state=7)
    X_smt, y_smt = smt.fit_resample(x_train, y_train)
    if model_name(model) != 'Sequential':
        model.fit(X_smt, y_smt)
    else:
        early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-5, patience=5)
        model.fit(X_smt, y_smt, validation_split=0.3, epochs=80, 
                  callbacks=[early_stopping, TqdmCallback(verbose=0)], verbose=0)
    return model_f1_conf_mat(model)


""" First calcultaing all imbalance approaches for lightgbm model """

print('1. Calculating F1 score without using over-sampling SMOTE (baseline) ...............')
for model in models[:1]:
    get_write_result(model, get_f1_score)

print('2. F1 score using hyperparameter tuning inside algorithm to handle imbalance ................')
for model in models[:1]:
    get_write_result(model, get_f1_confusion_mat_hyperparameter_class_wt)

print('Calculating 3. F1 score using over-sampling SMOTETomek .................')
for model in models[:1]:
    get_write_result(model, get_f1_confusion_mat_custom_SMOTETomek)

print('4. Calculating F1 score using over-sampling SMOTE .................')
for model in models[:1]:
    get_write_result(model, get_f1_confusion_mat_SMOTE)


""" Calcultaing all imbalance approaches for rest models """

print('1. Calculating F1 score without using over-sampling SMOTE (baseline) ...............')
for model in models:
    get_write_result(model, get_f1_score)

print('2. F1 score using hyperparameter tuning inside algorithm to handle imbalance ................')
for model in models[:3]:
    get_write_result(model, get_f1_confusion_mat_hyperparameter_class_wt)

print('Calculating 3. F1 score using over-sampling SMOTETomek .................')
for model in models:
    get_write_result(model, get_f1_confusion_mat_custom_SMOTETomek)

print('4. Calculating F1 score using over-sampling SMOTE .................')
for model in models:
    get_write_result(model, get_f1_confusion_mat_SMOTE)
