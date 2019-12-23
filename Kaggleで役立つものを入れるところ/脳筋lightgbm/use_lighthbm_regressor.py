import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import KFold
from statistics import mean
import pandas as pd
import numpy as np


def load_data():
    train_df = pd.read_csv("house_datasets/train.csv")
    test_df = pd.read_csv("house_datasets/test.csv")
    return train_df, test_df


def preprocessing(train_df,test_df, id_name):
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    target = set(list(train_df.columns)) ^ set(list(test_df.columns))
    target_df = train_df[target]
    train_df = train_df.drop(target,axis=1)
    all_data = pd.concat([train_df,test_df])
    # objectカラムの取得
    change_columns =[]
    for index_name, dtype in zip(list(train_df.dtypes.index),list(train_df.dtypes.values)):
        if  str(dtype) =='object':
            change_columns.append(index_name)
    train_df[change_columns] = list(train_df[change_columns].fillna("欠損"))
    target_column = train_df[change_columns]
    for column in change_columns:
        le = LabelEncoder()
        select_df = all_data[column].astype(str)
        le.fit(select_df)
        all_data[column] = le.transform(select_df)
    all_data = all_data.drop(id_name,axis=1)
    train_df = all_data[all_data["is_train"]==1].drop("is_train",axis=1)
    test_df = all_data[all_data["is_train"]==0].drop("is_train",axis=1)
    return train_df, test_df, target_df

def objective(trial):
    # optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
    param = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    kfold = KFold(5, random_state = 0, shuffle = True)
    scores = []
    for tr_inds, val_inds in kfold.split(X):
        X_train, y_train = X.iloc[tr_inds],Y.iloc[tr_inds]
        X_eval, y_eval = X.iloc[val_inds],Y.iloc[val_inds]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        model = lgb.train(param, lgb_train, valid_sets=lgb_eval, verbose_eval=False)
        preds = model.predict(X_eval)
        pred_labels = np.rint(preds)
        score = mean_squared_error(y_eval, pred_labels)
        scores.append(score)
    return mean(scores)

def make_prediction(params):
    kfold = KFold(5, random_state = 0, shuffle = True)
    test_preds = []
    for tr_inds, val_inds in kfold.split(X):
        X_tr, y_tr = X.iloc[tr_inds],Y.iloc[tr_inds]
        X_eval, y_eval = X.iloc[val_inds],Y.iloc[val_inds]
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        model = lgb.train(params, lgb_train, valid_sets=lgb_eval, verbose_eval=False)
        preds = model.predict(Y_test)
        test_preds.append(preds)
    return list(np.sum(np.array(test_preds),axis=0))

def to_csv():
    submission = pd.DataFrame()
    submission[id_name] = test_df[id_name]
    submission[Y.columns[0]] = test_preds
    submission.to_csv("regression_submission.csv", index=False)

if __name__=='__main__':
    train_df, test_df = load_data()
    id_name = "Id"
    X, Y_test, Y = preprocessing(train_df, test_df, id_name)

    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    test_preds = make_prediction(study.best_params)
    print("val_score→"+str(study.best_value))
    to_csv()
