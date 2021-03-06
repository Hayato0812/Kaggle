import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import KFold
from statistics import mean
import pandas as pd
import numpy as np


def load_data():
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
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
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
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
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[[X_eval, y_eval]],verbose=0)
        preds = model.predict(X_eval)
        pred_labels = np.rint(preds)
        score = accuracy_score(y_eval, pred_labels)
        scores.append(score)
    return mean(scores)

def make_prediction(params):
    kfold = KFold(5, random_state = 0, shuffle = True)
    test_preds = []
    feature_importance_df = pd.DataFrame()
    feats = X.columns
    n_fold = 1
    for tr_inds, val_inds in kfold.split(X):
        X_train, y_train = X.iloc[tr_inds],Y.iloc[tr_inds]
        X_eval, y_eval = X.iloc[val_inds],Y.iloc[val_inds]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[[X_eval, y_eval]])
        preds = model.predict(Y_test)
        test_preds.append(preds)
    feature_importance_df["feature"] = feats
    feature_importance_df["importance"] = model.feature_importances_
    return list(np.sum(np.array(test_preds),axis=0)),feature_importance_df

def to_csv():
    submission = pd.DataFrame()
    submission[id_name] = test_df[id_name]
    submission[Y.columns[0]] = test_preds
    submission.to_csv("classify_submission.csv", index=False)

if __name__=='__main__':
    train_df, test_df = load_data()
    id_name = "PassengerId"
    X, Y_test, Y = preprocessing(train_df, test_df, id_name)

    study = optuna.create_study()
    study.optimize(objective, n_trials=2)
    test_preds,feature_importance_df = make_prediction(study.best_params)
    print("val_score→"+str(study.best_value))
    to_csv()
