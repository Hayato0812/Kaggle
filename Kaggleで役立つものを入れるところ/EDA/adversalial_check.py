

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gc

params = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'random_state': 47}

def load_data():
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    return train_df, test_df

def covariate_shift(feature):
    try:
        df_card1_train = pd.DataFrame(data={feature: train_df[feature], 'is_train': 1})
        df_card1_test = pd.DataFrame(data={feature: train_df[feature], "is_train": 0})

        # Creating a single dataframe
        df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)

        # Encoding if feature is categorical
        if str(df[feature].dtype) in ['object', 'category']:
            df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))

        # Splitting it to a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(df[feature], df['is_train'], test_size=0.33, random_state=47, stratify=df['is_train'])

        clf = lgb.LGBMClassifier(**params, num_boost_round=500)
        clf.fit(X_train.values.reshape(-1, 1), y_train)
        roc_auc =  roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])
    except ValueError:
        roc_auc = 0
    del df, X_train, y_train, X_test, y_test
    gc.collect();

    return roc_auc

if __name__=='__main__':
    train_df, test_df = load_data()
    train_df = train_df.drop("",axis=1)
    for name in list(train_df.columns):
        print(name,covariate_shift(name))
