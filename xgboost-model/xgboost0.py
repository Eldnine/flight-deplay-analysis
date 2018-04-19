import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import confusion_matrix


def main():
    x_csv_file = 'flights_svm.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)

    df_x = pd.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_x['ARRIVAL_DELAY']
    df_x = df_x.drop(['ARRIVAL_DELAY'], axis=1)

    X = np.array(df_x)
    Y = np.array(df_y).ravel()

    msk = np.random.rand(len(X)) < 0.8
    train_x = X[msk]
    train_y = Y[msk]
    test_x = X[~msk]
    test_y = Y[~msk]

    model = xgboost.XGBClassifier(
        max_depth=1000,
        objective='binary:logistic',
        scale_pos_weight=8)
    model.fit(train_x, train_y.ravel())
    print(model)
    predict = model.predict(test_x)
    print(confusion_matrix(test_y, predict.round()))


if __name__ == '__main__':
    main()
