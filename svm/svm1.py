import pandas
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold


def main():
    x_csv_file = 'flights_svm.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)

    df_x = pandas.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_x['ARRIVAL_DELAY']
    df_x = df_x.drop(['ARRIVAL_DELAY'], axis=1)

    print df_x.count

    X = np.array(df_x)
    Y = np.array(df_y).ravel()

    msk = np.random.rand(len(X)) < 0.8

    train_x = X[msk]
    train_y = Y[msk]

    test_x = X[~msk]
    test_y = Y[~msk]
    #
    # clf = svm.SVC()
    # clf.fit(train_x, train_y.ravel())
    # predict = clf.predict(test_x)
    #
    # print(accuracy_score(test_y.ravel(), predict))  # 0.901657458564
    # print(confusion_matrix(test_y.ravel(), predict))
    # # [[1602    3]
    # #  [175   30]]

    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(train_x, train_y.ravel())
    # predict = clf.predict(test_x)
    #
    # print(accuracy_score(test_y.ravel(), predict))  # 0.89218328841
    # print(confusion_matrix(test_y.ravel(), predict))
    # # [[1624    1]
    # #  [ 199   31]]

    # clf = svm.SVC()
    # clf.fit(train_x, train_y.ravel())
    # predict = clf.predict(test_x)
    #
    # # for one hot all attributes
    # print(accuracy_score(test_y, predict.round()))  # 0.81161007667
    # print(confusion_matrix(test_y, predict.round()))
    # # [[1482    0]
    # # [ 344    0]]

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(train_x, train_y.ravel())
    predict = clf.predict(test_x)

    # for one hot all attributes
    print(accuracy_score(test_y, predict.round()))  # 0.81161007667
    print(confusion_matrix(test_y, predict.round()))
    # [[1482    0]
    # [ 344    0]]

    # kf = KFold(n_splits=3)
    # for train, test in kf.split(df_x):
    #     train_data = np.array(df_x)[train]
    #     test_data = np.array(df_x)[test]

if __name__ == '__main__':
    main()
