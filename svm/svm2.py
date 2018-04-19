import pandas
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold


def main():
    x_csv_file = 'flights_one_hot_balanced.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)

    df_x = pandas.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_x['ARRIVAL_DELAY']
    df_x = df_x.drop(['ARRIVAL_DELAY'], axis=1)

    X = np.array(df_x)
    Y = np.array(df_y).ravel()

    print df_x.count

    msk = np.random.rand(len(X)) < 0.8

    print(msk)

    train_x = X[msk]
    train_y = Y[msk]

    test_x = X[~msk]
    test_y = Y[~msk]

    # clf = svm.SVC(class_weight='balanced')
    # clf.fit(train_x, train_y.ravel())
    # print(clf)
    # predict = clf.predict(test_x)
    #
    # print(accuracy_score(test_y, predict.round()))
    # print(confusion_matrix(test_y, predict.round()))
    # # SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    # #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    # #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    # #     tol=0.001, verbose=False)
    # # 0.547922437673
    # # [[829 650]
    # #  [166 160]]

    # clf = svm.SVC()
    # clf.fit(train_x, train_y.ravel())
    # print(clf)
    # predict = clf.predict(test_x)
    #
    # print(accuracy_score(test_y, predict.round()))
    # print(confusion_matrix(test_y, predict.round()))
    # # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    # #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    # #     tol=0.001, verbose=False)
    # # 0.599263200982
    # # [[1337  100]
    # #  [879  127]]

if __name__ == '__main__':
    main()
