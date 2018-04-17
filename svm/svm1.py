import pandas
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy
from sklearn.model_selection import KFold


def main():
    x_csv_file = 'flights_one_hot_x.csv'
    y_csv_file = 'flights_one_hot_y.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)
    y_csv_path = '../data/{}'.format(y_csv_file)

    df_x = pandas.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    X = numpy.array(df_x)
    df_y = pandas.read_csv(y_csv_path)
    df_y = df_y.drop(df_y.columns[[0]], axis=1)
    Y = numpy.array(df_y)

    msk = numpy.random.rand(len(X)) < 0.8

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

    clf = svm.SVR()
    clf.fit(train_x, train_y.ravel())
    predict = clf.predict(test_x)

    print(accuracy_score(test_y.ravel(), predict))  # 0.89218328841
    print(confusion_matrix(test_y.ravel(), predict))
    # [[1624    1]
    #  [ 199   31]]

    kf = KFold(n_splits=3)
    for train, test in kf.split(X):
        train_data = np.array(data)[train]
        test_data = np.array(data)[test]

if __name__ == '__main__':
    main()
