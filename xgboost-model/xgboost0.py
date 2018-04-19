import pandas as pd
import numpy as np
import xgboost
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positimve Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')



def score_result(y_test, predictions):
    label = 'ROC Curve'
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
    ROC = metrics.roc_auc_score(y_test, predictions)
    accuracy = metrics.accuracy_score(y_test, predictions)
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print("confusion_matrix:")
    print(confusion_matrix)
    print("Precision TP/(TP+FP),%.2f%%" % (metrics.precision_score(y_test, predictions) * 100.0))
    print("Recall TP/(TP+FN),%.2f%%" % (metrics.recall_score(y_test, predictions) * 100.0))
    print("ROC_area,%.2f%%" % (ROC * 100.0))
    print("Accuracy,%.2f%%" % (accuracy * 100.0))
    plot_roc_curve(fpr, tpr, label)



def main():
    x_csv_file = 'flights_one_hot_balanced_train_3.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)
    df_x = pd.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_x['ARRIVAL_DELAY']
    df_x = df_x.drop(['ARRIVAL_DELAY'], axis=1)
    X = np.array(df_x)
    Y = np.array(df_y).ravel()
    # msk = np.random.rand(len(X)) < 0.8
    train_x = X
    train_y = Y

    x_csv_file = 'flights_one_hot_balanced_test_3.csv'
    x_csv_path = '../data/{}'.format(x_csv_file)
    df_x = pd.read_csv(x_csv_path)
    df_x = df_x.drop(df_x.columns[[0]], axis=1)
    df_y = df_x['ARRIVAL_DELAY']
    df_x = df_x.drop(['ARRIVAL_DELAY'], axis=1)
    X = np.array(df_x)
    Y = np.array(df_y)
    test_x = X
    test_y = Y

    model = xgboost.XGBClassifier(
        max_depth=10,
        objective='binary:logistic')
    model.fit(train_x, train_y)
    print(model)
    predict = model.predict(test_x)
    score_result(test_y, predict)
    # XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
    #               max_depth=1000, min_child_weight=1, missing=None, n_estimators=100,
    #               n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
    #               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
    #               silent=True, subsample=1)
    # [[1415  134]
    #  [30  989]]

if __name__ == '__main__':
    main()
