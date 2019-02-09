#####################Import the packages 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate,train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib import pyplot
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
from sklearn import metrics


def fun1():
    columns = ['Age','Workclass','Education','Marital Status',
            'Occupation','Race','Sex',
            'Hours/Week','Native country','Income']
    train = pd.read_csv('income.train.txt', names=columns)
    test = pd.read_csv('income.test.txt', names=columns, skiprows=1)
    train.info()
    df = pd.concat([train, test], axis=0)
    dff=df
    k=df

    df['Income'] = df['Income'].apply(lambda x: 1 if x==' >50K' else 0)

    for col in df.columns:
        if type(df[col][0]) == str:
            print("Working on " + col)
            df[col] = df[col].apply(lambda val: val.replace(" ",""))


        
    df.replace(' ?', np.nan, inplace=True)###making copy for visualization


    df = pd.concat([df, pd.get_dummies(df['Workclass'],prefix='Workclass',prefix_sep=':')], axis=1)
    df.drop('Workclass',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Marital Status'],prefix='Marital Status',prefix_sep=':')], axis=1)
    df.drop('Marital Status',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Occupation'],prefix='Occupation',prefix_sep=':')], axis=1)
    df.drop('Occupation',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Race'],prefix='Race',prefix_sep=':')], axis=1)
    df.drop('Race',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Sex'],prefix='Sex',prefix_sep=':')], axis=1)
    df.drop('Sex',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Native country'],prefix='Native country',prefix_sep=':')], axis=1)
    df.drop('Native country',axis=1,inplace=True)

    df.drop('Education', axis=1,inplace=True)

    df.head()

    X = np.array(df.drop(['Income'], 1))
    y = np.array(df['Income'])
    X = preprocessing.scale(X)
    y = np.array(df['Income'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def fun2(X_train, X_test, y_train, y_test):
    clf_tree = DecisionTreeClassifier( max_depth = 4 )
    clf_tree.fit( X_train, y_train )
    tree_predict = clf_tree.predict( X_test )
    metrics.accuracy_score( y_test, tree_predict )

    print(confusion_matrix(y_test,tree_predict))
    print(classification_report(y_test,tree_predict))
    DTA = accuracy_score(y_test, tree_predict)
    print("The Accuracy for Decision Tree Model is {}".format(DTA))

def fun3(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier()
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    return grid_search

def fun6(grid_search):
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    pyplot.errorbar(learning_rate, means, yerr=stds)
    pyplot.title("XGBoost learning_rate vs Log Loss")
    pyplot.xlabel('learning_rate')
    pyplot.ylabel('Log Loss')
    pyplot.savefig('learning_rate.png')

def fun4(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    n_estimators = [100, 200, 300, 400, 500]
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    # plot results
    scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    for i, value in enumerate(learning_rate):
        pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    pyplot.legend()
    pyplot.xlabel('n_estimators')
    pyplot.ylabel('Log Loss')
    pyplot.savefig('n_estimators_vs_learning_rate.png')

def fun5():
    model = xgb.XGBClassifier(learning_rate=0.1,
                           n_estimators=500,
                           max_depth=5,
                           min_child_weight=4
                           )
    final_m=model.fit(X_train, y_train)
    xgb.plot_importance(final_m)
    plt.show()
    predictions = model.predict(X_test)
    print("training set auc:",accuracy_score(y_test, predictions))
    predictions = model.predict(X_test)
    print("test set auc:",accuracy_score(y_test, predictions))
    print(model.get_params())

    XGBA = accuracy_score(y_test, predictions)
    print("The Accuracy  is {}".format(XGBA))

X_train, X_test, y_train, y_test = fun1()

if __name__ == '__main__':
    gs = fun3(X_train, X_test, y_train, y_test)
    fun6(gs)

#fun2(X_train, X_test, y_train, y_test)
fun3(X_train, X_test, y_train, y_test)
# fun4(X_train, X_test, y_train, y_test)
# fun5(X_train, X_test, y_train, y_test)



