import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC


def read_dataset(train_data_path, dev_data_path, test_data_path):
    train_dataset = pd.read_csv(train_data_path, header=None)
    dev_dataset = pd.read_csv(dev_data_path, header=None)
    test_dataset= pd.read_csv(test_data_path, header=None)

    combined = pd.concat([train_dataset, dev_dataset, test_dataset], keys=[0,1,2])

    temp = pd.get_dummies(combined, columns = [1,2,3,4,5,6,8])

    train, dev, test = temp.xs(0), temp.xs(1), temp.xs(2)

    train_y = train[[9]]
    dev_y = dev[[9]]
    test_y = test[[9]]

    train_y = train_y.iloc[:,:].values
    dev_y = dev_y.iloc[:,:].values
    test_y = test_y.iloc[:,:].values

    train_y = train_y.reshape((train_y.shape[0]))
    dev_y = dev_y.reshape((dev_y.shape[0]))
    test_y = test_y.reshape((test_y.shape[0]))

    train_y[train_y == ' <=50K'] = -1
    dev_y[dev_y == ' <=50K'] = -1
    test_y[test_y == ' <=50K'] = -1

    train_y[train_y == ' >50K'] = 1
    dev_y[dev_y == ' >50K'] = 1
    test_y[test_y == ' >50K'] = 1    

    train_y = train_y.astype(int)
    dev_y = dev_y.astype(int)
    test_y = test_y.astype(int)

    train = train.drop(9, axis = 1)
    dev = dev.drop(9, axis = 1)
    test = test.drop(9, axis = 1)
    
    train_x = train.iloc[:,:].values
    dev_x = dev.iloc[:,:].values
    test_x = test.iloc[:,:].values

    return (train_x, train_y, dev_x, dev_y, test_x, test_y)

def train_svmpoly(train_x, train_y, dev_x, dev_y, test_x, test_y):
    
    degree = [2,3,4]
    for i in degree:
        classifier = SVC(kernel='poly', C=10**-2, degree= i )
        train_svm = classifier.fit(train_x,train_y).score(train_x, train_y)
        dev_svm = classifier.score(dev_x, dev_y)
        test_svm = classifier.score(test_x, test_y)
        print(train_svm)
        print(dev_svm)
        print(test_svm)

        
    return (train_svm, dev_svm, test_svm)
    

if __name__ == '__main__':
    train_data_path = "income-data/income.train.txt"
    dev_data_path = "income-data/income.dev.txt"
    test_data_path = "income-data/income.test.txt"

    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_dataset(train_data_path, dev_data_path, test_data_path)
    (train_svm, dev_svm, test_svm) = train_svmpoly(train_x, train_y, dev_x, dev_y, test_x, test_y)
    

