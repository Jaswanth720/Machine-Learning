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

def train_svmlinear(train_x, train_y, dev_x, dev_y, test_x, test_y):
    c = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1]
    train_acc = []
    dev_acc = []
    test_acc = []
    for i in c:
        classifier = SVC(kernel='linear', C=i)
        train_svm = classifier.fit(train_x,train_y).score(train_x, train_y)
        dev_svm = classifier.score(dev_x, dev_y)
        test_svm = classifier.score(test_x, test_y)
        print(train_svm * 100)
        print(dev_svm * 100)
        print(test_svm * 100)

        train_acc.append(train_svm)
        dev_acc.append(dev_svm)
        test_acc.append(test_svm)
    
    plt.plot(c, train_acc)
    plt.title('accuracy of svm train data')    
    plt.xlabel('parameter')
    plt.ylabel('train_svm')
    plt.savefig('svm_train.png')
    plt.close(1)

    plt.plot(c, dev_acc)
    plt.title('accuracy of svm dev data')    
    plt.xlabel('parameter')
    plt.ylabel('dev_svm')
    plt.savefig('svm_dev.png')
    plt.close(1)

    plt.plot(c, test_acc)
    plt.title('accuracy of svm test data')    
    plt.xlabel('parameter')
    plt.ylabel('test_svm')
    plt.savefig('svm_test.png')
    plt.close(1)

    return (train_svm, dev_svm, test_svm)

if __name__ == '__main__':
    train_data_path = "income-data/income.train.txt"
    dev_data_path = "income-data/income.dev.txt"
    test_data_path = "income-data/income.test.txt"

    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_dataset(train_data_path, dev_data_path, test_data_path)
    (train_svm, dev_svm, test_svm) = train_svmlinear(train_x, train_y, dev_x, dev_y, test_x, test_y)