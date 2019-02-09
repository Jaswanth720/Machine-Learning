import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

def read_dataset(train_data_path, dev_data_path, test_data_path):
    train_dataset = pd.read_csv(train_data_path, header=None)
    dev_dataset = pd.read_csv(dev_data_path, header=None)
    test_dataset = pd.read_csv(test_data_path, header=None)

    print(train_dataset)

    finalset = pd.concat([train_dataset, dev_dataset, test_dataset], keys=[0,1,2])

    temp = pd.get_dummies(finalset, columns = [1,2,3,4,5,6,8])

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


def naive(x,y,x1,y1,x2,y2):
        
        w=np.zeros((x.shape[1]))
        wsum = np.zeros((x.shape[1]))
        c = 0
        mistakes = 0
        acc1=[]
        acc2=[]
        acc3=[]
        for i in range(5):
            mistakes = 0
            for j in range(x.shape[0]):
                y_h = np.sign(np.dot(w,x[j]))
                if(y_h == 0):
                    y_h-1
                   
                if(y_h!=y[j]):
                   w = w + (y[j]*x[j])
                   wsum = wsum + w
                   c = c + 1 
                   mistakes += 1
                   
        wavg=wsum/c
        acc1 = 1 - (mistakes/x.shape[0])
       
        
        acc2 = peptest(x1,y1,wavg)
        acc3 = peptest(x2,y2,wavg)
            
     
        return (wavg,acc1,acc2,acc3)

def peptest(x,y,w):
    c=0
    for j in range (x.shape[0]):  
        y_hat = np.sign(np.dot(w, x[j])) 
        if(y_hat == 0):
            y_hat = -1
        if(y_hat != y[j]):
            c = c+1
        accuracy= 1-(c/x.shape[0])
     
    
    return (accuracy)

if __name__ == '__main__':
    train_data_path = "income-data/income.train.txt"
    dev_data_path = "income-data/income.dev.txt"
    test_data_path = "income-data/income.test.txt"

    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_dataset(train_data_path, dev_data_path, test_data_path)
    (wavg,acc1,acc2,acc3) = naive(train_x, train_y, dev_x, dev_y, test_x, test_y)
    print(acc1)
    print(acc2)
    print(acc3) 