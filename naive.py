import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

def read_dataset(train_data_path, dev_data_path, test_data_path):
    
    train_dataset = pd.read_csv(train_data_path, header=None)
    
    dev_dataset = pd.read_csv(dev_data_path, header=None)
    
    test_dataset= pd.read_csv(test_data_path, header=None)

    
    
    
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
 
 

def peptrain(x,y, x_dev, y_dev, x_test, y_test):
    w = np.array([0]* x.shape[1])   # initailize weights to zero
    wsum = 0                        # cummulative weight
    count = 0
    num_of_ex = x.shape[0]
    mistake = 0
    mistakes = []
    train_accs = []
    dev_accs = []
    test_accs = []
    maxiter = []
    ETA = 1
    correct_pred = 0
    
    for i in range(5):
        for t in range(num_of_ex):
            yt_hat = np.sign(np.dot(w,x[t]))
            if yt_hat==0:
                yt_hat=-1
                correct_pred += 1
            if yt_hat!=y[t]:
                w = w + ETA * y[t]*x[t]
                wsum = wsum+w
                count = count+1
                mistake = mistake+1

        train_acc = calcAccuracy(mistake,x.shape[0])
        train_accs.append(train_acc)
        
        dev_acc = testPerceptron(x_dev,y_dev,w)
        dev_accs.append(dev_acc)

        test_acc = testPerceptron(x_test,y_test, w)
        test_accs.append(test_acc)

        mistakes.append(mistake)
        mistake = 0
        maxiter.append(i+1)
        
    wavg = wsum/count 
    return wavg,mistakes,train_accs, dev_accs, test_accs, maxiter 

def calcAccuracy():
    pass 


def testPerceptron():
    pass

if __name__ == '__main__':
    train_data_path = "income-data/income.train.txt"
    dev_data_path = "income-data/income.dev.txt"
    test_data_path = "income-data/income.test.txt"

    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_dataset(train_data_path, dev_data_path, test_data_path)
    (train_acc, dev_acc, test_acc) = peptrain(train_x, train_y, dev_x, dev_y, test_x, test_y)

    print (train_accs )
    print (dev_accs)
    print (test_accs)

