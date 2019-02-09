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

def peptrain(x, y, dev_x, dev_y, test_x, test_y):
    w = np.zeros((x.shape[1]))
    maxiter = []
    errors = []
    train_acc = []
    test_acc = []
    dev_acc = []
    for i in range(5):
        error = 0
        for t in range(x.shape[0]):
            y_cap = np.sign(np.dot(w,x[t]))

            if y_cap == 0:
                y_cap = -1

            if y_cap != y[t]:
               error = error + 1
               w = w + y[t] * x[t]
        errors.append(error)
        maxiter.append(i + 1)
        
        trainacc = 1 - ( error / x.shape[0] ) 
        devacc = peptest(dev_x, dev_y, w)
        testacc = peptest(test_x, test_y, w) 

        train_acc.append(trainacc)
        dev_acc.append(devacc)
        test_acc.append(testacc)

    return (train_acc, test_acc, dev_acc, maxiter, errors)
        





def peptest(x, y, w):
   mistakes = 0
   for t in range(x.shape[0]):
       y_cap = np.sign(np.dot(w,x[t]))
       
       if y_cap == 0:
          y_cap = -1
        
       if y_cap != y[t]:
           mistakes = mistakes + 1

   return (1 - (mistakes / x.shape[0])) 







if __name__ == '__main__':
    train_data_path = "income-data/income.train.txt"
    dev_data_path = "income-data/income.dev.txt"
    test_data_path = "income-data/income.test.txt"

    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_dataset(train_data_path, dev_data_path, test_data_path)
    (train_acc, test_acc, dev_acc, maxiter, errors) = peptrain(train_x, train_y, dev_x, dev_y, test_x, test_y)
    
    plt.plot(maxiter, errors)
    plt.title('learning curve')
    plt.xlabel('iterations')
    plt.ylabel('errors')
    plt.savefig('peplearningcurve.png')
    plt.close(1)

    plt.plot(maxiter, dev_acc)
    plt.title('dev accuracy')
    plt.xlabel('iterations')
    plt.ylabel('devaccuracy')
    plt.savefig('devaccuracy.png')
    plt.close(1)

    plt.plot(maxiter, test_acc)
    plt.title('test accuracy')
    plt.xlabel('iterations')
    plt.ylabel('testaccuracy')
    plt.savefig('testaccuracy.png')
    plt.close(1)

    plt.plot(maxiter, train_acc)
    plt.title('training accuracy')
    plt.xlabel('iterations')
    plt.ylabel('training accuracy')
    plt.savefig('trainingcurve.png')
    plt.close(1)