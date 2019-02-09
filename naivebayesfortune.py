import pandas as pd
import numpy as np 
import matplotlib.pyplot as p
from math import log 
from texttable import Texttable
import math
import operator 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression

def preprocess(givendatalocation, classlabellocation, extern_vocababulary=None):
    print("Preprocesing the dataset")
    stop_list_loc = "stoplist.txt"
    with open(givendatalocation, 'r') as file:
        givendata = file.read().splitlines()
    with open(classlabellocation, 'r') as file:
        classlabels_raw = file.read().splitlines()
    with open(stop_list_loc, 'r') as file:
        stop_list_raw = file.read().splitlines()
        stop_occurences = set()
    for line in stop_list_raw:
        occurence = line
        stop_occurences.add(occurence)
    vocabulary = set()
    if extern_vocababulary is None:
        for line in givendata:
            occurences = line.split()
            for occurence in occurences:
                if occurence not in stop_occurences:
                    vocabulary.add(occurence)
        sorted_vocabulary = sorted(vocabulary)
    else: vocabulary = extern_vocababulary
    classlabels = list(map(int, classlabels_raw))
    variable_sets = []
    for cookie in givendata:
        cookie_occurences = dict.fromkeys(vocabulary, 0)
        occurences = cookie.split()
        for occurence in occurences:
            if occurence in cookie_occurences:
                cookie_occurences[occurence] = 1
        variable_sets.append(cookie_occurences)
    variables = []
    for variable in variable_sets:
        sorted_set = sorted(variable.items(), key=operator.itemgetter(0))
        sorted_set_occ = [x[1] for x in sorted_set]
        variables.append(sorted_set_occ)

    if extern_vocababulary is None:
        return givendata, sorted_vocabulary, variables, classlabels
    else: return givendata, variables, classlabels    

    variables = []
    for variable in variable_sets:
        sorted_set = sorted(variable.items(), key=operator.itemgetter(0))
        sorted_set_occ = [x[1] for x in sorted_set]
        variables.append(sorted_set_occ)

    if extern_vocababulary is None:
        return givendata, sorted_vocabulary, variables, classlabels
    else: return givendata, variables, classlabels

class Naivebayes:
    variables = []
    classlabels = []
    cookies = []
    vocabulary = []
    def get_classlabel_prob(self, variable, compare_classlabel):
        numberofclasslabel = 0
        numberoftotalclasslabels = len(self.classlabels)
        for old_classlabel in self.classlabels:
            if old_classlabel == compare_classlabel: 
                numberofclasslabel = numberofclasslabel + 1
        X_classlabel = (numberofclasslabel + 1) / (numberoftotalclasslabels + 2)
         
        count_Y = [0 for x in range(len(self.vocabulary))]
        for i in range(len(self.variables)):
            v = self.variables[i]
            c = self.classlabels[i]
            if c == compare_classlabel:
               
                for j in range(len(v)):
                    if v[j] == variable[j]: count_Y[j] += 1
        X_product = 1
        i = 0
        for a in count_Y:
           
            prob_Y = (a + 1) / (numberofclasslabel + 2)
            X_product *= prob_Y
        return X_classlabel * X_product      
              
 
    def prediction(self, variable, classlabel, is_testing=True):
        
        classlabel_0_prob = self.get_classlabel_prob(variable, 0)
        classlabel_1_prob = self.get_classlabel_prob(variable, 1)
        
              
        predicted_classlabel = 0
        if (classlabel_1_prob > classlabel_0_prob): 
            predicted_classlabel = 1
        
              
        if is_testing is True: 
           
            self.variables.append(variable)
            self.classlabels.append(classlabel)
        
        return predicted_classlabel

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary


      
def train(cookies, vocabulary, variables, classlabels):

    classifier = Naivebayes(vocabulary)
    
    numberofmistakes = 0

    for i in range(len(cookies)):
    
        variable = variables[i]
        classlabel = classlabels[i]
    
        predicted_classlabel = classifier.prediction(variable, classlabel)
        if (predicted_classlabel != classlabel): 
            numberofmistakes = numberofmistakes + 1
    
    
    return classifier

def test(classifier, variables, classlabels):
    
    numberofmistakes = 0
    numberofvariables = len(variables)

    for i in range(numberofvariables):

        variable = variables[i]
        classlabel = classlabels[i]
        predicted_classlabel = classifier.prediction(variable, classlabel, True)
    
        if (predicted_classlabel != classlabel):
            numberofmistakes = numberofmistakes + 1

    test_accuracy = (1 - (numberofmistakes / numberofvariables)) * 100
    return test_accuracy



def main():
 
    train_cookies, vocabulary, train_variables, train_classlabels = preprocess("traindata.txt", "trainlabels.txt")
    classifier = train(train_cookies, vocabulary, train_variables, train_classlabels)
    train_cookies, test_variables, test_classlabels = preprocess("testdata.txt", "testlabels.txt", vocabulary) 
    train_accuracy = test(classifier, train_variables, train_classlabels)
    test_accuracy = test(classifier, test_variables, test_classlabels)
    print("Train accuracy = ", train_accuracy)
    print("Test accuracy = ", test_accuracy)
    train_logistic_variables = np.array(train_variables)
    train_logistic_classlabels = np.array(train_classlabels)
    test_logistic_variables = np.array(test_variables)
    test_logistic_classlabels = np.array(test_classlabels)
    regression = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_logistic_variables, train_logistic_classlabels)
    training_regression = regression.score(train_logistic_variables, train_logistic_classlabels)
    print("Training Accuracy of Logistic Regression=", training_regression)
    test_regression = regression.score(test_logistic_variables, test_logistic_classlabels)
    print("Testing Accuracy of Logistic Regression=", test_regression)
main()







 

