#!/usr/bin/env python
# coding: utf-8

from sklearn.svm import SVC
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import entropy_sampling
from modAL.density import information_density
from IPython.display import display
import numpy as np
import pandas as pd
import random

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt



def train_dev_test_split(data, labels, train_size, test_size):
    dev_size=1-train_size-test_size
    X_dev, X_rem, y_dev, y_rem = train_test_split(data,labels, train_size=dev_size)
    X_train, X_test, y_train, y_test = train_test_split(X_rem,y_rem, test_size=test_size/(train_size+test_size))
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def displayAddressPair(index, addressPairs):
    lines_indices = ['Address 1', 'Address 2']
    columns = ['INBUILDING','EXTBUILDING','POILOGISTIC','ZONE','DSITRICT','HOUSENUM','ROADNAME', 'CITY']
    valuesAdd1, valuesAdd2 = [], []
    for i in range (22, 30):
        valuesAdd1.append(addressPairs[index][i])
        valuesAdd2.append(addressPairs[index][i+8])
    values = [valuesAdd1, valuesAdd2]
    pair = pd.DataFrame(values, index = lines_indices, columns = columns )
    display(pair)

def manualAL(classifier, sampleRequest, nbIterations, X_train, y_train,X_test, y_test, X_pool, y_pool, X_poolWithAdd):
    learner = ActiveLearner(estimator=classifier, 
                            query_strategy = sampleRequest,
                            X_training=X_train, 
                            y_training=y_train)
    model_accuracy = learner.score(X_test, y_test)
    performance_history = [model_accuracy]
    trace = []
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X = X_pool[query_index]
        displayAddressPair(int(query_index), X_poolWithAdd)
        y = [float(input('Labellisation : 2: Match , 1: PartialMatch, 0: NoMatch \n'))]
        learner.teach(X=X, y=y)
        trace.append(X_poolWithAdd[query_index[0]])
        X_pool,  y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index, axis=0)
        X_poolWithAdd = np.delete(X_poolWithAdd, query_index, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        performance_history.append(model_accuracy)
    return performance_history, trace

def autoAL(classifier, sampleRequest, nbIterations, X_train, y_train,X_test, y_test, X_pool, y_pool, X_poolWithAdd):
    learner = ActiveLearner(estimator=classifier, 
                        query_strategy = sampleRequest,
                        X_training=X_train, 
                        y_training=y_train)
    model_accuracy = learner.score(X_test, y_test)
    performance_history = [model_accuracy]
    trace = []
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index], y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        trace.append(X_poolWithAdd[query_index[0]])
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        X_poolWithAdd = np.delete(X_poolWithAdd, query_index, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        performance_history.append(model_accuracy)
    return performance_history, trace


def autoBatchAL(classifier, sampleRequest, nbIterations, X_train, y_train,X_test, y_test, X_pool, y_pool, X_poolWithAdd,batchSize):
    learner = ActiveLearner(estimator=classifier, 
                        query_strategy = sampleRequest,
                        X_training=X_train, 
                        y_training=y_train)
    model_accuracy = learner.score(X_test, y_test)
    performance_history = [model_accuracy]
    trace = []
    for index in range(nbIterations):
        for o in range(batchSize):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index], y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            trace.append(X_poolWithAdd[query_index[0]])
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            X_poolWithAdd = np.delete(X_poolWithAdd, query_index, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        performance_history.append(model_accuracy)
    return performance_history, trace

def random_sampling(classifier, X_pool):
    return [random.randrange(len(X_pool))]
    
def density_sampling(classifier, X_pool, metric):
    n_samples = len(X_pool)
    density = information_density(X_pool, metric)
    query_idx = 0
    min_density = density[0]
    for i in range(n_samples):
        if(density[i] < min_density):
            min_density = density[i]
            query_idx = i
    return [query_idx]

def euclideanDensity_sampling(classifier, X_pool):
    return density_sampling(classifier, X_pool, 'euclidean')

def manhattanDensity_sampling(classifier, X_pool):
    return density_sampling(classifier, X_pool, 'manhattan')

def cosineDensity_sampling(classifier, X_pool):
    return density_sampling(classifier, X_pool, 'cosine')