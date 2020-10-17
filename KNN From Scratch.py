#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[2]:


# Load data
iris = load_iris()
X, y = iris.data, iris.target


# In[3]:


# Split data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1810)


# In[4]:


# Print shape of data
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[5]:


class KNN:
    '''
    -------- Base KNN Class -------
    
    Inputs:
    
    X: the training set features to fit the data on
    Y: the training set labels to fit the data on
    n_neighbors: the number of neighbours
    
    Methods:
    
    1. euclidean: A method to calculate euclidean distance
    
    Input:
    x1: the first point to calculate the distance from
    x2: the second point to calculate the distance from
    
    Returns: euclidean distance between the two points
    
    2. fit_knn: internal method to fit a KNN on the data
    
    Input:
    X_test = the test set to calculate distances from
    
    Returns: nearest labels to the test set instances
    
    3. predict: method to make predictions
    
    Input:
    X_test = the test set to make predictions from
    
    Returns:
    preds: the predictions vector
    
    
    '''
    def __init__(self,X,y,n_neighbors=3):
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        
    def euclidean(self,x1,x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit_knn(self,X_test):
        distances = [self.euclidean(X_test,x) for x in X_train]
        k_nearest = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [y_train[i] for i in k_nearest]
        
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
    
    def predict(self,X_test):
        preds = [self.fit_knn(x) for x in X_test]
        return preds


# In[6]:


# Instantiate our KNN classifier
knn = KNN(X_train,y_train)


# In[7]:


# Calculate and print accuracy of model
accuracy = (y_test == knn.predict(X_test)).mean()
accuracy


# In[ ]:




