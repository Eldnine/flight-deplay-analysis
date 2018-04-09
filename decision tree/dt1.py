
# coding: utf-8

# In[2]:


#decision tree model
import pandas as pd
import math
import numpy as np
import itertools


from sklearn.neural_network import MLPClassifier

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#read data
data=pd.read_csv('flights_one_hot_x.csv',sep=',')
#to exclude the first no_meaning column
data=data.drop(data.columns[[0]],axis=1)
y=pd.read_csv('flights_one_hot_y.csv',sep=',')
#to exclude the first no_meaning column
y=y.drop(y.columns[[0]],axis=1)
y=np.array(y)






# In[3]:


#To split the dataset
seed=3
test_size=0.2
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=test_size,random_state=seed)


# In[10]:


#fit the model


ada_model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),algorithm="SAMME",n_estimators=200)
ada_model.fit(x_train,y_train)
print(ada_model)


# In[11]:


#make Predictions
y_pred=ada_model.predict(x_test)
predictions=[round(value) for value in y_pred]
#evaluate Predictions
accuracy=metrics.accuracy_score(y_test,predictions)
print("Accuracy,%.2f%%"%(accuracy*100.0))


