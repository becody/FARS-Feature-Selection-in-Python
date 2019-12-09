# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:13:15 2019

@author: 17046
"""
###############################################
# Orange HW Team 2 
###############################################

###############################################
#  Reading in data 
###############################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

ncdata = pd.read_csv("C:/Users/17046/OneDrive/Documents/MSA 20/Visualization/ncdata.csv", 
                     index_col=0, na_values=['no info','.']) #need to code the NA's as strings

#Dropping VIN variables and converting to a dataframe
ncdata = pd.DataFrame(data=ncdata)
ncdata=ncdata.loc[:, ncdata.isnull().mean() < .8*ncdata.shape[0]]
ncdata = ncdata[ncdata.columns.drop(list(ncdata.filter(regex='VIN')))]
ncdata = ncdata.drop(['DEATHS','PERSONS', 'ST_CASE','VE_FORMS'], axis=1)

# Get the columns that are objects and keep only the ones with less than 20 unique categories

for col in ncdata.columns[ncdata.dtypes.eq('object')]:
    if len(ncdata.columns[ncdata.dtypes.eq('object')].unique()) > 20:
        ncdata.drop(col, inplace=True, axis=1)

for col in ncdata.columns:
    print(col)
    
##################################################
# Make new dataframe with dummy variables 
##################################################

cols = list(ncdata.columns[ncdata.dtypes.eq('object')])

ncdata_dums = pd.get_dummies(ncdata[cols], drop_first=True)
ncdata_dums

nc_dataNEW = pd.concat([ncdata, ncdata_dums], axis=1)

nc_dataNEW = nc_dataNEW.drop(cols, axis=1)

#dropping columns / features with majority NaN values
nc_dataNEW = nc_dataNEW.dropna(thresh=len(nc_dataNEW) -2, axis=1)

##################################################
# Starting machine learning techniques on data
##################################################
#Starting feature selection by creating target & features

X = nc_dataNEW.drop('FATALS', 1) #feature matrix
y = nc_dataNEW['FATALS'] #target variable

##################################################
# Backward Elimination
##################################################

#adding constant column of ones as intercept 
X1 = sm.add_constant(X)

#fitting sm.OLS model
model = sm.OLS(y,X1.astype(float)).fit()
model.pvalues

#Backward Selection
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.00001):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

#######################################################
# Recursive Feature Elimination
#######################################################
​
model = LinearRegression()
rfe = RFE(model, 7)
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
​
#Finding optimum number of features:
nof_list=np.arange(1,30)            
high_score=0

nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
​
#Final model using optimum number of features:
cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 29)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

##############################################################
# Feature Importance using Random forest and Extra Trees 
##############################################################
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib
​
model = ExtraTreesClassifier(n_estimators=338)
model.fit(X,y)
​
for name, importance in zip(X.columns, model.feature_importances_):
    print(name, "=", importance)
​
#Listing 30 most important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()
print(feat_importances.nlargest(30))






















