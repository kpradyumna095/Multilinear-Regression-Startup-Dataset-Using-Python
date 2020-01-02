# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:53:49 2019

@author: Hello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

help(plt.boxplot)
plt.boxplot(startup["Profit"])
plt.boxplot(startup["RnD"])
plt.boxplot(startup["Administration"])
plt.boxplot(startup["Marketing"])


Startup= pd.get_dummies(startup['State'])

startup= pd.concat([startup,Startup],axis=1)

startup= startup.drop(["State"],axis=1)

startup= startup.iloc[:,[3,0,1,2,4,5,6]]

import seaborn as sn

sn.pairplot(startup)
cor_values= startup.corr()


from sklearn.model_selection import train_test_split

train_data,test_data= train_test_split(startup)

import statsmodels.formula.api as smf
startup.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)
train_data.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)
test_data.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)

train_data.to_csv("train_data.csv",encoding="utf-8")
test_data.to_csv("test_data.csv",encoding="utf-8")


model1= smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork",data=train_data).fit()
model1.summary() ##0.951
# Administration is insignificant
## building alone without any other input variable

model1_ad= smf.ols("Profit~Administration", data= train_data).fit()
model1_ad.summary()
## alone also Administration is insignificant

model1_ma = smf.ols("Profit~Marketing", data= train_data).fit()
model1_ma.summary()
## only marketing is significant

model1_com= smf.ols("Profit~Administration+Marketing", data= train_data).fit()
model1_com.summary()
#both variables are significant, the intercept becomes insignificant


## plotting influence plot

import statsmodels.api as sm

sm.graphics.influence_plot(model1)

train_data1= train_data.drop(train_data.index[[4]], axis=0)

model2= smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork", data= train_data1).fit()
model2.summary() ##0.965, Both Marketing and Administration are insignificant

train_data2 = train_data.drop(train_data.index[[4,24]],axis=0)
model3 = smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork", data= train_data2).fit()
model3.summary()##0.967 and administration is insignificant

##looking into VIF plot to check if there is any dependency among input variables

rsq_rnd = smf.ols("RnD~Administration+Marketing+California+Florida+NewYork", data= train_data2).fit().rsquared
ViF_rnd = 1/(1-rsq_rnd)## 3.347

rsq_adm = smf.ols("Administration~RnD+Marketing+California+Florida+NewYork", data=train_data2).fit().rsquared
ViF_adm = 1/(1-rsq_adm) ## 1.36

rsq_mar = smf.ols("Marketing ~ RnD+Administration+California+Florida+NewYork", data= train_data2).fit().rsquared
ViF_mar = 1/(1-rsq_mar) ## 3.37

## all VIF values are below  10 so there is no dependency among input variables

## checking for AV plot

sm.graphics.plot_partregress_grid(model2)
## as the correlation value between Profit and Administration is low and the AV plot also shows the same.lets remove Administration variable

model3= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data = train_data2).fit()
model3.summary()##0.967 ## all the variables are signifiacnt

############################################## final model ##################################
finalmodel= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data = train_data2).fit()
finalmodel.summary()##  0.967 (final model)

#training data prediction
train_pred = finalmodel.predict(train_data2)

## training residual

train_res= train_data2["Profit"]-train_pred

### train rmse
train_rmse = np.sqrt(np.mean(train_res*train_res)) ##6921

## test data prediction

test_pred = finalmodel.predict(test_data)

##test residuals

test_res= test_data["Profit"]- test_pred
##test rmse

test_rmse = np.sqrt(np.mean(test_res*test_res))##9828

## as the data is very less to properly train the model,there is difference in test rmse and train rmse

## training the model with the whole data set

startup1= startup.drop(startup.index[[4,24]],axis=0)
bestmodel= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data =startup1).fit()
bestmodel.summary()

bestmodel_pred = bestmodel.predict(startup1)
####validation

####################linearity##############
###  Observed values v/s Fitted values
plt.scatter(startup1.Profit,bestmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Fitted values")

##Residuals v/s Fitted values
plt.scatter(bestmodel_pred,bestmodel.resid_pearson, c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")

##Normality plot for residuals
##histogram
plt.hist(bestmodel.resid_pearson)

##QQplot

import pylab
import scipy.stats as st

st.probplot(bestmodel.resid_pearson, dist='norm', plot=pylab)

###Homoscadasticity
plt.scatter(bestmodel_pred,bestmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")
