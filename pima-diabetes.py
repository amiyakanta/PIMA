# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:43:08 2018

@author: Amiyakanta Sahu
"""
#Import python modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read data from CSV file to a Data Frame
df=pd.read_csv("D:\diabetes.csv")
#print(df)

#Describe the data to get first hand information like count,mean etc...
print(df.describe())

#Correlation Matrix
print(df.corr())

#Create variables (which have got highest correlation with Outcome)
Glucose = df['Glucose']
BMI = df['BMI']
Outcome = df['Outcome']

#Visualise Univariate i.e. Plot Histogram for the variables created above
plt.hist(Glucose)
plt.hist(BMI)
Outcome = df['Outcome']
plt.show()

#Visualise Bivariate i.e. scatter plot
plt.scatter(Glucose,BMI)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()

#Visualise Trivariate 
#Glucose in X-axis, BMI in Y-axis and Outcome in color i.e. Red is diabtic and Green is non-diabatic
colors={1:'red',0:'green'}
#Getting color to the scattered graphs
plt.scatter(Glucose, BMI, c=Outcome.apply(lambda x:colors[x]))
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()

#Sample and Split dataset for training and testing
train,test = np.split(df.sample(frac=1),[int(.7*len(df))]) 

#Inspect the training data
train.describe()
train.corr()
plt.hist(train.Glucose)
plt.hist(train.BMI)
#Inspect the test data
test.describe()
test.corr()
plt.hist(test.Glucose)
plt.hist(test.BMI)

#Outlier Analysis
sns.boxplot(x="Outcome",y="Glucose",data=df)
sns.stripplot(x="Outcome",y="Glucose",jitter=True,data=df)
plt.show()
sns.boxplot(x="Outcome",y="BMI",data=df)
sns.stripplot(x="Outcome",y="BMI",jitter=True,data=df)
plt.show()

#Combination Plot
sns.set(style="darkgrid", color_codes="true")
sns.FacetGrid(df, hue="Outcome", size=5 ).map(plt.scatter,"Glucose","BMI").add_legend()

#Joint Plot
sns.jointplot(x="Glucose",y="BMI",data=df,size=5)

#Bin Variables after Rules are discovered 
#Rule 1: Outcome is 1 if 30<BMI<=40
#Rule 2: Outcome is 1 if 120<Glucose<=170
df['BMIBin'] = np.where(df['BMI'] <30,'<30', (np.where(df['BMI'] >= 40 ,'>=40','30-40')))
df['GlucoseBin'] = np.where(df['Glucose'] <120,'<120', (np.where(df['Glucose'] >= 170 ,'>=170','120-170')))

#create Pivot Table on training data
pd.pivot_table(train, values="BMI", index=["BMIBin"], columns="Outcome", aggfunc = "count",fill_value=0)
pd.pivot_table(train, values="Glucose", index=["GlucoseBin"], columns="Outcome", aggfunc = "count",fill_value=0)

#create Pivot Table on test data
pd.pivot_table(test, values="BMI", index=["BMIBin"], columns="Outcome", aggfunc = "count",fill_value=0)
pd.pivot_table(test, values="Glucose", index=["GlucoseBin"], columns="Outcome", aggfunc = "count",fill_value=0)


