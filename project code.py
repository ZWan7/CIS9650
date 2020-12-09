#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:45:51 2020

@author: tzuyili
"""

#import libraries we will use
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from sklearn.ensemble import RandomForestClassifier
#from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba.metric import AIC_score
#%matplotlib inline
pd.options.mode.chained_assignment = None 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


print("\nPart 1: Correlation")
df = pd.read_csv('student-mat.csv')


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(40, 40))

print("Correlation Plot")
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, vmin=-.5,center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})


print("\n(1) FEdu & Medu")
'''
(1) FEdu & Medu
In this correlation chart, we find the correlation efficient between "Fedu" and "Medu" was high, 
    and we can take a closer look at the relationship. 
From the following bar chart, we can see Father's education had a different distribution compared to mother's education. 
Most father's education were 2, while mothers tended to have higher education level.
Therefore, we choose to keep both FEdu and MEdu.
'''

# Bar charts of FEdu & Medu
plt.rc('figure', figsize=(10, 5))

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
Fedu = df['Fedu'].value_counts()
Fedu = pd.DataFrame(Fedu)
ax1.set_title("Father's Education Distribution")
plt.ylabel('Count of Fedu')
plt.xlabel('Fedu')
ax1.bar(Fedu.index, Fedu['Fedu'], color = 'cornflowerblue')

ax2 = fig.add_subplot(1, 2, 2)
Medu = df['Medu'].value_counts()
Medu = pd.DataFrame(Medu)
ax2.set_title("Mother's Education Distribution")
plt.ylabel('Count of Medu')
plt.xlabel('Medu')
ax2.bar(Medu.index, Medu['Medu'], color = 'pink')


print("\n(2) Walc & Dalc")
'''
Walc & Dalc
Also, we find the correlation coefficient between "Walc" and "Dalc" was high. 
We used the bar chart to visualize their distribution, 
    and found although both Walc and Dalc had an increase pattern, 
   the distribution seemed to be a little different. 
After level 2, the Dalc dropped dramatically than Walc.
Therefore, we chose to keep both variables.
'''

# Bar charts of Walc & Dalc

plt.rc('figure', figsize=(10, 5))

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
Walc = df['Walc'].value_counts()
Walc = pd.DataFrame(Walc)
ax1.set_title("Weekly Alchocol Consumption Distribution")
plt.ylabel('Count of Walc')
plt.xlabel('Walc')
ax1.bar(Walc.index, Walc['Walc'], color = 'Brown')

ax2 = fig.add_subplot(1, 2, 2)
Dalc = df['Dalc'].value_counts()
Dalc = pd.DataFrame(Dalc)
ax2.set_title("Daily Alchocol Consumption Distribution")
plt.ylabel('Count of Dalc')
plt.xlabel('Dalc')
ax2.bar(Dalc.index, Dalc['Dalc'], color = 'indianred')


print("\n(3) G1, G2, and G3")

'''
G1, G2 and G3
The 3 grades seemed to have high correlation. 
We visualzed their distribution to see it clearly. 
We found most students falled around 10 points in the three grades. 
However, more students got 0 in G2 and G3, while G1 had fewer students with 0 points.
The 3 grades distributions were different, so we chose to keep all of them.
'''

# Bar charts of G1, G2 and G3

plt.rc('figure', figsize=(15, 5))

fig = plt.figure()

ax1 = fig.add_subplot(1, 3, 1)
G1 = df['G1'].value_counts()
G1 = pd.DataFrame(G1)
ax1.set_title("G1 Distribution")
plt.ylabel('Count of G1')
plt.xlabel('G1')
ax1.bar(G1.index, G1['G1'], color = 'turquoise')

ax2 = fig.add_subplot(1, 3, 2)
G2 = df['G2'].value_counts()
G2 = pd.DataFrame(G2)
ax2.set_title("G2 Distribution")
plt.ylabel('Count of G2')
plt.xlabel('G2')
ax2.bar(G2.index, G2['G2'], color = 'lightseagreen')

ax3 = fig.add_subplot(1, 3, 3)
G3 = df['G3'].value_counts()
G3 = pd.DataFrame(G3)
ax3.set_title("G3 Distribution")
plt.ylabel('Count of G3')
plt.xlabel('G3')
ax3.bar(G3.index, G3['G3'], color = 'mediumaquamarine')


'''
To sum up, we choose to keep Medu, Fedu, Walc, Dalc, G1, G2, and G3 in our analysis. 
As for grades, we choose to convert 3 grades into average grade to do further analysis.
'''

path = r'C:\\Users\\xinyu\\Desktop\\Baruch Class\\CIS 9650\\group project'
os.chdir(path)
import os
os.environ["path"]+=os.pathsep+ 'C:\\ProgramData\\Anaconda3\\Library\\bin\\graphviz'

#We will drop the variables that are not valuable for our anaysis.
df.drop(columns=['school','age','Fjob','Mjob'],inplace=True)


#convert the data types
for col in df.columns:
    if df[col].dtypes=='object':
        df[col] = df[col].astype('category')



#There is no missing value in the dataset 
df[df.isna()].count()


# There is no duplicated value in the dataset 
df[df.duplicated()]

#Convert average_grade to binary.
df["average_grade"]=(df['G1']+df['G2']+df['G3'])/3

def BinaryResult(row):
    if row["average_grade"]>=10:
        return 1
    else:
        return 0          
df['grade_pass']=df.apply(BinaryResult,axis=1)




# * logistic regression



# partition data
df = pd.get_dummies(df, drop_first=True)
predictors = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'sex_M',
       'address_U', 'famsize_LE3', 'Pstatus_T','reason_home', 'reason_other',
       'reason_reputation', 'guardian_mother', 'guardian_other',
       'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes',
       'nursery_yes', 'higher_yes', 'internet_yes', 'romantic_yes']
X=df[predictors]
y=df['grade_pass']




# partition data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=1)

# fit a logistic regression (set penalty=l2 and C=1e42 to avoid regularization)
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg.fit(train_X, train_y)

print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': sorted(abs(logit_reg.coef_[0]),reverse=True)}, index=X.columns))
print()
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X), df = len(train_X.columns) + 1))

classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))

prediction_valid = logit_reg.predict(valid_X)
prediction_train = logit_reg.predict(train_X)
# precision
print("precision on test is:",precision_score(valid_y,prediction_valid))
# recall
print("recall on test is:",recall_score(valid_y,prediction_valid))
#f1
print("f1 on test is:",f1_score(valid_y,prediction_valid))
print("Logistic Regression:Accuracy on train is:",accuracy_score(train_y,prediction_train))
print("Logistic Regression:Accuracy on test is:",accuracy_score(valid_y,prediction_valid))


# * decision tree



fullClassTree = DecisionTreeClassifier(max_depth=4,random_state = 1)
fullClassTree.fit(train_X, train_y)
plotDecisionTree(fullClassTree, feature_names=train_X.columns)


prediction_train = fullClassTree.predict(train_X)#use the DT model to predict on the training data
prediction_valid = fullClassTree.predict(valid_X)#use the DT model to predict on the validation data
# precision
print("precision on test is:",precision_score(valid_y,prediction_valid))
# recall
print("recall on test is:",recall_score(valid_y,prediction_valid))
#f1
print("f1 on test is:",f1_score(valid_y,prediction_valid))
print("Logistic Regression:Accuracy on train is:",accuracy_score(train_y,prediction_train))
print("Logistic Regression:Accuracy on test is:",accuracy_score(valid_y,prediction_valid))

importances = fullClassTree.feature_importances_
important_df = pd.DataFrame({'feature': train_X.columns, 'importance': importances})#,"std":std})
important_df = important_df.sort_values('importance',ascending=False)
print(important_df)

#Compute the average grade of G1, G2 and G3
df["average_grade"]=(df['G1']+df['G2']+df['G3'])/3

# Compute the Medu's counts and the average_grade based on the Medu's groups

mean_medu = df.groupby('Medu').mean()['average_grade'].values.tolist()
del mean_medu[0]  # neglect the first element

count_medu = df.groupby('Medu').count()['average_grade'].values.tolist()
del count_medu[0]  # neglect the first element

x = ['1', '2', '3', '4']

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('medu')
ax1.set_ylabel('average grade', color=color)
ax1.plot(x, mean_medu, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
ax2.bar(x, count_medu, 0.4, color=color, alpha = 0.8)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Correlation between Average Grade and Medu')
plt.show()

# Compute the Fedu's counts and the average_grade based on the Fedu's groups

mean_fedu = df.groupby('Fedu').mean()['average_grade'].values.tolist()
del mean_fedu[0]  # neglect the first element

count_fedu = df.groupby('Fedu').count()['average_grade'].values.tolist()
del count_fedu[0]  # neglect the first element

x = ['1', '2', '3', '4']

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Fedu')
ax1.set_ylabel('average grade', color=color)
ax1.plot(x, mean_fedu, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
ax2.bar(x, count_fedu, 0.4, color=color, alpha = 0.8)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Correlation between Average Grade and Fedu')
plt.show()

# Compute the Studytime's counts and the average_grade based on the Studytime's groups

mean_studytime = df.groupby('studytime').mean()['average_grade'].values.tolist()


count_studytime = df.groupby('studytime').count()['average_grade'].values.tolist()



fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('studytime')
ax1.set_ylabel('average grade', color=color)
ax1.plot(x, mean_studytime, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
ax2.bar(x, count_studytime, 0.4, color=color, alpha = 0.8)
ax2.tick_params(axis='y', labelcolor=color)


plt.title('Correlation between Average Grade and Studytime')
plt.show()

# Compute the traveltime's counts and the average_grade based on the traveltime's groups

mean_traveltime = df.groupby('traveltime').mean()['average_grade'].values.tolist()
del mean_medu[0]  # neglect the first element

count_traveltime = df.groupby('traveltime').count()['average_grade'].values.tolist()
del count_medu[0]  # neglect the first element

x = ['1', '2', '3', '4']

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('traveltime')
ax1.set_ylabel('average grade', color=color)
ax1.plot(x, mean_traveltime, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('frequency', color=color)  # we already handled the x-label with ax1
ax2.bar(x, count_fedu, 0.4, color=color, alpha = 0.8)
ax2.tick_params(axis='y', labelcolor=color)


plt.title('Correlation between Average Grade and travel')
plt.show()




