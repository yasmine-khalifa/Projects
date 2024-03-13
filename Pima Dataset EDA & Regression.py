#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Downloading relevant packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


# Loading the data file

df = pd.read_csv('PimaDiabetes.csv')


# In[13]:


# Information about the dataset. We can see there are no null values in any of the columns, which is a good sign 
# regarding missing values. 

df.info()


# In[21]:


# However, there may be '0' values in place of a null cell. In most columns, '0' values would be out of the ordinary,
# except for'Pregnancies' and 'Outcome'.
print('Glucose:',(df['Glucose']==0).sum())
print('Blood Pressure:',(df['BloodPressure']==0).sum())
print('Skin Thickness:',(df['SkinThickness']==0).sum())
print('Insulin:',(df['Insulin']==0).sum())
print('BMI:',(df['BMI']==0).sum())
print('Diabetes Pedigree:',(df['DiabetesPedigree']==0).sum())
print('Age:',(df['Age']==0).sum())


# In[115]:


for column in df:
    print((df[column]==0).sum())


# In[25]:


# To remediate for these missing values, the zeroes can be replaced with the feature's median value.
# However, if there is a considerable volume of missing values in a column, using the median as a repalcement may 
# affect the bias and thus it would be best to remove the feature altogether to ensure precision.
print('Percent of "0" value in each feature:')
print('Glucose:',100*((df['Glucose']==0).sum()/750))
print('Blood Pressure:',100*((df['BloodPressure']==0).sum()/750))
print('Skin Thickness:',100*((df['SkinThickness']==0).sum()/750))
print('Insulin:',100*((df['Insulin']==0).sum()/750))
print('BMI:',100*((df['BMI']==0).sum()/750))
print('Diabetes Pedigree:',100*((df['DiabetesPedigree']==0).sum()/750))
print('Age:', 100*((df['Age']==0).sum()/750))
      


# In[ ]:


#Insulin and Skin Thickness have very significant percentages of zero values, therefore inserting the median
#to fill the empties may not suffice as the dataset will become too biased. Therefore, they should be removed. 


# In[28]:


df.drop('SkinThickness', axis=1, inplace=True)
df.drop('Insulin', axis=1, inplace=True)
df.info()


# In[35]:


#replacing zeroes with the median value for each feature
new_df = pd.DataFrame()
new_df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].median())
new_df['BMI'] = df['BMI'].replace(0,df['BMI'].median())
new_df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].median())


# In[38]:


# Checking that the percentage of '0' values in the new columns is zero
print('Percent of "0" value in each feature:')
print('Glucose:',100*((new_df['Glucose']==0).sum()/750))
print('Blood Pressure:',100*((new_df['BloodPressure']==0).sum()/750))
print('BMI:',100*((new_df['BMI']==0).sum()/750))


# In[49]:


# Replacing the unamended columns with the new ones.
df['Glucose'] = new_df['Glucose'].values
df['BMI'] = new_df['BMI'].values
df['BloodPressure'] = new_df['BloodPressure'].values


# In[52]:


df.head()


# In[40]:


#separating to features and target outcome

con_cols = list(df.drop('Outcome',axis=1).columns)
target = ['Outcome']
print("Features :",(con_cols))
print("Target Outcome   :",(target))


# In[53]:


#summary statistics

df[con_cols].describe().transpose()


# In[ ]:





# In[109]:


# Pearson's Correlation Coefficient in Matrix form
df_corr = df.corr().transpose()
mask = np.triu(np.ones_like(df_corr))
sns.heatmap(df_corr,mask=mask,fmt=".3f",annot=True)
plt.show()


# In[ ]:


# Notable correlation between Glucose and Outcome.
# The correlation between Age and Pregnancies can likely be explained by the fact that women are more likely to have
# had (more) children the older they get. 


# In[112]:


sns.pairplot(df,hue='Outcome')
plt.show()


# In[113]:


# Despite the lack of a clear separation in any of the above plots, a visible distinction appears in the glucose
#tables, whereby the 'Age', 'Diabetes Pedigree', 'BMI', and 'Blood Pressure' plots show a very general separation
#between outcomes as the value for Glucose increases. 


# In[ ]:





# In[117]:


fig = plt.figure(figsize=(5,5))
fig.text(1,1, 'Histogram for Pregnancies', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['Pregnancies'],kde=True)


# In[88]:


fig1 = plt.figure(figsize=(5,5))
fig1.text(1,1, 'Histogram for Glucose', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['Glucose'],kde=True)


# In[89]:


fig2 = plt.figure(figsize=(5,5))
fig2.text(1,1, 'Histogram for Blood Pressure', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['BloodPressure'],kde=True)


# In[90]:


fig3 = plt.figure(figsize=(5,5))
fig3.text(1,1, 'Histogram for BMI', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['BMI'],kde=True)


# In[91]:


fig4 = plt.figure(figsize=(5,5))
fig4.text(1,1, 'Histogram for Diabetes Pedigree', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['DiabetesPedigree'],kde=True)


# In[156]:


fig5 = plt.figure(figsize=(5,5))
fig5.text(1,1, 'Histogram for Age', fontsize=14, fontweight='bold',horizontalalignment='right')
sns.histplot(x=df['Age'],kde=True)


# In[331]:


df.skew()


# In[332]:


X1 = np.asarray(df[0:7])
xbar = np.mean(X1,1)

print(xbar)


# In[ ]:





# In[333]:


for i in con_cols:
    df.boxplot(i,'Outcome',figsize=(5,5))


# In[334]:


# if colum pregnancies is more than 3, create and set column threemorekids to True
df.loc[df ['Pregnancies']>= 3,'ThreeOrMoreKids'] = 1 
df.loc[df['Pregnancies'] <3, 'ThreeOrMoreKids'] = 0
df.head()


# In[335]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[336]:


X_train, X_test, y_train, y_test = train_test_split(df[['ThreeOrMoreKids']], df['Outcome'], test_size=0.10, random_state=1)
model = LogisticRegression()
model. fit(X_train, y_train)


# In[337]:


model.score(X_train, y_train)


# In[338]:


model.score(X_test,y_test)


# In[339]:


# Slope of the function.
model.coef_


# In[340]:


# Intercept of the function. 
model.intercept_


# In[379]:


# Confusion Matrix.
from sklearn.metrics import confusion_matrix
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


# In[342]:


# Classification report. 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[343]:


# What is the probability that you get diabetes, given that you have two or fewer children?
# Outcome = 0
# Odds calculation
import math

odds0= math.exp(model.intercept_+model.coef_*0)
print('Odds:',odds0)

# Calculating probability
print('Probability:',odds0/(1+odds0))


# In[344]:


# What is the probability that you get diabetes, given that you have three or more children?
# Outcome = 1

odds1=math.exp(model.intercept_+model.coef_*1)
print('Odds:',odds1)

#Calculating probability.
print('Probability:',odds1/(1+odds1))


# In[345]:


print('Probability of developing diabetes given\ntwo or fewer children:',odds0/(1+odds0),'\nthree or more children:',odds1/(1+odds1))


# In[ ]:





# In[346]:


df['Outcome'].value_counts()


# In[347]:


df['Count']=1
df1 = df[['Outcome', 'ThreeOrMoreKids', 'Count']]
df1.head()


# In[ ]:





# In[348]:


pd.pivot_table(df,values='Count', index=['Outcome'], columns=['ThreeOrMoreKids'], aggfunc=np.size, fill_value=0)


# In[349]:


from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigree', 'Age']]
pca = PCA(n_components=4)
X_r=pca.fit_transform(X)
print('Original shape:', X.shape)
print('PCA shape:', X_r.shape)
colours=ListedColormap(["r","g"])
values = list(df['Outcome'])
plot = plt.scatter(X_r[:,0], X_r[:,1], cmap =colours, c=values)
plt.legend(handles=plot.legend_elements()[0], labels=[0,1])
plt.show()


# In[352]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
Xt = pipe.fit_transform(X)
plot = plt.scatter(Xt[:,0], Xt[:,1], cmap =colours, c=values)
plt.legend(handles=plot.legend_elements()[0], labels=[0,1])
plt.show()


# In[ ]:


# PCA aims to explain variance in the data by reducing the number of features. In this case, the number of features was reduced from 6 to 4.
# To enhance the PCA explanation, the features were normalised. 
# The two classes (Outcome 0 and Outcome 1) are still relatively unexplained by this PCA. There is a slight
# difference where class 0 values lay somehwhat to the left-hand-side, however, not enough to make a clear 
# distinction. Another feature importance method involves creating a model with all the features and using the 
# feature coeffiicents to rank importance. This is attempted below.


# In[353]:


X_train, X_test, y_train, y_test = train_test_split(df[['ThreeOrMoreKids','Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigree', 'Age']], df['Outcome'], test_size=0.10, random_state=1)
model1 = LogisticRegression(max_iter=500)
model1. fit(X_train, y_train)


importances = pd.DataFrame(data={'Attribute': X_train.columns,'Importance': model1.coef_[0]})
importances = importances.sort_values(by='Importance', ascending=False)

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='blue')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()


# In[330]:


importances


# In[ ]:


# This model shows that the four most important features are: 'DiabetesPedigree','ThreeOrMoreKids','Pregnancies', 
# and 'BMI'. Notably,'DiabetesPedigree' surpasses all others with a coefficient of 0.662. Moreover, the bottom 
# four features have a coefficient under 0.1, meaning they are almost negligible. Existing studies, however, 
# suggest the opposite. Chang et al. (2022) find that 'Glucose', 'BMI', and 'Age' have high importance in explaining 
# the data, while the other three features had very low importance ('ThreeOrMoreKids' is excluded). Thus,
# two regression models will be created, each with three features. The first will include 'DiabetesPedigree',
# 'ThreeOrMoreKids', and 'Pregnancies', and the second will include 'Glucose', 'BMI', and 'Age'.


# In[357]:


# Model 1: Logistic Regression using 'ThreeOrMoreKids', 'Pregnancies', and 'DiabetesPedigree' to predict Diabetes
X_train, X_test, y_train, y_test = train_test_split(df[['ThreeOrMoreKids','Pregnancies', 'DiabetesPedigree']], df['Outcome'], test_size=0.10, random_state=1)
model2 = LogisticRegression(max_iter=500)
model2. fit(X_train, y_train)


# In[358]:


model2.score(X_train, y_train)


# In[359]:


model2.score(X_test,y_test)


# In[360]:


# Model Coefficients.
model2.coef_


# In[361]:


# Intercept of the function. 
model2.intercept_


# In[380]:


# Model 1 Confusion Matrix
confusion_matrix2 = metrics.confusion_matrix(y_test,y_pred2)

cm_display2 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2, display_labels = [False, True])

cm_display2.plot()
plt.show()


# In[363]:


# Classification report. 

print(classification_report(y_test, y_pred2))


# In[364]:


X_train, X_test, y_train, y_test = train_test_split(df[['Glucose','BMI', 'Age']], df['Outcome'], test_size=0.10, random_state=1)
model3 = LogisticRegression(max_iter=500)
model3. fit(X_train, y_train)


# In[365]:


model3.score(X_train, y_train)


# In[366]:


model3.score(X_test,y_test)


# In[367]:


# Model Coefficients.
model3.coef_


# In[368]:


# Intercept of the function. 
model3.intercept_


# In[381]:


# Confusion Matrix
confusion_matrix3 = metrics.confusion_matrix(y_test,y_pred3)

cm_display3 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix3, display_labels = [False, True])

cm_display3.plot()
plt.show()


# In[370]:


# Classification report. 

print(classification_report(y_test, y_pred3))


# In[384]:


# To decide which model is best for predicting Diabetes, a Confusion Matrix and Classification Report were generated
# for each model. The most critical quadrant in the Confusion matrices is the 'False Positive' quadrant. 
# This quadrant refers to the observations that the model predicted to not develop Diabetes, but in truth actually did.
# The classification method that prioritises eliminating false positives is recall. Hence, the model with the lower
# value for false positives and higher value for recall is model3, and thus it will be the chosen one. 
# Furthermore, model3 has a higher accuracy score, both for the training set and the testing set, than model2. 

#model2 training set score: 0.696
#model2 test set score: 0.640

#model3 training set score: 0.662
#model3 test set score: 0.733


# In[385]:


# Loading the prediction data file

df_predict = pd.read_csv('ToPredict.csv')


# In[389]:


df_predict.head()


# In[393]:


model3.predict_proba(df_predict[['Glucose','BMI', 'Age']])


# In[ ]:


# the probabilities that these women have diabetes, based on the model, is the values on the right column (outcome=1)


# In[ ]:


#Glucose: 0.03350315, BMI: 0.08855645, Age: 0.02768888


# In[ ]:





# In[ ]:





# In[ ]:




