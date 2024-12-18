#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_train = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/train.csv")
titanic_test = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/test.csv")


# In[3]:


titanic_train.head()


# In[4]:


sns.histplot(titanic_train["Age"], bins=70)


# In[5]:


titanic_train.info()


# ### Drop "Cabin", "PassengerId"

# In[6]:


true_names = titanic_train["Name"].str.split(",", expand=True)
true_names.head()


# In[7]:


titanic_train["Name"] = true_names[0]


# In[8]:


titanic_train


# In[9]:


titanic_train["Cabin"].value_counts()


# In[10]:


titanic_train.isnull().sum().sort_values(ascending=False)


# In[11]:


titanic_train.describe()


# In[12]:


titanic_train = titanic_train.drop(["Cabin", "PassengerId"], axis=1)
titanic_test = titanic_test.drop(["Cabin", "PassengerId"], axis=1)


# In[13]:


titanic_train


# In[14]:


titanic_train["Age"] = titanic_train["Age"].fillna(value=28)


# In[15]:


titanic_train = titanic_train.dropna()


# In[16]:


titanic_train.head()


# In[17]:


plt.figure(figsize=(12, 5))
sns.heatmap(titanic_train.corr(numeric_only=True), annot=True, cmap="YlGnBu")


# In[18]:


#Let's check the correlation between the target variable and continuous variables
from scipy import stats
age = np.array(titanic_train["Age"])
sur = np.array(titanic_train["Survived"])
stats.pointbiserialr(age, sur)


# In[37]:


# Binning the Age groups
titanic_train["AgeGroup"] = pd.cut(
    titanic_train["Age"],
    bins=[0, 13, 19, 60, np.inf],
    labels=["Child", "Teen", "Adult", "Senior"]
)

titanic_test["AgeGroup"] = pd.cut(
    titanic_test["Age"],
    bins=[0, 13, 19, 60, np.inf],
    labels=["Child", "Teen", "Adult", "Senior"]
)


# In[38]:


sns.barplot(data=titanic_train, x="AgeGroup", y="Survived", errorbar=None)
plt.title("Survival Rate by Age Group")
plt.show()


# In[39]:


titanic_train.head()


# In[43]:


df = titanic_train[titanic_train['Name'].str.contains('Cumings')] 
df


# In[45]:


titanic_train.query("Pclass = =1")


# In[48]:


titanic_train[titanic_train['Ticket'].str.contains('695')]


# In[46]:


titanic_train.info()


# In[ ]:




