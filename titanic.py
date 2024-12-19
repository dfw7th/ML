#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


titanic_train = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/train.csv")
titanic_test = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/test.csv")


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_train.info()


# ### Drop "Cabin", "PassengerId"

# In[ ]:


true_names = titanic_train["Name"].str.split(",", expand=True)
true_names.head()


# In[ ]:


titanic_train["Name"] = true_names[0]


# In[ ]:


titanic_train


# In[ ]:


titanic_train["Cabin"].value_counts()


# In[ ]:


titanic_train.isnull().sum().sort_values(ascending=False)


# In[ ]:


titanic_train.describe()


# In[ ]:


titanic_train = titanic_train.drop(["Cabin", "PassengerId"], axis=1)
titanic_test = titanic_test.drop(["Cabin", "PassengerId"], axis=1)


# In[ ]:


titanic_train


# In[ ]:


sns.histplot(titanic_train["Age"], bins=70)


# In[ ]:


titanic_train["Age"] = titanic_train["Age"].fillna(value=28)


# In[ ]:


titanic_train = titanic_train.dropna()


# In[ ]:


titanic_train.head()


# In[ ]:


plt.figure(figsize=(12, 5))
sns.heatmap(titanic_train.corr(numeric_only=True), annot=True, cmap="YlGnBu")


# In[ ]:


#Let's check the correlation between the target variable and continuous variables
from scipy import stats
age = np.array(titanic_train["Age"])
sur = np.array(titanic_train["Survived"])
stats.pointbiserialr(age, sur)


# In[ ]:


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


# In[ ]:


sns.barplot(data=titanic_train, x="AgeGroup", y="Survived", errorbar=None)
plt.title("Survival Rate by Age Group")
plt.show()


# In[ ]:


titanic_train.head()


# In[ ]:


sns.boxplot(data=titanic_train, x="Survived", y="Age")


# In[ ]:


titanic_train["Pclass"].unique()


# In[ ]:


#correlation btw Pclass and survived, we use a chi-square test
from scipy.stats import chi2_contingency

# Create a contingency table (cross-tabulation)
contingency_table = pd.crosstab(titanic_train['Pclass'], titanic_train['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# We keep Pclass, the p value is essentially 0 and the chi2 stat is very high == good hahahahhah
# A high value indicates that the observed frequencies deviate significantly from the expected frequencies under the assumption of independence.


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Name'], titanic_train['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# Keep Name as well


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Survived'], titanic_train['Sex'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# So a particular Sex had a higher chane of survival woww


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Survived'], titanic_train['AgeGroup'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")


# In[ ]:


# Let's check the survival rates of each group
survival_rates = titanic_train.groupby('AgeGroup')['Survived'].mean()
print(survival_rates)

# So children had the highest rate of survival?? LFG those adults


# In[ ]:


titanic_train["SibSp"].sort_values(ascending=False)


# In[ ]:


from scipy.stats import spearmanr
# Let's take the spearman's rank between the no. of siblings and the survived
corr, pval = spearmanr(titanic_train["Survived"], titanic_train["SibSp"])
 
# print the result
print("Spearman's correlation coefficient:", corr)
print("p-value:", pval)

# Seems there is deviation in the no of siblings and how they affect the survival, but the features aren't necessarily correlated


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Survived'], titanic_train['SibSp'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")


# In[ ]:


from scipy.stats import pointbiserialr

correlation, p_value = pointbiserialr(titanic_train['SibSp'], titanic_train['Survived'])
print(f"Correlation: {correlation}, P-value: {p_value}")


# In[ ]:


# Let's do the same with point biseral correlation
corr, pval = stats.pointbiserialr(titanic_train["Survived"], titanic_train["SibSp"])

print("Point biseral correlation:", corr)
print("p-value:", pval)

# Seems this value is kinda washed for Log reg, maybe another ML algo might capture it better but we're gonna have to drop it, sadly.


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Name'], titanic_train['Parch'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Name'], titanic_train['Ticket'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# Keep Name as well


# In[ ]:


contingency_table = pd.crosstab(titanic_train['Name'], titanic_train['Embarked'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# Keep Name as well


# In[ ]:


from scipy.stats import f_oneway

stat, pval = f_oneway(titanic_train["Fare"], titanic_train["Survived"])
print(f"This is the F-statistic: {stat}")
print(f"This is the p_value: {pval}")


# In[ ]:


titanic_train.info()


# # drop Age, SibSp, Parch

# In[ ]:


titanic_train = titanic_train.drop(["Age", "SibSp", "Parch"], axis=1)
titanic_test = titanic_test.drop(["Age", "SibSp", "Parch"], axis=1)


# In[ ]:




