#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv('adult.csv')


# In[3]:


# exploration


# In[4]:


df


# In[5]:


df.occupation.value_counts()


# In[6]:


df.workclass.value_counts()


# In[7]:


#instead of one-hot encoding income and gender and have 2-2 seperate columns, we will turn both these colummns as binary feature (0 or 1).


# In[8]:


pd.get_dummies(df.occupation).astype(int)


# In[9]:


pd.get_dummies(df.occupation).astype(int).add_prefix('occupation-')


# In[10]:


df=pd.concat([df.drop('occupation',axis=1), pd.get_dummies(df.occupation).astype(int).add_prefix('occupation-')],axis=1)


# In[11]:


df


# In[12]:


df=pd.concat([df.drop('workclass',axis=1), pd.get_dummies(df.workclass).astype(int).add_prefix('workclass-')],axis=1)



# In[13]:


df


# In[14]:


df=df.drop('education',axis=1)


# In[15]:


df=pd.concat([df.drop('relationship',axis=1), pd.get_dummies(df.relationship).astype(int).add_prefix('relationship_')],axis=1)


# In[16]:


df=pd.concat([df.drop('native-country',axis=1), pd.get_dummies(df ['native-country'] ).astype(int).add_prefix('native-country_')],axis=1)


# In[17]:


df=pd.concat([df.drop('marital-status',axis=1), pd.get_dummies(df['marital-status']).astype(int).add_prefix('marital-status_')],axis=1)


# In[18]:


df=pd.concat([df.drop('race',axis=1), pd.get_dummies(df.race).astype(int).add_prefix('race_')],axis=1)


# In[19]:


df


# In[20]:


#now encode gender and income as binary


# In[21]:


df['gender']=df['gender'].apply(lambda x: 1 if x== 'Male' else 0)
df['income']=df['income'].apply(lambda x: 1 if x== '>50K' else 0)


# In[22]:


df.columns.values


# In[23]:


df['gender']


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')


# In[25]:


df.corr()


# In[26]:


#we took abs as negative corr is also very important.


# In[27]:


correlations= df.corr()['income'].abs()
sorted_correlations= correlations.sort_values()


# In[28]:


#dropping bottom 80% of lowest showing correlations for visualizations(to have better visualization)


# In[29]:


num_cols_to_drop=int(0.8*len(df.columns))
cols_to_drop=sorted_correlations.iloc[:num_cols_to_drop].index
df_dropped=df.drop(cols_to_drop,axis=1)


# In[30]:


df_dropped


# In[31]:


plt.figure(figsize=(18,12))
sns.heatmap(df_dropped.corr(), annot=False, cmap='coolwarm')


# In[32]:


plt.figure(figsize=(15,10))
sns.heatmap(df_dropped.corr(), annot=True, cmap='coolwarm')


# In[33]:


#feature correaltion is not the same as feature importance
#desicion tree is the most natural way to approach this dataset and random forest is used


# In[51]:


df=df.drop('fnlwgt', axis=1)


# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


train_df, test_df= train_test_split(df, test_size=0.2)


# In[53]:


train_df


# In[54]:


test_df


# In[55]:


train_X=train_df.drop('income',axis=1)
train_Y=train_df['income']

test_X=test_df.drop('income',axis=1)
test_Y=test_df['income']


# In[56]:


forest=RandomForestClassifier()
forest.fit(train_X, train_Y)


# In[57]:


forest.score(test_X, test_Y)


# In[58]:


forest.feature_importances_


# In[59]:


forest.feature_names_in_


# In[60]:


importances=dict(zip(forest.feature_names_in_,forest.feature_importances_))


# In[61]:


importances


# In[67]:


import numpy
importances= {k: v for k, v in sorted(importances.items(), key=lambda X: X[1], reverse= True)}


# In[68]:


importances


# In[69]:


#hyper parameter tuning


# In[72]:


from sklearn.model_selection import GridSearchCV

param_grid= {
    'n_estimators': [50,100,250],
     'max_depth': [5,10,30,None],
     'min_samples_split':[2,4],
      'max_features': ['sqrt','log2']
}

grid_search=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid, verbose=10)


# In[73]:


grid_search.fit(train_X,train_Y)


# In[74]:


grid_search.best_estimator_


# In[75]:


forest=grid_search.best_estimator_


# In[76]:


forest.score(test_X, test_Y)


# In[77]:


importances=dict(zip(forest.feature_names_in_,forest.feature_importances_))
importances= {k: v for k, v in sorted(importances.items(), key=lambda X: X[1], reverse= True)}


# In[78]:


importances


# In[ ]:




