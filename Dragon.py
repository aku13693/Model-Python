#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.info()


# In[6]:


housing['MEDV'].value_counts()


# In[7]:


housing.describe()


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt


# In[9]:


#housing.hist(figsize=(20,20))
#plt.show()


# ## Train-Test Splitting

# In[10]:


#import numpy as np
#def split_train_test(data,test_ratio):
 #   np.random.seed(42)
  #  shuffled = np.random.permutation(len(data))
   # test_set_size = int(len(data)*test_ratio)
    #test_indices = shuffled[:test_set_size]
    #train_indices = shuffled[test_set_size:]
    #return data.iloc[test_indices],data.iloc[train_indices]


# In[11]:


#test_set,train_set = split_train_test(housing,0.2)


# In[12]:


#print (len(test_set))
#print (len(train_set))


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size=0.2,random_state=42)


# In[14]:


print (len(test_set))
print (len(train_set))


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split1.split(housing,housing['CHAS']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]


# In[16]:


strat_test_set['CHAS'].value_counts()


# In[17]:


housing =strat_train_set.copy()


# ## Looking for corelations

# In[18]:


corr_matrix = housing.corr()
#print(corr_matrix)


# In[19]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attr = ['MEDV','B','ZN','CHAS','LSTAT']


# In[21]:


scatter_matrix(housing[attr], figsize= (20,20))


# In[22]:


housing.plot(kind="scatter", x='B', y='MEDV', alpha=0.9)


# ## Trying out Attribute combinations

# In[23]:


housing['TPM'] = housing['TAX']/housing['RM']


# In[24]:


housing['TPM']


# In[25]:


housing.head()


# In[26]:


x=  housing.corr()
x['MEDV'].sort_values(ascending=False)


# In[27]:


housing = strat_train_set.drop('MEDV',axis=1)
housing_labels = strat_train_set['MEDV'].copy()


# ## Missing Attributes

# In[28]:


housing.info()


# In[29]:


# TO take care of missing attributes
# you have 3 options:
# 1. Get rid of the missing datapoints
#2. Get rid of the entire attribute
#3. Set the value to mean , median or 0


# In[30]:


a= housing.dropna(subset=['RM'])
#Original housing dataframe will remain unchanged


# In[31]:


a.info() 
a.shape


# In[32]:


b = housing.drop('RM', axis=1)
b.info()
#Original housing dataframe will remain unchanged


# In[33]:


median = housing['RM'].median()


# In[34]:


median


# In[35]:


housing['RM'].fillna(median)
#original dataframe unchanged


# In[36]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[37]:
    

imputer.statistics_


# In[38]:


X = imputer.transform(housing)


# In[39]:


X


# In[40]:


housing_pr = pd.DataFrame(X)


# In[41]:


housing_pr.info()


# ## Sci-kit learn Design

# Primarily 3 types of objects:
# 1. Estimators - It estimates some paramter based on data set. Eg. Imputer
# 
# It has a fit method and transform method.
# Fit - Fits the dataset and calculates internal parameters.
# 
# 2.Transformers - 
# 
# Transform method takes input and gives output based on learnings from fit().It also has a convinience function called fit_transform() which fits and then transforms.
# 
# 3.Predictors  - LinearRegression model is example. fit and predict() are 2 common functions. It also gives score() which will evaluate the predictions.

# ## Feature Scaling
# 
# Primarily 2 types of feature scaling methods:
# 1. Min-max scaling (Normalization)
# 
# Formula = value-min/(max-min)
# Sklearn provides a class for this called as MinMaxScaler
# 
# 2. Standardisation
# 
# Formula = value - mean / std dev
# Sklearn provides a class called StandardScaler

# ## Creating a Pipeline

# In[42]:


from sklearn.pipeline import Pipeline


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
     ('stand_scale', StandardScaler())
])


# In[45]:


housing_num_pr = my_pipeline.fit_transform(housing)


# In[46]:


housing_num_pr


# In[47]:


x=pd.DataFrame(housing_num_pr)
x


# ## Selecting a desired model for Dragon estates

# In[49]:


x.shape


# In[50]:


x.shape


# In[131]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[132]:


#model=LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()


# In[133]:


model.fit(housing_num_pr,housing_labels)


# In[134]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[135]:


prepared_data= my_pipeline.transform(some_data)


# In[136]:


model.predict(prepared_data)


# In[137]:


list(some_labels)


# ## Evaluating the model

# In[138]:


from sklearn.metrics import mean_squared_error


# In[139]:


housing_predictions = model.predict(housing_num_pr)


# In[140]:


mse = mean_squared_error(housing_predictions,housing_labels)


# In[141]:


import numpy as np
rmse = np.sqrt(mse)
rmse


# ## Using Better evaluation technique - Cross validation

# In[142]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_pr,housing_labels,scoring="neg_mean_squared_error", cv=10)


# In[143]:


scores
rmse_sc = np.sqrt(-scores)
rmse_sc


# In[144]:


def print_scores(scores):
    print("Scores are:",scores)
    print("Mean:",np.mean(scores))
    print("Standard Deviation:",np.std(scores))


# In[145]:


print_scores(rmse_sc)


# ## Saving the model

# In[146]:


from joblib import dump,load


# In[147]:


dump(model,'Dragon.joblib')


# ## Testing the model

# In[149]:


X_test = strat_test_set.drop('MEDV',axis =1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)


# In[150]:


final_pred= model.predict(X_test_prepared)


# In[151]:


final_mse = mean_squared_error(final_pred,Y_test)


# In[152]:


final_mse


# In[153]:


final_rmse = np.sqrt(final_mse)


# In[154]:


final_rmse


# In[157]:


prepared_data[0]


# In[ ]:




