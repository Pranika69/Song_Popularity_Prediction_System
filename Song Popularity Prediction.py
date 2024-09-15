#!/usr/bin/env python
# coding: utf-8

# In[304]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[305]:


# Import csv file
songdf = pd.read_csv('song_data.csv')


# In[306]:


# Read top 5 data
songdf.head(5)


# In[307]:


songdf.isnull() #Check if there is any null values


# In[308]:


songdf.isnull().sum() # Check if there is any null value in a column


# In[309]:


songdf.describe() #Provide a summary of statistics for each numerical column 


# In[310]:


songdf.info()


# # Data Manipulation

# In[311]:


# Drop columnn with not required data
songdf.drop(['song_name'],axis=1, inplace = True)


# In[312]:


songdf.head(5)


# In[313]:


# Segregate data into input and output data sets
X = songdf.drop('song_popularity',axis=1) #Features


# In[314]:


X.head()


# In[315]:


y = songdf['song_popularity'] #Target Variable


# In[316]:


y.head()


# # Data Visualization

# In[317]:


import seaborn as sns 
import matplotlib.pyplot as plt


# In[318]:


plt.figure(figsize=[10,5])

sns.histplot(
    songdf['song_popularity'], kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor="black", linewidth =1
)

plt.show()



# In[ ]:





# # Features Selection

# ***scikit-learn's SelectKBest method***

# In[319]:


# Define number of features to keep 
k=8

#Perform feature selection
selector = SelectKBest(score_func=f_regression, k=k).fit_transform(X,y)

# Get features name of selected features
selected_features = X.columns[SelectKBest(f_regression, k=k).fit(X,y).get_support()]


# In[320]:


print(selector)

print("\nSelected features based on ANOVA F-test:\n")
# print name of selected features
print(selected_features)

 


# **Feature Importance**

# In[321]:


model = ExtraTreesRegressor()
model.fit(X,y)


# In[322]:


print(model.feature_importances_)


# In[323]:


X.columns


# In[324]:


#Select top 8 features
X = X[['acousticness', 'danceability', 'instrumentalness', 'liveness',
       'loudness', 'tempo','time_signature', 'audio_valence']]


# In[325]:


X.shape


# # Modeling

# In[326]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[327]:


X_train.shape


# In[328]:


# Initialize the linear regression model
model = LinearRegression()


# In[329]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[330]:


# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[331]:


# Assuming X_test is a matrix or array with multiple columns
plt.figure(figsize=(10, 6))

# Plotting actual vs. predicted values
plt.figure(figsize=[10,5])

sns.histplot(
    y_test-y_pred, kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor="black", linewidth =1
)

plt.show()


# In[332]:


plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Actual vs. Predicted Output')
plt.show()

 


# In[333]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[334]:


print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


# In[335]:


sample_features = X_test.iloc[0] # taking one sample from test data


# In[336]:


sample_features


# In[337]:


sample_features_reshaped = sample_features.values.reshape(1, -1)


# In[338]:


sample_features_reshaped.shape


# In[339]:


sample_features.shape


# In[340]:


model.predict(sample_features_reshaped) # en


# In[341]:


np.sqrt(mse) #Root mean squared error

