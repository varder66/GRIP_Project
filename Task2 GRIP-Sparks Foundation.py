#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Predicting the percentage of marks that a student is expected to score based upon the number of hours they studied.

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


s_data = pd.read_csv(r"http://bit.ly/w-data")
print("Data imported successfully")

s_data.head(10)


# In[5]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ##### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# In[19]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 
print(X)
print(y)


# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[12]:


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[20]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
print(y_pred)


# In[14]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[22]:


y1_pred = regressor.predict([[9.25]])
print("predicted score",y1_pred)


# In[23]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




