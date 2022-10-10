#!/usr/bin/env python
# coding: utf-8

# # PREDICTING PAY WITH OTHER INDICATORS

# ### WHAT FACTORS CONTIBUTES TO GENDER PAY GAP IN 2022

# ### PREPARE DATA

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


# #### IMPORT 

# In[2]:


#read in the data and inspect for cleaning
df = pd.read_excel('/Users/Cheryl/Downloads/WBL-1971-2022-Dataset-Updated.xlsx', sheet_name=2)
print(df.shape)
print(df.info())
df.head()


# In[3]:


#summary statistics
df.describe()


# In[4]:


mean_indicator = df[['MOBILITY', 'WORKPLACE', 'PAY', 'MARRIAGE', 'PARENTHOOD', 'ENTREPRENEURSHIP', 'ASSETS', 'PENSION']].mean().sort_values(ascending=False)
mean_WBL_index = df["WBL INDEX"].mean()

mean_indicator.plot(kind="bar")
plt.axhline(mean_WBL_index, linestyle="--", color="red", label="Global avearge 76.5")
plt.xlabel("Indicators")
plt.ylabel("WBL Index")
plt.title("Average WBL Scores by Indicator");


# In[5]:


mean_pay_by_income_group = df.groupby("Income Group")["WBL INDEX"].mean().sort_values(ascending=False)
print(mean_pay_by_income_group)
mean_pay_by_income_group.plot(kind="bar", xlabel="Income Group", ylabel="WBL Index", title="Mean WBL Scores by Income Group");


# #### EXPLORE

# In[6]:


# Drop leaky column
df.drop(columns="WBL INDEX", inplace=True)


# In[7]:


# Drop low- and high-cardinality categorical features
df.drop(columns=["Economy", "ISO Code", "Region", "Income Group"])


# In[8]:


# Check for Multicollinearity
corr = df.drop(columns = ["PAY"]).corr()
sns.heatmap(corr);


# In[9]:


print("corr1:", df["PAY"].corr(df["ASSETS"]))
print("corr2:", df["PAY"].corr(df["MARRIAGE"]))
print("corr3:",df["PAY"].corr(df["MOBILITY"]))


# In[10]:


# Drop multicollinearity column
df.drop(columns=["ASSETS", "MARRIAGE", "MOBILITY"], inplace=True)


# #### SPLIT

# In[12]:


# Create feature matrix "X_train" and target vector "y_train"
target = "PAY"
features = ['WORKPLACE', 'PARENTHOOD', 'ENTREPRENEURSHIP', 'PENSION']
X_train = df[features]
y_train = df[target]


# ### BUILD MODEL

# #### BASELINE

# In[13]:


# Calculate the baseline mean absolute error
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Mean PAY:", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))


# #### ITERATE

# In[14]:


# Create a pipeline named "model" 
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LinearRegression()
)
model.fit(X_train, y_train)


# ### EVALUATE

# In[15]:


# Calculate the training mean absolute error
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# Our model beats the baseline by 4.02! That's a good indicator that will be helpful in predicting pay. 

# In[16]:


X_train.head()


# In[17]:


# Make sure the order of columns in X_test is the same as in the X_train
X_test = pd.read_excel('/Users/Cheryl/Downloads/WBL-1971-2022-Dataset-Updated.xlsx', sheet_name=3)
X_test.head()


# In[18]:


# Check test performance
X_test = pd.read_excel('/Users/Cheryl/Downloads/WBL-1971-2022-Dataset-Updated.xlsx', sheet_name=3)[features]
y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()


# In[19]:


# Calculate the test mean absolute error
mae_test = mean_absolute_error(y_train, y_test_pred)
print("Test MAE:", round(mae_test, 2))


# Our test performance is about the same as our training performance. The training (21.71) and test performance (21.99) are close to each other, this means our model will generalize well. 

# ### COMMUNICATE RESULTS

# In[20]:


# Extract the intercepts from the model and assign to the variable intercepts
intercept = model.named_steps["linearregression"].intercept_
coefficients = model.named_steps["linearregression"].coef_
print(coefficients)


# In[21]:


# Extract feature names 
feature_names = model.named_steps["onehotencoder"].get_feature_names()
feature_names


# In[22]:


# Pandas series where the index is "features" and values are the "coefficients"
feat_imp = pd.Series(coefficients, index = feature_names)
feat_imp.head()


# In[23]:


# Print the equation for predicting pay based on WBL indicators
print(f"PAY = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f" + ({round(c, 2)} + {f})")


# In[24]:


# Horizontal bar chart showing the top coefficients for your model
feat_imp.sort_values(key=abs).plot(kind="barh")
plt.xlabel("Importance [PAY]")
plt.ylabel("Feature")
plt.title("Feature Importance for PAY Indicator");


# Looking at this bar chart, you can see that the most important indicators are "Workplace" and "Parenthood" for your model in predicting equal pay.  

# In[ ]:




