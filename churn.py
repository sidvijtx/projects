#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Generate random data
np.random.seed(42)
num_samples = 1000
data = {
    'Feature1': np.random.rand(num_samples),
    'Feature2': np.random.rand(num_samples),
    'Feature3': np.random.rand(num_samples),
    'Churn': np.random.choice([0, 1], size=num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)



# In[8]:


# Assuming the target variable is named 'Churn'
X = df.drop('Churn', axis=1)
y = df['Churn']


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# In[18]:


# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# In[19]:


from sklearn.tree import DecisionTreeClassifier
# Create and train a decision tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Make predictions with the decision tree
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate the decision tree model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
classification_rep_tree = classification_report(y_test, y_pred_tree)

# Print the results for the decision tree
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_tree:.2f}")
print(f"Confusion Matrix:\n{conf_matrix_tree}")
print(f"Classification Report:\n{classification_rep_tree}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




