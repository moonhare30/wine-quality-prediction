#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY LIBRARIES

# In[300]:


# Importing all the necessary libraries

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv")


# # DATA EXPLORATION & CLEANING

# In[301]:




df.head() # Display first few rows


# In[302]:


df.info()  # Display summary of data types and missing values


# In[303]:


df.describe() # Display statistical summary of numerical columns


# In[304]:


# Check for missing values
print(df.isnull().sum())


# # FEATURE SELECTION & FEATURE ENGINEERING

# In[305]:


# Handle categorical variables
df['quality'] = df['quality'].apply(lambda x: 'low' if x <= 5 else 'medium' if x <= 7 else 'high')
df['quality'] = df['quality'].astype('category')

# Perform feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = df.drop('quality', axis=1)
X = scaler.fit_transform(X)

# Split data into features (X) and target (y)
y = df['quality']

# Create a new feature 'total_acidity' as the sum of fixed acidity and volatile acidity
df1 = df.copy()
df1['total_acidity'] = df1['fixed acidity'] + df1['volatile acidity']
df1.head()


# # DATA VISUALIZATION & EDA

# In[306]:


# Visualize the data
sns.countplot(x='quality', palette='magma', data=df)
plt.title('Distribution of Wine Quality')
plt.show()


# In[307]:


sns.set_style("whitegrid")


# In[308]:


df.hist(figsize=(15,20), color="#A82548");


# In[309]:


#Visualizing the correlations between numerical variables

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), alpha=0.5, cmap="Greens")
plt.title("Correlations Between Variables", size=15)
plt.show()


# In[310]:


# catplot for bhk vs price

def cat_plot(data,title):
    sns.catplot(x="alcohol", y="quality", palette="magma", alpha=0.5, data=data)
    plt.title('alcohol vs the quality of red wine '+ title,size=16)
    plt.gcf().set_size_inches(6,8)
    plt.show()


# In[311]:


cat_plot(df ,"quality")


# In[312]:


numerical_cols = df.select_dtypes(include='number').columns.tolist()

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

for col, ax in zip(numerical_cols, axs.flat): #iterating over numerical columns, and its respective subplot in flattened axes using zip()
    #col contains number of current numerical column and ax contains the current subplot
    sns.histplot(data=df, x=col, kde=True, ax=ax, color="#A82548")
    ax.set(title=col)
    if ax.get_subplotspec().is_last_row():
        ax.set(xlabel=col)

plt.tight_layout() #adjust the spacing between the subplots to eliminate overlapping
plt.show()


# In[313]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # subplots is used to make multiple plots in the same block
axes = axes.flatten()

sns.scatterplot(ax = axes[0],
                x = "residual sugar",
                y = "quality", hue = "quality",
                data = df).set(title = "Relationship between 'residual_sugar' and 'quality'");

sns.scatterplot(ax = axes[1],
                x = "alcohol",
                y = "quality", hue = "quality",
                data = df).set(title = "Relationship between 'alcohol' and 'quality'");

sns.scatterplot(ax = axes[2],
                x = "pH",
                y = "quality", hue = "quality",
                data = df).set(title = "Relationship between 'pH' and 'quality'");

sns.scatterplot(ax = axes[3],
                x = "density",
                y = "quality", hue = "quality",
                data = df).set(title = "Relationship between 'density' and 'quality'");


# In[314]:


plt.figure(figsize = [20, 10], facecolor = 'white')
sns.heatmap(df.corr(), annot = True, linewidths = 2, alpha=0.7, cmap = "magma");


#  # MODEL BUILDING

# In[315]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # MODEL SELECTION & HYPER PARAMETER TUNING

# **KNN**

# In[316]:


# Train and evaluate KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print('KNN Accuracy:', acc_knn)
print('Classification Report:', classification_report(y_test, y_pred_knn))


# In[317]:


# HYPER PARAMETER TUNING

knn_params = {"n_neighbors": np.arange(2, 40),
             "weights": ["uniform", "distance"],
             "p": [1, 2]}

knn_cv_model = GridSearchCV(knn, knn_params, cv = 10)
knn_cv_model.fit(X_train, y_train)


# In[318]:


print("Best score for train set: " + str(knn_cv_model.best_score_))

print("best K value: " + str(knn_cv_model.best_params_["n_neighbors"]),
     "\nbest weights: " + knn_cv_model.best_params_["weights"],
     "\nbest value of p: " + str(knn_cv_model.best_params_["p"])) 


# In[319]:


knn_model = KNeighborsClassifier(n_neighbors = knn_cv_model.best_params_["n_neighbors"],
                                weights = knn_cv_model.best_params_["weights"],
                                p = knn_cv_model.best_params_["p"],
                                )

knn_model.fit(X_train, y_train)


# In[320]:


y_pred = knn_model.predict(X_test)
newacc_knn = accuracy_score(y_test, y_pred)
print('KNN accuracy after hypertuning', newacc_knn)


# **LOGISTIC REGRESSION**

# In[321]:


# Train and evaluate Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print('Logistic Regression Accuracy:', acc_lr)
print('Classification Report:', classification_report(y_test, y_pred_lr))


# In[322]:


# HYPER PARAMETER TUNING

lr_params = {"C": [0.5, 0.75, 1, 1.5, 1.5, 2]}

lr_cv_model = GridSearchCV(lr, lr_params, cv = 10)
lr_cv_model.fit(X_train, y_train)


# In[323]:


print("Best score for train set: " + str(lr_cv_model.best_score_))

print("best C value: " + str(lr_cv_model.best_params_["C"]))


# In[324]:


lr_model = LogisticRegression(C = 0.5)
lr_model.fit(X_train, y_train)


# In[325]:


y_pred = lr_model.predict(X_test)
newacc_lr = accuracy_score(y_test, y_pred)
print('Logistic regression accuracy after hypertuning', newacc_lr)


# **DECISION TREE**

# In[326]:


# Train and evaluate Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print('Decision Tree Accuracy:', acc_dt)
print('Classification Report:', classification_report(y_test, y_pred_dt))


# In[327]:


# HYPER PARAMETER TUNING

dt_params = {"criterion": ["gini", "entropy"],
             "max_depth": [3, 4, 5, 6, 7],
             "max_features": [4, 5, 6, 7],
             "min_samples_split": [2, 3, 4, 5, 6]}

dt_cv_model = GridSearchCV(dt, dt_params, cv = 10)
dt_cv_model.fit(X_train, y_train)


# In[328]:


print("Best score for train set: " + str(dt_cv_model.best_score_))

print("best criterion : " + dt_cv_model.best_params_["criterion"],
     "\nbest max_depth: " + str(dt_cv_model.best_params_["max_depth"]),
     "\nbest max_features: " + str(dt_cv_model.best_params_["max_features"]),
     "\nbest min_samples_split: " + str(dt_cv_model.best_params_["min_samples_split"]))


# In[329]:


dt = DecisionTreeClassifier(criterion = dt_cv_model.best_params_["criterion"],
                                 max_depth = dt_cv_model.best_params_["max_depth"],
                                 max_features = dt_cv_model.best_params_["max_features"],
                                 min_samples_split = dt_cv_model.best_params_["min_samples_split"])
dt_model = dt.fit(X_train, y_train)


# In[330]:


y_pred = dt_model.predict(X_test)
newacc_dt = accuracy_score(y_test, y_pred)
print('Decision tree accuracy after hypertuning', newacc_dt)


# **RANDOM FOREST**

# In[331]:


# Train and evaluate Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print('Random Forest Accuracy:', acc_rf)
print('Classification Report:', classification_report(y_test, y_pred_rf))


# In[332]:


# HYPER PARAMETER TUNING

# rf_params = {
# #               "n_estimators": [100, 150, 250,],
# #               "max_depth": [2, 3, 5, 7],
# #               "min_samples_split": [2, 3, 4, 6]
# #             }


rf_params = {

'n_estimators': [50, 100, 200, 300],  # no of trees in the forest
    'max_depth': [None, 10, 20, 30],  # max depth of trees
    'min_samples_split': [2, 5, 10],  # min no of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # min no of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # no of features to consider when looking for the best split
    'random_state': [42]  # seed for random number generation

}

rf_cv_model = GridSearchCV(rf, rf_params, cv = 10, n_jobs = -1)
rf_cv_model.fit(X_train, y_train)


# In[333]:


print("Best score for train set: " + str(rf_cv_model.best_score_))

print("\nbest n_estimators: " + str(rf_cv_model.best_params_["n_estimators"]),
     "\nbest max_depth: " + str(rf_cv_model.best_params_["max_depth"]),
     "\nbest min_samples_split: " + str(rf_cv_model.best_params_["min_samples_split"]))


# In[334]:


rf = RandomForestClassifier(
                                 max_depth = rf_cv_model.best_params_["max_depth"],
                                 n_estimators = rf_cv_model.best_params_["n_estimators"],
                                 min_samples_split = rf_cv_model.best_params_["min_samples_split"],
                                #  min_samples_leaf = rf_cv_model.best_params_["min_samples_leaf"],
                                #  max_features = rf_cv_model.best_params_["max_features"],
                                 )

rf_model = rf.fit(X_train, y_train)


# In[335]:


y_pred = rf_model.predict(X_test)
newacc_rf = accuracy_score(y_test, y_pred)
print('Random forest accuracy after hypertuning', newacc_rf)


# # COMPARING ACCCURACIES

# In[336]:


# Compare model performance
models = ['KNN', 'LR', 'DT', 'RF']
accuracies = [acc_knn, acc_lr, acc_dt, acc_rf]

color = "#800020"

plt.bar(models, accuracies)
plt.title('Models accuracy before hypertuning')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
plt.show()

# Select the best-performing model
best_model_idx = accuracies.index(max(accuracies))
best_model = models[best_model_idx]
print('Best Model:', best_model)


# In[337]:


# Compare model performance after hypertuning
models = ['KNN', 'LR', 'DT', 'RF']
accuracies = [newacc_knn, newacc_lr, newacc_dt, newacc_rf]

plt.bar(models, accuracies)
plt.title('Models accuracy after hypertuning')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
# plt.color("#A82548")
plt.show()

# Select the best-performing model
best_model_idx = accuracies.index(max(accuracies))
best_model = models[best_model_idx]
print('Best Model:', best_model)


# In[338]:


y_pred_knn = knn.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Confusion matrix for all models

plt.figure(figsize=(4,4),dpi=150)
confusion_matrix_knn = confusion_matrix(y_test,y_pred_knn)
sns.heatmap(confusion_matrix_knn,annot=True,xticklabels=['Positive','Negative'], yticklabels=[ 'Positive','Negative'],
           cmap='Blues',fmt='d')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("K-Neighbors Classifier")
plt.show()


# In[339]:


plt.figure(figsize=(4,4),dpi=150)
confusion_matrix_lr = confusion_matrix(y_test,y_pred_lr)
sns.heatmap(confusion_matrix_lr, annot=True, xticklabels=['Positive','Negative'], yticklabels=[ 'Positive','Negative'],
           cmap='BuPu',fmt='d')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("Logistic regression")
plt.show()


# In[340]:


plt.figure(figsize=(4,4),dpi=150)
confusion_matrix_dt = confusion_matrix(y_test,y_pred_dt)
sns.heatmap(confusion_matrix_dt, annot=True, xticklabels=['Positive','Negative'], yticklabels=[ 'Positive','Negative'],
           cmap='Greens',fmt='d')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("Decision Tree")
plt.show()


# In[341]:


plt.figure(figsize=(4,4),dpi=150)
confusion_matrix_rf = confusion_matrix(y_test,y_pred_rf)
sns.heatmap(confusion_matrix_rf, annot=True, xticklabels=['Positive','Negative'], yticklabels=[ 'Positive','Negative'],
           cmap='Purples',fmt='d')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("Random Forest")
plt.show()

