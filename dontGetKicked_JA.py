
# coding: utf-8

# # "Don't Get Kicked": a classification practice
# **Jamal Alikhani**    
# jamal.alikhani@gmail.com    
# July 2017  
# 
# Dealerships buy used cars in large scale to get benefit after reselling them to new customers. However, they sometimes mistakenly buy cars with major issues that prevent them from reselling these cars to customers, which is slanged as "kicked" cars.     
# 
# The goal of this practice is to develop a predictive model that helps dealerships to detect “kicked” cars before purchasing them. For this reason, data are downloaded from kaggle.com competition called "Don't get kicked" as same as this report's title.    
# 
# The dataset is uploaded to the Python3 environment, and after preprocessing and data cleaning phases, the scikit-learn tool has been used to provide three classification models based on logistic regression, bagging ensemble, and boosting ensemble approaches. The Python codes plus a short description is documented as below.    

# In[2]:


import pandas as pd
import numpy as np

# classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

# Selected classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 

# Cross validation models
from sklearn.preprocessing import LabelEncoder, normalize, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score


# # Data Preparation
# ## Loading the training dataset

# In[3]:


train = pd.read_csv("training.csv")
print("Origional size of the training set: ", train.shape)


# In[4]:


pd.set_option('display.max_columns', 50)
print("A preview of training dataset prior data processing:")
train.head()


# #### Dropping redundant and none-informative attributes:
# 
# The none informative attributes like IDs and dates, redundant attributes like zipcode and "VehYear" while "VehicleAge"     
# is provided with the same information, and attributes with none-atomic contents like "Model" and "SubModel" are dropped.     
# 
# It should be pointed out that attributes with none-atomic members can be kept and their information can be retrieved by natural language processing techniques (like *nltk*). However, for this practice, they are just simply dropped out.

# In[5]:


# dropping none informative (redundant) attributes
train.drop(['RefId', 'PurchDate', 'VehYear', 'WheelType', 'VNZIP1','SubModel', 'Model'], axis=1, inplace =True)


# #### Checking the number of missing data in each attribute:

# In[6]:


print('Number of NA/NaNs in each attribute:')
train.isnull().sum(axis=0)


# #### Dropping the attributes with high number of missing values:

# In[7]:


train.drop(['PRIMEUNIT', 'AUCGUART'], axis=1, inplace =True)


# The missing cells of 'WheelTypeID' and 'Trim' attributes are filled with some out-of-range values to keep around 3,200 rows. The classification algorithm can distinguish this out-of-range values in their model. For the rest of the missing value, the rows associated with each missing value is droppd.

# In[8]:


train['WheelTypeID'].fillna(-99, axis=0, inplace=True)
train['Trim'].fillna('?', axis=0, inplace=True)
train.dropna(axis=0, inplace=True)
print("Size of training set after trimming: ", train.shape)


# ### One-hot encoding of categorical data to dymmy features: 

# In[14]:


dt = train['Auction'].dtypes 
for col in train.columns: 
    if str(train[col].dtypes) == 'object':  
        train = pd.get_dummies(train, columns=[col])              


# In[15]:


print("A preview of training dataset after data processing:")
train.head()


# ### Splitting label attribute from training set

# In[16]:


train = np.array(train,dtype='float')
X = train[:,1:]
y = train[:,0]
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)


# # Classification
# 
# In this practice, we are facing with an unbalanced labeled dataset where class "0" are 87.7% and class "1" (kicks) are just 12.3%! This makes the classification training phase tricky because the accuracy of random selection is already a high value (87.7%). Therefore, accuracy is not a proper metric for this case to be evaluated. 
# 
# The important part of the model prediction is to reduce the number of kicked cars. Therefore, we have to increase the rate of true negative (TN) while decreasing the false negative (FN) rates. By looking at *precision*, *recall*, *F1-score*, and *Area under the ROC curve (ROC-AUC)* we can get a better insight into the classification performance.

# In[17]:


print("Ratio of class 1 to class 0 in the training set:")
np.round(np.mean(y), 3)


# ### *k*-fold cross validation model for classification training score

# In[18]:


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# 
# ## Logistic Regression
# To overrcome the unbalanced lables, the “balanced” mode is activated for equally weighting the binary classes in the calculation of the loss function. The balanced mode automatically adjusts weights inversely proportional to class frequencies in the input data as:    
#                                   `n_samples / (n_classes * np.bincount(y))`

# In[19]:


clf = LogisticRegression(class_weight='balanced')

scoring = 'f1'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Logistic Regression,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))


# In[20]:


scoring = 'roc_auc'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Logistic Regression,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))


# In[21]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Logistic Regression, confusion matrix: ")
print(conf_mat)


# In[22]:


acc = accuracy_score(y_test, y_pred)
print("Logistic Regression, accuracy: {0:.1%}".format(acc))


# In[23]:


print("Logistic Regression, summary report: ")
print(classification_report(y_test, y_pred))


# f1-score and ROC-AUC metrics show reasonably good results for the first try, regardless of seemingly low accuracy of 75.5%! It is already mentioned that accuracy is not a good measure for this case. Values in the confusion matrix show that model was able to correctly predict the 58% of cars with major issues.  

# ##  Random Forest Classifier
# Random forest classifier uses bagging technique to construct many parallel decision trees over bootstrapped resampling of the training dataset to reduce the variance. Presence of several categorial attributes (like "Transmission" and "WheelTypeID") makes it suitable to apply decision tree as the base classifier. 
# 
# To overcome the overfitting in the decision tree, a kfold cross validation technique over a grid search is applied to find the optimum depth of the trees. It is shown that 7 is the optimum depth.

# In[24]:


# Random Forest Classifier
n_depth = [2, 5, 7, 10, 15, 20]

pipeline_estimator = Pipeline([('clf', RandomForestClassifier(class_weight='balanced'))])

params = [{'clf__max_depth': n_depth}]

grid = GridSearchCV(estimator=pipeline_estimator, param_grid=params, scoring='f1', cv=3, n_jobs=-1)

grid.fit(X, y)
mean_scores = np.array(grid.cv_results_['mean_test_score'])
reults_df = pd.DataFrame()
reults_df['n_depth'] = n_depth
reults_df['f1_score'] = mean_scores
print("Random Forest Classifier, Grid Search result:")
print(reults_df)


# In[25]:


opt_depth = n_depth[np.argmax(mean_scores)]
clf = RandomForestClassifier(max_depth=opt_depth, n_estimators=100, class_weight='balanced')

scoring = 'roc_auc'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Random Forest Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))


# In[26]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Random Forest Classifier, confusion matrix: ")
print(conf_mat)


# In[27]:


acc = accuracy_score(y_test, y_pred)
print("Random Forest Classifier, accuracy: {0:.1%}".format(acc))


# In[28]:


print("Random Forest Classifier, summary report: ")
print(classification_report(y_test, y_pred))


# Both the ROC-AUC factor (0.761) and overal f1-score (0.82) are slightly better than the results obtained from logistic regression. However, just by looking at the TN rate (recall) from the confusion matrix, logistic regression finds a higher value of 58% compared to the 51% obtained by the Random Forest.

# ## AdaBoost Classifier
# AdaBoost is an adaptive boosting ensemble that collects many weak classifiers into a strong classifier. Similar to the Random Forest, a decision tree is again used as the base classifier. 
# 
# Since in the scikit-learn package this algorithm does not support the weighted classes, the majority class is "down-sampled" to get a size equal to the minority class. Afterward, the balanced subsamples of the training data is fed into the model training phase of the AdaBoost. 

# In[29]:


Xt_c1 = X_train[y_train==1] 
yt_c1 = y_train[y_train==1] 

Xt_c0 = X_train[y_train==0] 
yt_c0 = y_train[y_train==0]
Xt_c0 = Xt_c0[:Xt_c1.shape[0],:]
yt_c0 = yt_c0[:Xt_c1.shape[0]]

X_train_balanced = np.concatenate((Xt_c0, Xt_c1), axis=0)
y_train_balanced = np.concatenate((yt_c0, yt_c1), axis=0)


# In[30]:


clf = AdaBoostClassifier(n_estimators=100, random_state=seed)

scoring = 'f1'
results = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=kfold, scoring=scoring, n_jobs=-1)
print("AdaBoost Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))


# In[31]:


scoring = 'roc_auc'
results = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=kfold, scoring=scoring, n_jobs=-1)
print("AdaBoost Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))


# In[32]:


clf.fit(X_train_balanced, y_train_balanced)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("AdaBoost Classifier, confusion matrix: ")
print(conf_mat)


# In[33]:


acc = accuracy_score(y_test, y_pred)
print("AdaBoost Classifier, accuracy: {0:.1%}".format(acc))


# In[34]:


print("AdaBoost Classifier, summary report: ")
print(classification_report(y_test, y_pred))


# The AdaBoost performs better in recall percentage compare to other two clasiifiers. The f1-score and the ROC-AUC are just slightly lower than random forest. 

# # Conclusion
# 
# Three classification models are applied to predict the cars with major issues referred to as kicked cars:
# 
# |classifier| recall over kicked cars | overall f1-score |
# |---------|-------|---------|
# |Logistic Regression|0.60|0.78|
# |Random Forest|0.44|0.84|
# |AdaBoost|0.62|0.76|
# 
# It is concluded that the accuracy is a poor metric to evaluate the model performance due to the heavily unbalanced labels in the binary classes. True negative classification rate or "Recall" is considered to be a target model performance. The TN prevents car dealerships from purchasing a kicked. On the other hand, there is a trade-off between the recall and the precision, as when the rate of false negative (FN) increases, the company loses the opportunity to purchase a good car that could potentially benefit the company. The financial objective function can be constructed as:
# 
#                              total_profit = (TP-FN)*benefit + (FP-TN)*Loss
# 
# where *benefit* and *loss* are the positive and negative benefit from each good and bad car, respectively. By maximizing this objective function, the optimum trade-off between the FN and TN rates can be obtained and then a best predictive model can be selected and trained. 
# 
# Other classifications can also be tested and may perform better in this case.
# 
