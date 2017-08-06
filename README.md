
# "Don't Get Kicked": a classification practice
**Jamal Alikhani**    
jamal.alikhani@gmail.com    
July 2017  

Dealerships buy used cars in large scale to get benefit after reselling them to new customers. However, they sometimes mistakenly buy cars with major issues that prevent them from reselling these cars to customers, which is slanged as "kicked" cars.     

The goal of this practice is to develop a predictive model that helps dealerships to detect “kicked” cars before purchasing them. For this reason, data are downloaded from kaggle.com competition called "Don't get kicked" as same as this report's title.    

The dataset is uploaded to the Python3 environment, and after preprocessing and data cleaning phases, the scikit-learn tool has been used to provide three classification models based on logistic regression, bagging ensemble, and boosting ensemble approaches. The Python codes plus a short description is documented as below.    


```python
import pandas as pd
import numpy as np

# classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

# Selected classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 

# Cross validation models
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
```

# Data Preparation
## Loading the training dataset


```python
train = pd.read_csv("training.csv")
print("Origional size of the training set: ", train.shape)
```

    Origional size of the training set:  (72983, 34)



```python
pd.set_option('display.max_columns', 50)
print("A preview of training dataset prior data processing:")
train.head()
```

    A preview of training dataset prior data processing:





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RefId</th>
      <th>IsBadBuy</th>
      <th>PurchDate</th>
      <th>Auction</th>
      <th>VehYear</th>
      <th>VehicleAge</th>
      <th>Make</th>
      <th>Model</th>
      <th>Trim</th>
      <th>SubModel</th>
      <th>Color</th>
      <th>Transmission</th>
      <th>WheelTypeID</th>
      <th>WheelType</th>
      <th>VehOdo</th>
      <th>Nationality</th>
      <th>Size</th>
      <th>TopThreeAmericanName</th>
      <th>MMRAcquisitionAuctionAveragePrice</th>
      <th>MMRAcquisitionAuctionCleanPrice</th>
      <th>MMRAcquisitionRetailAveragePrice</th>
      <th>MMRAcquisitonRetailCleanPrice</th>
      <th>MMRCurrentAuctionAveragePrice</th>
      <th>MMRCurrentAuctionCleanPrice</th>
      <th>MMRCurrentRetailAveragePrice</th>
      <th>MMRCurrentRetailCleanPrice</th>
      <th>PRIMEUNIT</th>
      <th>AUCGUART</th>
      <th>BYRNO</th>
      <th>VNZIP1</th>
      <th>VNST</th>
      <th>VehBCost</th>
      <th>IsOnlineSale</th>
      <th>WarrantyCost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>12/7/2009</td>
      <td>ADESA</td>
      <td>2006</td>
      <td>3</td>
      <td>MAZDA</td>
      <td>MAZDA3</td>
      <td>i</td>
      <td>4D SEDAN I</td>
      <td>RED</td>
      <td>AUTO</td>
      <td>1</td>
      <td>Alloy</td>
      <td>89046</td>
      <td>OTHER ASIAN</td>
      <td>MEDIUM</td>
      <td>OTHER</td>
      <td>8155</td>
      <td>9829</td>
      <td>11636</td>
      <td>13600</td>
      <td>7451</td>
      <td>8552</td>
      <td>11597</td>
      <td>12409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21973</td>
      <td>33619</td>
      <td>FL</td>
      <td>7100</td>
      <td>0</td>
      <td>1113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>12/7/2009</td>
      <td>ADESA</td>
      <td>2004</td>
      <td>5</td>
      <td>DODGE</td>
      <td>1500 RAM PICKUP 2WD</td>
      <td>ST</td>
      <td>QUAD CAB 4.7L SLT</td>
      <td>WHITE</td>
      <td>AUTO</td>
      <td>1</td>
      <td>Alloy</td>
      <td>93593</td>
      <td>AMERICAN</td>
      <td>LARGE TRUCK</td>
      <td>CHRYSLER</td>
      <td>6854</td>
      <td>8383</td>
      <td>10897</td>
      <td>12572</td>
      <td>7456</td>
      <td>9222</td>
      <td>11374</td>
      <td>12791</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19638</td>
      <td>33619</td>
      <td>FL</td>
      <td>7600</td>
      <td>0</td>
      <td>1053</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>12/7/2009</td>
      <td>ADESA</td>
      <td>2005</td>
      <td>4</td>
      <td>DODGE</td>
      <td>STRATUS V6</td>
      <td>SXT</td>
      <td>4D SEDAN SXT FFV</td>
      <td>MAROON</td>
      <td>AUTO</td>
      <td>2</td>
      <td>Covers</td>
      <td>73807</td>
      <td>AMERICAN</td>
      <td>MEDIUM</td>
      <td>CHRYSLER</td>
      <td>3202</td>
      <td>4760</td>
      <td>6943</td>
      <td>8457</td>
      <td>4035</td>
      <td>5557</td>
      <td>7146</td>
      <td>8702</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19638</td>
      <td>33619</td>
      <td>FL</td>
      <td>4900</td>
      <td>0</td>
      <td>1389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>12/7/2009</td>
      <td>ADESA</td>
      <td>2004</td>
      <td>5</td>
      <td>DODGE</td>
      <td>NEON</td>
      <td>SXT</td>
      <td>4D SEDAN</td>
      <td>SILVER</td>
      <td>AUTO</td>
      <td>1</td>
      <td>Alloy</td>
      <td>65617</td>
      <td>AMERICAN</td>
      <td>COMPACT</td>
      <td>CHRYSLER</td>
      <td>1893</td>
      <td>2675</td>
      <td>4658</td>
      <td>5690</td>
      <td>1844</td>
      <td>2646</td>
      <td>4375</td>
      <td>5518</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19638</td>
      <td>33619</td>
      <td>FL</td>
      <td>4100</td>
      <td>0</td>
      <td>630</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>12/7/2009</td>
      <td>ADESA</td>
      <td>2005</td>
      <td>4</td>
      <td>FORD</td>
      <td>FOCUS</td>
      <td>ZX3</td>
      <td>2D COUPE ZX3</td>
      <td>SILVER</td>
      <td>MANUAL</td>
      <td>2</td>
      <td>Covers</td>
      <td>69367</td>
      <td>AMERICAN</td>
      <td>COMPACT</td>
      <td>FORD</td>
      <td>3913</td>
      <td>5054</td>
      <td>7723</td>
      <td>8707</td>
      <td>3247</td>
      <td>4384</td>
      <td>6739</td>
      <td>7911</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19638</td>
      <td>33619</td>
      <td>FL</td>
      <td>4000</td>
      <td>0</td>
      <td>1020</td>
    </tr>
  </tbody>
</table>
</div>



#### Dropping redundant and none-informative attributes:

The none informative attributes like IDs and dates, redundant attributes like zipcode and "VehYear" while "VehicleAge"     
is provided with the same information, and attributes with none-atomic contents like "Model" and "SubModel" are dropped.     

It should be pointed out that attributes with none-atomic members can be kept and their information can be retrieved by natural language processing techniques (like *nltk*). However, for this practice, they are just simply dropped out.


```python
# dropping none informative (redundant) attributes
train.drop(['RefId', 'PurchDate', 'VehYear', 'WheelType', 'VNZIP1','SubModel', 'Model'], axis=1, inplace =True)
```

#### Checking the number of missing data in each attribute:


```python
print('Number of NA/NaNs in each attribute:')
train.isnull().sum(axis=0)
```

    Number of NA/NaNs in each attribute:





    IsBadBuy                                 0
    Auction                                  0
    VehicleAge                               0
    Make                                     0
    Trim                                  2360
    Color                                    8
    Transmission                             9
    WheelTypeID                           3169
    VehOdo                                   0
    Nationality                              5
    Size                                     5
    TopThreeAmericanName                     5
    MMRAcquisitionAuctionAveragePrice       18
    MMRAcquisitionAuctionCleanPrice         18
    MMRAcquisitionRetailAveragePrice        18
    MMRAcquisitonRetailCleanPrice           18
    MMRCurrentAuctionAveragePrice          315
    MMRCurrentAuctionCleanPrice            315
    MMRCurrentRetailAveragePrice           315
    MMRCurrentRetailCleanPrice             315
    PRIMEUNIT                            69564
    AUCGUART                             69564
    BYRNO                                    0
    VNST                                     0
    VehBCost                                 0
    IsOnlineSale                             0
    WarrantyCost                             0
    dtype: int64



#### Dropping the attributes with high number of missing values:


```python
train.drop(['PRIMEUNIT', 'AUCGUART'], axis=1, inplace =True)
```

The missing cells of 'WheelTypeID' and 'Trim' attributes are filled with some out-of-range values to keep around 3,200 rows. The classification algorithm can distinguish this out-of-range values in their model. For the rest of the missing value, the rows associated with each missing value is droppd.


```python
train['WheelTypeID'].fillna(-99, axis=0, inplace=True)
train['Trim'].fillna('?', axis=0, inplace=True)
train.dropna(axis=0, inplace=True)
print("Size of training set after trimming: ", train.shape)
```

    Size of training set after trimming:  (72658, 25)


### Transfering none-numeric categorial attributes to numeric categories:


```python
le = LabelEncoder()
dt = train['Auction'].dtypes    
for col in train.columns: 
    if str(train[col].dtypes) == 'object':  
        le.fit(train[col])
        train[col] = le.transform(train[col]) + 1        
```


```python
print("A preview of training dataset after data processing:")
train.head()
```

    A preview of training dataset after data processing:





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IsBadBuy</th>
      <th>Auction</th>
      <th>VehicleAge</th>
      <th>Make</th>
      <th>Trim</th>
      <th>Color</th>
      <th>Transmission</th>
      <th>WheelTypeID</th>
      <th>VehOdo</th>
      <th>Nationality</th>
      <th>Size</th>
      <th>TopThreeAmericanName</th>
      <th>MMRAcquisitionAuctionAveragePrice</th>
      <th>MMRAcquisitionAuctionCleanPrice</th>
      <th>MMRAcquisitionRetailAveragePrice</th>
      <th>MMRAcquisitonRetailCleanPrice</th>
      <th>MMRCurrentAuctionAveragePrice</th>
      <th>MMRCurrentAuctionCleanPrice</th>
      <th>MMRCurrentRetailAveragePrice</th>
      <th>MMRCurrentRetailCleanPrice</th>
      <th>BYRNO</th>
      <th>VNST</th>
      <th>VehBCost</th>
      <th>IsOnlineSale</th>
      <th>WarrantyCost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>18</td>
      <td>134</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>89046</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>8155</td>
      <td>9829</td>
      <td>11636</td>
      <td>13600</td>
      <td>7451</td>
      <td>8552</td>
      <td>11597</td>
      <td>12409</td>
      <td>21973</td>
      <td>6</td>
      <td>7100</td>
      <td>0</td>
      <td>1113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>95</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>93593</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6854</td>
      <td>8383</td>
      <td>10897</td>
      <td>12572</td>
      <td>7456</td>
      <td>9222</td>
      <td>11374</td>
      <td>12791</td>
      <td>19638</td>
      <td>6</td>
      <td>7600</td>
      <td>0</td>
      <td>1053</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>100</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>73807</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>3202</td>
      <td>4760</td>
      <td>6943</td>
      <td>8457</td>
      <td>4035</td>
      <td>5557</td>
      <td>7146</td>
      <td>8702</td>
      <td>19638</td>
      <td>6</td>
      <td>4900</td>
      <td>0</td>
      <td>1389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>100</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>65617</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1893</td>
      <td>2675</td>
      <td>4658</td>
      <td>5690</td>
      <td>1844</td>
      <td>2646</td>
      <td>4375</td>
      <td>5518</td>
      <td>19638</td>
      <td>6</td>
      <td>4100</td>
      <td>0</td>
      <td>630</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>128</td>
      <td>14</td>
      <td>2</td>
      <td>2</td>
      <td>69367</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3913</td>
      <td>5054</td>
      <td>7723</td>
      <td>8707</td>
      <td>3247</td>
      <td>4384</td>
      <td>6739</td>
      <td>7911</td>
      <td>19638</td>
      <td>6</td>
      <td>4000</td>
      <td>0</td>
      <td>1020</td>
    </tr>
  </tbody>
</table>
</div>



### Splitting label attribute from training set


```python
train = np.array(train,dtype='float')
X = train[:,1:]
y = train[:,0]
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
```

# Classification

In this practice, we are facing with an unbalanced labeled dataset where class "0" are 87.7% and class "1" (kicks) are just 12.3%! This makes the classification training phase tricky because the accuracy of random selection is already a high value (87.7%). Therefore, accuracy is not a proper metric for this case to be evaluated. 

The important part of the model prediction is to reduce the number of kicked cars. Therefore, we have to increase the rate of true negative (TN) while decreasing the false negative (FN) rates. By looking at *precision*, *recall*, *F1-score*, and *Area under the ROC curve (ROC-AUC)* we can get a better insight into the classification performance.


```python
print("Ratio of class 1 to class 0 in the training set:")
np.round(np.mean(y), 3)
```

    Ratio of class 1 to class 0 in the training set:





    0.123



### *k*-fold cross validation model for classification training score


```python
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
```


## Logistic Regression
To overrcome the unbalanced lables, the “balanced” mode is activated for equally weighting the binary classes in the calculation of the loss function. The balanced mode automatically adjusts weights inversely proportional to class frequencies in the input data as:    
                                  `n_samples / (n_classes * np.bincount(y))`


```python
clf = LogisticRegression(class_weight='balanced')

scoring = 'f1'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Logistic Regression,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))
```

    Logistic Regression, f1 : 0.362, (std: 0.00938)



```python
scoring = 'roc_auc'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Logistic Regression,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))
```

    Logistic Regression, roc_auc : 0.746, (std: 0.00697)



```python
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Logistic Regression, confusion matrix: ")
print(conf_mat)
```

    Logistic Regression, confusion matrix: 
    [[16382  4648]
     [ 1231  1717]]



```python
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression, accuracy: {0:.1%}".format(acc))
```

    Logistic Regression, accuracy: 75.5%



```python
print("Logistic Regression, summary report: ")
print(classification_report(y_test, y_pred))
```

    Logistic Regression, summary report: 
                 precision    recall  f1-score   support
    
            0.0       0.93      0.78      0.85     21030
            1.0       0.27      0.58      0.37      2948
    
    avg / total       0.85      0.75      0.79     23978
    


f1-score and ROC-AUC metrics show reasonably good results for the first try, regardless of seemingly low accuracy of 75.5%! It is already mentioned that accuracy is not a good measure for this case. Values in the confusion matrix show that model was able to correctly predict the 58% of cars with major issues.  

##  Random Forest Classifier
Random forest classifier uses bagging technique to construct many parallel decision trees over bootstrapped resampling of the training dataset to reduce the variance. Presence of several categorial attributes (like "Transmission" and "WheelTypeID") makes it suitable to apply decision tree as the base classifier. 

To overcome the overfitting in the decision tree, a kfold cross validation technique over a grid search is applied to find the optimum depth of the trees. It is shown that 7 is the optimum depth.


```python
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
```

    Random Forest Classifier, Grid Search result:
       n_depth  f1_score
    0        2  0.328922
    1        5  0.342690
    2        7  0.355611
    3       10  0.360597
    4       15  0.342642
    5       20  0.296646



```python
opt_depth = n_depth[np.argmax(mean_scores)]
clf = RandomForestClassifier(max_depth=opt_depth, n_estimators=100, class_weight='balanced')

scoring = 'roc_auc'
results = cross_val_score(clf, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
print("Random Forest Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))
```

    Random Forest Classifier, roc_auc : 0.762, (std: 0.00551)



```python
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Random Forest Classifier, confusion matrix: ")
print(conf_mat)
```

    Random Forest Classifier, confusion matrix: 
    [[17785  3245]
     [ 1454  1494]]



```python
acc = accuracy_score(y_test, y_pred)
print("Random Forest Classifier, accuracy: {0:.1%}".format(acc))
```

    Random Forest Classifier, accuracy: 80.4%



```python
print("Random Forest Classifier, summary report: ")
print(classification_report(y_test, y_pred))
```

    Random Forest Classifier, summary report: 
                 precision    recall  f1-score   support
    
            0.0       0.92      0.85      0.88     21030
            1.0       0.32      0.51      0.39      2948
    
    avg / total       0.85      0.80      0.82     23978
    


Both the ROC-AUC factor (0.761) and overal f1-score (0.82) are slightly better than the results obtained from logistic regression. However, just by looking at the TN rate (recall) from the confusion matrix, logistic regression finds a higher value of 58% compared to the 51% obtained by the Random Forest.

## AdaBoost Classifier
AdaBoost is an adaptive boosting ensemble that collects many weak classifiers into a strong classifier. Similar to the Random Forest, a decision tree is again used as the base classifier. 

Since in the scikit-learn package this algorithm does not support the weighted classes, the majority class is "down-sampled" to get a size equal to the minority class. Afterward, the balanced subsamples of the training data is fed into the model training phase of the AdaBoost. 


```python
Xt_c1 = X_train[y_train==1] 
yt_c1 = y_train[y_train==1] 

Xt_c0 = X_train[y_train==0] 
yt_c0 = y_train[y_train==0]
Xt_c0 = Xt_c0[:Xt_c1.shape[0],:]
yt_c0 = yt_c0[:Xt_c1.shape[0]]

X_train_balanced = np.concatenate((Xt_c0, Xt_c1), axis=0)
y_train_balanced = np.concatenate((yt_c0, yt_c1), axis=0)
```


```python
clf = AdaBoostClassifier(n_estimators=100, random_state=seed)

scoring = 'f1'
results = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=kfold, scoring=scoring, n_jobs=-1)
print("AdaBoost Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))
```

    AdaBoost Classifier, f1 : 0.659, (std: 0.012)



```python
scoring = 'roc_auc'
results = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=kfold, scoring=scoring, n_jobs=-1)
print("AdaBoost Classifier,",scoring, ": {0:.3}, (std: {1:.3})".format(results.mean(), results.std()))
```

    AdaBoost Classifier, roc_auc : 0.753, (std: 0.00952)



```python
clf.fit(X_train_balanced, y_train_balanced)
y_pred = clf.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("AdaBoost Classifier, confusion matrix: ")
print(conf_mat)
```

    AdaBoost Classifier, confusion matrix: 
    [[15495  5535]
     [ 1129  1819]]



```python
acc = accuracy_score(y_test, y_pred)
print("AdaBoost Classifier, accuracy: {0:.1%}".format(acc))
```

    AdaBoost Classifier, accuracy: 72.2%



```python
print("AdaBoost Classifier, summary report: ")
print(classification_report(y_test, y_pred))
```

    AdaBoost Classifier, summary report: 
                 precision    recall  f1-score   support
    
            0.0       0.93      0.74      0.82     21030
            1.0       0.25      0.62      0.35      2948
    
    avg / total       0.85      0.72      0.77     23978
    


The AdaBoost performs better in recall percentage compare to other two clasiifiers. The f1-score and the ROC-AUC are just slightly lower than random forest. 

# Conclusion

Three classification model is applied to predict the cars with major issues referred to as kicked cars:

|classifier| recall over kicked cars | overall f1-score |
|---------|-------|---------|
|Logistic Regression|0.58|0.79|
|Random Forest|0.51|0.82|
|AdaBoost|0.62|0.77|

It is concluded that the accuracy is a poor metric to evaluate the model performance due to the heavily unbalanced labels in the binary classes. True negative classification rate or "Recall" is considered to be a target model performance. The TN prevents car dealerships from purchasing a kicked. On the other hand, there is a trade-off between the recall and the precision, as when the rate of false negative (FN) increases, the company loses the opportunity to purchase a good car that could potentially benefit the company. The financial objective function can be constructed as:

                             total_profit = (TP-FN)*benefit + (FP-TN)*Loss

where *benefit* and *loss* are the positive and negative benefit from each good and bad car, respectively. By maximizing this objective function, the optimum trade-off between the FN and TN rates can be obtained and then a best predictive model can be selected and trained. 

Other classifications can also be tested and may perform better in this case.

# Refernce and Data Source
Data and project idea are obtained from a kaggle project as:
[https://www.kaggle.com/c/DontGetKicked] (https://www.kaggle.com/c/DontGetKicked)
