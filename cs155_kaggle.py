#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np
from pandas import DataFrame as df
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder, OneHotEncoder,QuantileTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer




grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}

emp_mapping = {'1 year': 1,
 '10+ years': 10,
 '2 years': 2,
 '3 years': 3,
 '4 years': 4,
 '5 years': 5,
 '6 years': 6,
 '7 years': 7,
 '8 years': 8,
 '9 years': 9,
 '< 1 year': 0.5}


def prepare_data(filename, drop_cols, percent_cols, date_cols, test=False):
    data = pd.read_csv(filename, index_col='id')

    for col in drop_cols:
        data.drop(col, axis=1, inplace=True) 

    for col in percent_cols:
        data[col] = pd.to_numeric(data[col].str.strip('%')).div(100)

    data = data.replace({"grade": grade_mapping})
    data = data.replace({"emp_length": emp_mapping})


    for col in date_cols:
        data[col] = pd.to_numeric(data[col].str[4:])

    if not test: 
        X = data.iloc[:,:-1]

        y = data.iloc[:, -1]
        return X,y

    else:
        X = data

        return X, _

def make_balanced_test_set(filename, drop_cols, percent_cols, date_cols, test=False):
    data = pd.read_csv(filename, index_col='id')

    for col in drop_cols:
        data.drop(col, axis=1, inplace=True) 

    for col in percent_cols:
        data[col] = pd.to_numeric(data[col].str.strip('%')).div(100)

    data = data.replace({"grade": grade_mapping})
    data = data.replace({"emp_length": emp_mapping})


    for col in date_cols:
        data[col] = pd.to_numeric(data[col].str[4:])

    X_zeros = data[data['loan_status'] == 'Fully Paid']
    X_ones = data[data['loan_status'] == 'Charged Off']
    balanced = (X_ones.iloc[:30000].append(X_ones.iloc[:30000]))

    X = balanced.iloc[:,:-1]
    y = balanced.iloc[:, -1]

    return X, y


# In[23]:


# names of columns to drop 
drop = ['sub_grade',
        'emp_title',
        'title',
        'zip_code',
        'mort_acc',
        'application_type', 
       'verification_status']

# names of percent columns lol
percent = ['int_rate', 'revol_util']

# cols with dates to converts to ints
date = ['earliest_cr_line', 'issue_d']

X_train, y_train = prepare_data('desktop/LOANS_TRAIN.csv', drop, percent, date, test=False)

# optional code to split training set into a dummy test set 
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=34534)

X_test_real, _ = prepare_data('desktop/LOANS_TEST.csv', drop, percent, date, test=True)

# balanced validation set to help test models
balanced_X, balanced_y = make_balanced_test_set('desktop/LOANS_TRAIN.csv', drop, percent, date, test=False)


# In[24]:


# models we have tried in pipe

lgr = LogisticRegression(C=1e-3, solver='saga', max_iter= 1000, random_state = 2222, 
                         class_weight = {'Fully Paid': 1, 'Charged Off': 6})
# sgd = SGDClassifier(loss = 'log',max_iter=1000, tol=1e-3, early_stopping = True)
# lgrcv = LogisticRegressionCV(max_iter= 9)
rf = RandomForestClassifier(n_estimators=5, random_state=422, max_depth = 12)
# clf = MLPClassifier(
#                     alpha=1e-5,
#                     solver = 'sgd',
#                     hidden_layer_sizes=(50, 30, 25, 10,),
#                     activation = 'logistic', 
#                     batch_size = 50,
#                     early_stopping = True, 
#                     random_state=123, 
#                     max_iter=1000)


# can stack models in ensemble
models = [('rf',rf),('lgr', lgr)]
stacking = StackingClassifier(estimators=models)


# In[25]:


num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()), ('poly',PolynomialFeatures(degree = 2))
    , ('scaler', StandardScaler())
    ])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_features), 
                                                 ('cat', cat_transformer, cat_features)])


steps=[('preprocessor', preprocessor),('classifier', lgr)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)

train_err = pipe.score(X_train, y_train)
balanced_val_error = pipe.score(balanced_X, balanced_y)

print(train_err)
print(balanced_val_error)


# In[26]:


#code to output submisson
predictions = pipe.predict_proba(X_test_real)[:,0]
ids = X_test_real.index
df = pd.DataFrame({'id': ids, 'loan_status': predictions})
out = df.to_csv('newsubmission10.csv', index=False) 


# In[27]:


data = pd.read_csv(filename, index_col='id')


# In[ ]:


def errors(num):
  x = data[:num, 0]
  y = data[:num, 1]

  train_err = 0
  validation_err = 0

  for train_index, test_index in kf.split(x):
    
    # Training and testing data points for this fold:
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    fit = np.polyfit(x_train, y_train, degree)

    temp_train = np.mean((np.polyval(fit, x_train) - y_train) ** 2)
    train_err += temp_train
    average_train = train_err/num_folds

    temp_valid = np.mean((np.polyval(fit, x_test) - y_test) ** 2)
    validation_err += temp_valid
    average_valid = validation_err / num_folds;

  return average_train, average_valid


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

error = np.matrix([errors(num) for num in np.arange(20, 101, 5)])
train_err = error[:, 0]
validation_err = error[:, 1]

plt.plot(np.arange(20, 101, 5), train_err)
plt.plot(np.arange(20, 101, 5), validation_err)


plt.xlabel('data points')
plt.ylabel('mean squared error')
plt.legend('Training error', 'Validation error')
plt.title('Polynomial Degree ')

plt.show()


# In[ ]:





# In[ ]:




