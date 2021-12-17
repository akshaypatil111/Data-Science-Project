#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import missingno as msno
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read the Dataset
data = pd.read_csv('Myocardial infarction complications.csv')
data.head(10)


# In[3]:


pd.set_option('display.max_column', None)
data.columns


# In[4]:


data.info()


# In[5]:


data.dtypes


# In[6]:


data.describe(include='all').T


# In[7]:


null_values = pd.DataFrame(data.isnull().sum())
null_values.T


# In[8]:


for column_name in data.columns:
    print(column_name, data[column_name].isna().sum())


# In[9]:


for col in data.columns:
    pct_missing = data[col].isnull().sum()
    print(f'{col} - {pct_missing :.1%}')


# In[10]:


# Missing Value
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[11]:


msno.bar(data, figsize=(18, 55), fontsize=12, color='steelblue')


# #for columns IBS_NASL and KFK_BLOOD among 1700 datavalues 1628 and 1696 are missing respectively. I.e 95.8 % and 99.76% of data is missing. As there is no much information present we are removing this both columns

# In[12]:


data.drop(['ID', 'KFK_BLOOD', 'IBS_NASL'], axis=1, inplace=True)


# In[13]:


data.describe()


# In[14]:


# Keep all those columns with 70% Non-na values
data1 = data.dropna(axis=1, thresh=0.7 * len(data))
data1


# In[15]:


# Missing values Imputation
# 1. Imputing categorical or binary data with the mode
# 2. Imputing numerical data with median


# In[16]:


# creating a list of numerical columns
numericCols = [
    'AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'NA_BLOOD',
    'ROE', 'K_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD'
]


# In[17]:


# Iterating the loop for each column in the given dataset
def missing_val(data1):
    for column_name in data1.columns:
        # checking if the column is a numerical column or not.i.e if it exist in the numerical col or not,if exists value > 1 else 0
        exist_count = numericCols.count(column_name)
        if exist_count > 0:
            data1[column_name].fillna(data1[column_name].median(),
                                      inplace=True)
        else:
            data1[column_name].fillna(data1[column_name].mode(), inplace=True)
            if (data1[column_name].isna().sum() > 0):
                data1[column_name].fillna(0, inplace=True)
            else:
                pass


# In[18]:


# checking missing values


# In[19]:


missing_val(data)


# In[20]:


for column_name in data.columns:
    print(column_name, data[column_name].isna().sum())


# In[21]:


# All the missing values are imputed
# Checking for duplicate records


# In[22]:


data1[data1.duplicated()]


# In[23]:


# No Duplicates Found


# In[24]:


# Normalizing the data to make it scale free


# In[25]:


# normalization function


def normailzation(dataset):
    for column in dataset.columns:
        exist_count = numericCols.count(column)
        if exist_count == 1:
            dataset[column] = (dataset[column] - dataset[column].min()) / (
                dataset[column].max() - dataset[column].min())
        else:
            pass


# In[26]:


data2 = data1


# In[27]:


data2


# In[28]:


data1


# In[29]:


df_norm = data1
normailzation(df_norm)


# In[30]:


df_norm


# In[31]:


# Checking whether numerical data following normal distribution or not


# In[32]:


numericCols1 = [
    'AGE', 'S_AD_ORIT', 'D_AD_ORIT', 'NA_BLOOD', 'ROE', 'K_BLOOD', 'ALT_BLOOD',
    'AST_BLOOD', 'L_BLOOD'
]


# In[33]:


df_numeric = df_norm[numericCols1]
df_numeric.head(5)


# In[34]:


df_numeric.hist(grid=False, figsize=(100, 60), bins=40)


# In[35]:


# Columns roe,ALT_BLOOD,AST_BLOOD L_BLOOD follows left skewed distribution , to normalize it we have applied square route Transformation as log transformation resulting in infinate values


# In[36]:


df_transformed = df_norm.copy()


# In[37]:


df_transformed['ROE'] = np.sqrt(df_transformed['ROE'])
df_transformed['ALT_BLOOD'] = np.sqrt(df_transformed['ALT_BLOOD'])
df_transformed['AST_BLOOD'] = np.sqrt(df_transformed['AST_BLOOD'])
df_transformed['L_BLOOD'] = np.sqrt(df_transformed['L_BLOOD'])
df_transformed.head()


# In[38]:


df_numeric1 = df_transformed[numericCols1]
df_numeric1.hist(grid=False, figsize=(100, 60))


# In[39]:


# All the numerical veriables are almost normally distributed


# In[40]:


# checking the strength of relationship b/w input variables and output variable using predicitve power score


# In[41]:


pip install - U ppscore


# In[42]:


import ppscore as pps


# In[43]:


pps_pred = pps.predictors(df_transformed, "LET_IS")
pd.set_option('display.max_rows', None)
pps_pred.sort_values('ppscore', ascending=False).reset_index(drop=True)


# In[44]:


pps_mat = pps.matrix(df_transformed)
# pps_mat[x=='LET_IS']


# In[45]:


import seaborn as sns
plt.figure(figsize=(16, 10))
matrix_data = pps.matrix(df_transformed)[['x', 'y', 'ppscore']].pivot(
    columns='x', index='y', values='ppscore')
sns.heatmap(matrix_data, vmin=0, vmax=1,
            cmap="Blues", linewidths=0.5, annot=True)


# In[46]:


# correlation
data_corr = df_transformed.corr()


# In[47]:


import seaborn
plt.figure(figsize=(15, 10))
seaborn.heatmap(data_corr, cmap="YlGnBu")  # Displaying the Heatmap
seaborn.set(font_scale=2, style='white')

plt.title('Heatmap correlation')
plt.show()


# # Univariate Analysis for Categorical Data

# In[48]:


sns.stripplot(x='SEX', y='AGE', data=data)
plt.ylabel('AGE')
plt.show()


# In[49]:


sns.swarmplot(x='S_AD_ORIT', y='AGE', data=data)
plt.ylabel('AGE')
plt.show()


# In[50]:


sns.boxplot(x='INF_ANAM', y='AGE', data=data)
plt.ylabel('AGE')


# In[51]:


plt.subplot(1, 2, 1)
sns.violinplot(x='STENOK_AN', y='AGE', data=data)
plt.subplot(1, 2, 2)
sns.boxplot(x='FK_STENOK', y='AGE', data=data)
plt.show()


# In[52]:


f, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(x='SEX', y='AGE', data=data)


# In[53]:


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('Univariate analysis using categorical columns')

sns.stripplot(ax=axes[0, 0], data=data, x='GB', y='AGE')
sns.boxplot(ax=axes[0, 1], data=data, x='GB', y='S_AD_KBRIG')
sns.violinplot(ax=axes[0, 2], data=data, x='GB', y='D_AD_KBRIG')
sns.stripplot(ax=axes[1, 0], data=data, x='GB', y='S_AD_KBRIG')
sns.boxplot(ax=axes[1, 1], data=data, x='GB', y='D_AD_KBRIG')
sns.violinplot(ax=axes[1, 2], data=data, x='GB', y='D_AD_KBRIG')


# In[54]:


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('Univariate analysis using categorical columns')

sns.stripplot(ax=axes[0, 0], data=data, x='DLIT_AG', y='AGE')
sns.boxplot(ax=axes[0, 1], data=data, x='DLIT_AG', y='AGE')
sns.violinplot(ax=axes[0, 2], data=data, x='DLIT_AG', y='AGE')
sns.boxplot(ax=axes[1, 0], data=data, x='ritm_ecg_p_01', y='S_AD_KBRIG')
sns.barplot(ax=axes[1, 1], data=data, x='ritm_ecg_p_01', y='S_AD_KBRIG')
sns.violinplot(ax=axes[1, 2], data=data, x='ritm_ecg_p_01', y='S_AD_KBRIG')


# In[55]:


sns.lineplot(data=data, x='ritm_ecg_p_01', y='S_AD_KBRIG')


# In[56]:


sns.set_theme(style="whitegrid")
sns.boxenplot(x="AGE", y="LET_IS",
              color="b", order=data,
              scale="linear", data=data)


# In[57]:


sns.catplot(x="LET_IS", y="AGE", hue="SEX", kind="bar", data=data)


# In[58]:


numericCols = ['AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT',
               'NA_BLOOD', 'ROE', 'K_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD']


# In[59]:


data1 = pd.DataFrame()
data1


# In[60]:


for column_name in data.columns:
    exist_count = numericCols.count(column_name)
    if exist_count > 0:
        data1[column_name] = data[column_name]
    else:
        pass


# In[61]:


data1.head()


# In[62]:


#sns.PairGrid(data1, hue="AGE",palette="GnBu_d")
#map(plt.scatter, s=50, edgecolor="white")
# add_legend()


# In[63]:


sns.pairplot(data1)


# In[64]:


sns.heatmap(data1, vmin=0, vmax=1)


# In[65]:


sns.lmplot(x='S_AD_ORIT', y='D_AD_KBRIG', data=data)


# In[66]:


sns.distplot(data.AGE)


# In[67]:


sns.jointplot(x='S_AD_ORIT', y='AGE', data=data[data['S_AD_ORIT'] < 100], kind='hex',
              gridsize=20)


# In[ ]:





# In[68]:


sns.pairplot(data1.head(), hue='AGE')


# In[69]:


sns.pairplot(data1, hue='AGE', diag_kind='kde',
             plot_kws={'edgecolor': 'k'}, size=4)


# In[ ]:





# In[70]:


sns.heatmap(data1.corr(), annot=True, cmap='Blues')
plt.figure(figsize=(15, 9))
plt.show()


# In[71]:


sns.heatmap(data.corr() > 0.6, annot=True)


# In[72]:


sns.scatterplot(data=data, x='AGE', y='L_BLOOD')
plt.show()


# In[73]:


sns.scatterplot(data=data, x='S_AD_KBRIG', y='S_AD_ORIT')
plt.show()
sns.scatterplot(data=data, x='D_AD_KBRIG', y='D_AD_ORIT')
plt.show()


# In[74]:


sns.lmplot(data=data, x='S_AD_KBRIG', y='S_AD_ORIT')
plt.show()
sns.lmplot(data=data, x='D_AD_KBRIG', y='D_AD_ORIT')
plt.show()


# In[75]:


pd.crosstab(data.IBS_POST, data.SEX).plot(kind="bar")


# In[76]:


pd.crosstab(data.DLIT_AG, data.SEX).plot(kind="bar")


# In[77]:


sns.swarmplot(x='DLIT_AG', y='AGE', data=data)
plt.show()


# In[78]:


sns.violinplot(x='DLIT_AG', y='AGE', data=data)
plt.show()


# In[79]:


# Finding final varaibles value counts
df_norm['LET_IS'].value_counts()


# In[80]:


# Plotting the piechart
labels = ['unknown (alive)', 'cardiogenic shock', 'pulmonary edema', 'myocardial rupture',
          'progress of congestive heart failure', 'thromboembolism', 'asystole', 'ventricular fibrillation']
df_norm['LET_IS'].value_counts().plot(kind='pie', autopct='%1.0f%%',
                                      figsize=(10, 10), labels=labels)
plt.show()


# #Auto EDA

# In[81]:


get_ipython().system(' pip install https: // github.com/pandas-profiling/pandas-profiling/archive/master.zip')


# In[82]:


# 1. Using Pandas Profiling
import pandas_profiling as pp
EDA_report = pp.ProfileReport(data)
EDA_report.to_notebook_iframe()
EDA_report.to_file(output_file='Emails_PP_Report.html')


# In[83]:


get_ipython().system('pip install sweetviz')


# In[84]:


# 2. Using Sweetviz
import sweetviz as sv
sweet_report = sv.analyze(data)
sweet_report.show_notebook()
sweet_report.show_html('Emails_SV_Report.html')


# In[85]:


# Impute those NaNs with 0 values
df_norm.fillna(0, inplace=True)
df_norm


# In[86]:


X = data2.iloc[:, :-1]
X


# In[87]:


Y = data2.iloc[:, -1:]
Y


# #Data Balancing

# In[88]:


# pip install -U imbalanced-learn


# In[89]:


#import library
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(X, Y)


print('Original dataset shape', Counter(Y))
print('Resample dataset shape', Counter(y_ros))


# In[90]:


# import library
from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X, Y)

print('Original dataset shape', Counter(Y))
print('Resample dataset shape', Counter(y_smote))


# In[91]:


plt.hist(y_smote, bins=20)
plt.show()


# In[92]:


# Splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x_smote, y_smote, test_size=0.33, random_state=42)


# In[93]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[94]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Defining model
model_dt = DecisionTreeClassifier()

# fit the model with training data
model_dt.fit(x_train, y_train)

# depth of the decision tree
print("Depth of the decision tree:", model_dt.get_depth())

# predict the target on train data
predict_train = model_dt.predict(x_train)
predict_test = model_dt.predict(x_test)
predict_overall = model_dt.predict(X)

# Accuracy
Acc_train_dt = accuracy_score(y_train, predict_train)
Acc_test_dt = accuracy_score(y_test, predict_test)
Acc_overall_dt = accuracy_score(Y, predict_overall)
print("Train accuracy of the Decision Tree:", Acc_train_dt)
print("Test accuracy of the Decision Tree:", Acc_test_dt)
print("Overall accuracy of the Decision Tree:", Acc_overall_dt)


# In[95]:


# Classification Report for Decision Tree

from sklearn.metrics import classification_report

dec_tree_report = classification_report(y_test, predict_test, output_dict=True)

dec_tree_report = pd.DataFrame(dec_tree_report).transpose()
print('Classification Report for Decision Tree')
dec_tree_report


# In[96]:


import pickle
pickle.dump(model_dt, open("DT.pkl", "wb"))


# In[97]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Defining model
model_rf = RandomForestClassifier()

# fit the model with training data
model_rf.fit(x_train, y_train)

# predict the target on train data
predict_train = model_rf.predict(x_train)
predict_test = model_rf.predict(x_test)
predict_overall = model_rf.predict(X)

# Accuracy
Acc_train_rf = accuracy_score(y_train, predict_train)
Acc_test_rf = accuracy_score(y_test, predict_test)
Acc_overall_rf = accuracy_score(Y, predict_overall)
print("Train accuracy of the Random Forest:", Acc_train_rf)
print("Test accuracy of the Random Forest:", Acc_test_rf)
print("Overall accuracy of the Random Forest:", Acc_overall_rf)

Acc_train_rf, Acc_test_rf, print(
    'Number of Trees used: ', model_rf.n_estimators)


# In[98]:


# Classification Report for Random Forest

rf_report = classification_report(y_test, predict_test, output_dict=True)

rf_report = pd.DataFrame(rf_report).transpose()
print('Classification Report for Random Forest')
rf_report


# In[99]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

# Defining model
model_gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5)

# fit the model with training data
model_gbm.fit(x_train, y_train)

# predict the target on train data
predict_train = model_gbm.predict(x_train)
predict_test = model_gbm.predict(x_test)
predict_overall = model_gbm.predict(X)

# Accuracy
Acc_train_gbm = accuracy_score(y_train, predict_train)
Acc_test_gbm = accuracy_score(y_test, predict_test)
Acc_overall_gbm = accuracy_score(Y, predict_overall)
print("Train accuracy of the Gradient Boosting:", Acc_train_gbm)
print("Test accuracy of the Gradient Boosting:", Acc_test_gbm)
print("Overall accuracy of the Gradient Boosting:", Acc_overall_gbm)


# In[100]:


# Classification Report for Gradient Boosting

gb_report = classification_report(y_test, predict_test, output_dict=True)

gb_report = pd.DataFrame(gb_report).transpose()
print('Classification Report for Gradient Boosting')
gb_report


# In[101]:


# pip install xgboost


# In[102]:


# XG Boosting
from xgboost import XGBClassifier

# Defining model
model_xgb = XGBClassifier(n_estimators=100, max_depth=5)

# fit the model with training data
model_xgb.fit(x_train, y_train)


# In[103]:


# predict the target on train data
predict_train = model_xgb.predict(x_train)
predict_test = model_xgb.predict(x_test)
#predict_overall = model_xgb.predict(X)

# Accuracy
Acc_train_xgb = accuracy_score(y_train, predict_train)
Acc_test_xgb = accuracy_score(y_test, predict_test)
Acc_overall_xgb = accuracy_score(Y, predict_overall)
print("Train accuracy of the XG Boosting:", Acc_train_xgb)
print("Test accuracy of the XG Boosting:", Acc_test_xgb)
#print("Overall accuracy of the XG Boosting:", Acc_overall_xgb)


# In[104]:


# Classification Report for XB Boosting

gb_report = classification_report(y_test, predict_test, output_dict=True)

gb_report = pd.DataFrame(gb_report).transpose()
print('Classification Report for XB Boosting')
gb_report


# In[105]:


# SVM
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(x_train)
X_test = sc_X.fit_transform(x_test)


# In[106]:


from sklearn.svm import SVC
model = SVC(gamma=0.01, kernel='linear')
model.fit(x_train, y_train)


# In[107]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[108]:


# predict the target on train data
predict_train = model.predict(x_train)
predict_test = model.predict(x_test)
predict_overall = model.predict(X)

# Accuracy
Acc_train_svm = accuracy_score(y_train, predict_train)
Acc_test_svm = accuracy_score(y_test, predict_test)
Acc_overall_svm = accuracy_score(Y, predict_overall)
print("Train accuracy of the SVM:", Acc_train_svm)
print("Test accuracy of the SVM:", Acc_test_svm)
print("Overall accuracy of the SVM:", Acc_overall_svm)


# In[109]:


# Classification Report for SVM

svm_report = classification_report(y_test, predict_test, output_dict=True)

svm_report = pd.DataFrame(svm_report).transpose()
print("Classification Report for SVM ")
svm_report


# In[110]:


# Logistic Regression
# Training the Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# fit the model with training data
classifier.fit(x_train, y_train)


# In[111]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# predict the target on train data
predict_train = classifier.predict(x_train)
predict_test = classifier.predict(x_test)
predict_overall = classifier.predict(X)

# Accuracy
Acc_train_lg = accuracy_score(y_train, predict_train)
Acc_test_lg = accuracy_score(y_test, predict_test)
Acc_overall_lg = accuracy_score(Y, predict_overall)
print("Train accuracy of the Logistic Regression:", Acc_train_lg)
print("Test accuracy of the Logistic Regression:", Acc_test_lg)
print("Overall accuracy of the Logistic Regression:", Acc_overall_lg)


# In[112]:


# Classification Report for SVM Boosting

lg_report = classification_report(y_test, predict_test, output_dict=True)

lg_report = pd.DataFrame(lg_report).transpose()
print("Classification Report for SVM Boosting")
lg_report


# In[113]:


# ANN
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# define a function to build the keras model


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(1190, input_shape=(1190, 510), kernel_initializer='normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1190, kernel_initializer='normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax'))

    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


model = create_model()

print(model.summary())


# In[114]:


# Classification Report for ANN

nn_report = classification_report(y_test, predict_test, output_dict=True)

nn_report = pd.DataFrame(nn_report).transpose()
print("Classification Report for ANN")
nn_report


# In[115]:


from sklearn import model_selection


# In[116]:


models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('SVM', SVC()))
models.append(('LG', LogisticRegression()))


# In[123]:


# evaluate each model in turn
results = []
names = []
seed = 42
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = None, shuffle = True)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[125]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




