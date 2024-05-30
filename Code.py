# !pip install pytest-warnings==0.3.1
# !pip install pandas 2.1.2
# !pip install numpy 1.26.1
# !pip install matplotlib 3.8.1
# !pip install seaborn 0.13.0
# !pip freeze
# !pip install scikit-learn 3.2.0
# import warning
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
housing = pd.read_csv("train.csv")
housing.head()
housing.shape
housing.describe()
housing.info()

# This shows how many out of 1460 rows are not null
# To make all of them not null we replace NULL by None type in the Data cleaning first cell
housing.isnull().sum()/housing.shape[0]*100
cols = ['Alley', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'FireplaceQu', 'MiscFeature', 'Fence', 'PoolQC', 'GarageCond', 'GarageQual', 'GarageFinish',
        'GarageType']
for i in cols:
    housing[i].fillna("None", inplace=True)

# replacing null by None 
housing.info()
# Import visualization libs
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Checking if target variable SalePrice is normally distributed
plt.figure(figsize=[6,6])
sns.distplot(housing['SalePrice'])
print("Skewness:", housing["SalePrice"].skew())
print("Kurtosis:", housing["SalePrice"].kurtosis())
# Log transformation
housing['SalePrice'] = np.log(housing['SalePrice'])
# Checking if the target variable after transformation
plt.figure(figsize=[6,6])
sns.distplot(housing['SalePrice'])
plt.show()
print("Skewness:", housing["SalePrice"].skew())
print("Kurtosis:", housing["SalePrice"].kurtosis())
housing.drop('Id', axis=1, inplace=True)
housing[['MSSubClass', 'OverallQual', 'OverallCond']] = housing[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')
housing['LotFrontage'] = pd.to_numeric(housing['LotFrontage'], errors='coerce')
housing['MasVnrArea'] = pd.to_numeric(housing['MasVnrArea'], errors='coerce')
null_cols = housing.columns[housing.isnull().any()]
null_cols
for i in null_cols:
    if housing[i].dtype == np.float64 or housing[i].dtype == np.int64:
        housing[i].fillna(housing[i].mean(), inplace=True)
    else:
        housing[i].fillna(housing[i].mode()[0], inplace=True)
housing.isna().sum()




# List of Categorical Columns

cat_cols = housing.select_dtypes(include='object').columns
cat_cols
# List of numerical cols
num_cols = housing.select_dtypes(include=['int64', 'float64']).columns
num_cols
# Numerical columns
# Plotting boxplot to visualize distribution and check for any outliers

for i in num_cols:
    plt.figure(figsize=[8,5])
    print(i)
    sns.boxplot(housing[i])
    plt.show()

# Categorical Columns
# Plotting pie plots to visualize the values distribution
for i in cat_cols:
    print(housing[cols].value_counts(normalize=True))
    plt.figure(figsize=[5,5])
    housing[i].value_counts(normalize=True).plot.pie(labeldistance=None, autopct='%1.2f%%')
    plt.legend()
    plt.show()
    print("----------")
# Plot of MSZoning vs LotFrontage
sns.barplot(x='MSZoning', y='LotFrontage', data=housing)
plt.show()
# Plot of MSSubClass vs LotFrontage
sns.barplot(x='MSSubClass', y='LotFrontage', data=housing)
plt.show()
# plot of HouseStyle vs SalePrice based on street
sns.barplot(x='HouseStyle', y='SalePrice', hue='Street', data=housing)
# Plot of BldgType vs SalePrice
sns.barplot(x='BldgType', y='SalePrice', data=housing)
plt.show()
# sns.set()

# Plot of BsmtQual vs SalePrice
sns.barplot(x='BsmtQual', y='SalePrice', data=housing)
plt.show()

# Calculating Age of Property
housing['Age'] = housing['YrSold'] - housing['YearBuilt']
housing['Age'].head()

# Dropping YearSold and YearBuilt
housing.drop(columns=['YearBuilt', 'YrSold'], axis=1, inplace=True)
housing.head()
plt.figure(figsize=[25,25])
sns.heatmap(housing.corr(numeric_only=True), annot=True, cmap='BuPu')
plt.title("Correlation of Numerical Values")
k=10
plt.figure(figsize = [15,15])
cols = housing.corr(numeric_only = True).nlargest(k, "SalePrice").index
cm = np.corrcoef(housing[cols].values.T)
sns.heatmap(cm, annot=True, square=True, fmt = '.2f', cbar=True, annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'GarageCars','TotalBsmtSF','1stFlrSF']
plt.figure(figsize=[20,20])
sns.pairplot(housing[cols])
plt.show()



housing_num = housing.select_dtypes(include=['int64','float64'])
housing_cat = housing.select_dtypes(include='object')
housing_cat
housing_cat_dm = pd.get_dummies(housing_cat, drop_first = True, dtype = int)
housing_cat_dm
house = pd.concat([housing_num, housing_cat_dm], axis=1)
house.head()
house.shape
# SPlit into target and feature variables
X = house.drop(['SalePrice'], axis=1).copy()
y = house['SalePrice'].copy()
X.head()
y.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape
y_train.shape
num_cols = list(X_train.select_dtypes(include=['int64', 'float64']).columns)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])
BUilding a function to calculate evaluation metrics
def eval_metrics(y_train, y_train_pred, y_test, y_pred):
    # r2 values for train and test data
    print("r2 score (train) = ", '%.2f' % r2_score(y_train, y_train_pred))
    print("r2 score (test) = ", '%.2f' % r2_score(y_test, y_pred))




# IMprt ML libs
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
# Applying Ridge regressing with varing the hyperparameter 'Lambda'
params = {'alpha': [0.0001, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 20, 50, 100, 500, 1000]}
ridge = Ridge()
ridgeCV = GridSearchCV(estimator=ridge, param_grid=params, scoring='neg_mean_absolute_error', cv=5, return_train_score=True, verbose=1, n_jobs=-1)
ridgeCV.fit(X_train, y_train)
ridgeCV.best_params_
ridgeCV.cv_results_
ridge = Ridge(alpha=9)
ridge.fit(X_train, y_train)
ridge.coef_
y_train_pred = ridge.predict(X_train)
y_pred = ridge.predict(X_test)
eval_metrics(y_train, y_train_pred, y_test, y_pred)
ridgeCV_res = pd.DataFrame(ridgeCV.cv_results_)
ridgeCV_res.head()
plt.plot(ridgeCV_res['param_alpha'], ridgeCV_res['mean_train_score'], label='train')
plt.plot(ridgeCV_res['param_alpha'], ridgeCV_res['mean_test_score'], label='test')
plt.xlabel('alpha')
plt.ylabel('R2_score')
plt.xscale('log')
plt.legend()
plt.show()

# Applying lasso regression with varing the hyperparameter 'Lambda'
lasso = Lasso()
lassoCV = GridSearchCV(estimator=lasso, param_grid=params, scoring='neg_mean_absolute_error', cv=5, return_train_score=True, verbose=1, n_jobs=-1)
lassoCV.fit(X_train, y_train)
lassoCV.best_params_
lasso = Lasso(alpha=0.0001)
lasso.fit(X_train, y_train)
lasso.coef_
y_train_pred1 = lasso.predict(X_train)
y_pred1 = lasso.predict(X_test)
eval_metrics(y_train, y_train_pred1, y_test, y_pred1)
lassoCV_res = pd.DataFrame(lassoCV.cv_results_)
lassoCV_res.head()
plt.plot(lassoCV_res['param_alpha'], lassoCV_res['mean_train_score'], label='train')
plt.plot(lassoCV_res['param_alpha'], lassoCV_res['mean_test_score'], label='test')
plt.xlabel('alpha')
plt.ylabel('R2_score')
plt.xscale('log')
plt.legend()
plt.show()





betas = pd.DataFrame(index=X.columns) #convert the columns to a dataframe as betas
betas.rows = X.columns
# Creating columns for Ridge and lasso coefficients against each feature
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_
betas
# View the features removed by lasso
lasso_cols_removed = list(betas[betas['Lasso']==0].index)
print(lasso_cols_removed)
# View the features removed by lasso
lasso_cols_selected = list(betas[betas['Lasso']!=0].index)
print(lasso_cols_selected)
print(len(lasso_cols_removed)) # 179 features removed by lasso
print(len(lasso_cols_selected)) # 107 features are selected by lasso

# View the top 10 coefficients of Ridge regression in descending order
betas['Ridge'].sort_values(ascending=False)[:10]

# We have to take inverse log of betas to interpret the ridge coefficient in terms of target variables
ridge_coeffs = np.exp(betas['Ridge'])
ridge_coeffs.sort_values(ascending=False)[:10]

# View the top 10 coefficients of Lasso in descending order
betas['Lasso'].sort_values(ascending=False)[:10]
# We have to take inverse log of betas to interpret the ridge coefficient in terms of target variables
lasso_coeffs = np.exp(betas['Lasso'])
lasso_coeffs.sort_values(ascending=False)[:10]


