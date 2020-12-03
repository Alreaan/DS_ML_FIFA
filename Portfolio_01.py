
#%%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import plotly.graph_objs as go 
import plotly.offline as py
#%%
import xgboost as xgb
from xgboost import XGBRegressor

# %%
# install required modules 
pip install -U scikit-learn
pip install xgboost
pip install plotly
pip install nbformat
# %%
fifa = pd.read_csv('data/FIFA19.csv')
# %%
fifa
# %%
#Exploratory Data Analysis (EDA)
#EDA to help understanding the dataset and get an insight from it.
potential = pd.DataFrame(fifa[['Nationality','Potential']].sort_values(by = 'Potential', ascending=False)).head(200)
trace = [go.Choropleth(
    locationmode = 'country names',
    locations = potential['Nationality'],
    text = potential['Nationality'],
    z = potential['Potential'],
)]
layout = go.Layout(title = 'Top 200 players By country')
fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)

# Heatmap
#to find out correlation among variables in the dataset 
#so, when to veriables have a pos correlation that means they both move in same direction like value and wage
corr_mt = fifa.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr_mt, square=True)
plt.show()

#%%
# in this plot we clearly can notice that thevalue is highly skewed.
plt.figure(figsize=(8,8))
sns.distplot(fifa['Value'], color='black')
plt.title ('target variable (Value)',size=14)
plt.show
#%%
#with Log
plt.figure(figsize =(8,8))
sns.distplot(np.log(fifa['Value']), color='yellow')
plt.show()
# %%
# pre-processing

col_name_fifa = list(fifa.columns)
col_names = col_name_fifa[54:88]
col_names = ['Value','Wage','Name','Age','Position','International Reputation','Weak Foot','Skill Moves','Work Rate','Body Type','Height','Weight','Overall','Potential']+col_names
len(col_names)
fifa= fifa[col_names]
# %%
#converting variables to numerical

fifa['Value'] = fifa['Value'].apply(lambda x: x.replace ('€',''))
fifa['Value'] = (fifa['Value'].replace(r'[KM]+$', '', regex=True).astype(float)*
fifa['Value'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))
fifa['Value']

fifa['Wage'] = fifa['Wage'].apply(lambda x: x.replace ('€',''))
fifa['Wage'] = (fifa['Wage'].replace(r'[KM]+$', '', regex=True).astype(float)*
fifa['Wage'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))
fifa['Wage']
# %%
# change variables data types

fifa ['Weight'] = fifa['Weight'].astype(str)
fifa ['Weight'] = fifa['Weight'].apply(lambda x: x.replace('lbs',''))
fifa ['Weight'] = fifa['Weight'].apply(lambda x: float(x))

def convert_weight (wt):
    foot = float(wt.split("'")[0])
    inch = float(wt.split("'")[1])
    return (foot*12 + inch)

fifa['Height'] = fifa['Height'].astype(str)
fifa['Height'] = fifa['Height'].apply(convert_weight)

# %%
fifa ['Position'].value_counts()
# %%
Front = ['LF','RF','CF','ST','LW','RW','LS','RS']
Mid = ['RM','LM','RAM','RAM','CDM','RDM','LAM','LDM','CM','CAM','LCM','RCM']
Defense = ['CB','RB','LB','LCB','RCB','LWB','RWB']

fifa['Position'].replace(Defense, 'Defender', inplace=True)
fifa['Position'].replace(Mid, 'Midfield', inplace=True)
fifa['Position'].replace(Front, 'Forward', inplace=True)
# %%
one_hot_position = pd.get_dummies(fifa ['Position'].replace('GK', np.nan), prefix='Position')
fifa = pd.concat([fifa, one_hot_position], axis=1)
fifa.drop('Position',axis=1, inplace=True)

#%%
fifa ['Body Type'] = fifa ['Body Type'].replace(['Messi','C. Ronaldo', 'Neymar','Courtois','PLAYER_BODY_TYPE_25','Shaqiri','Akinfenwa'],'Normal')
fifa['Body Type']

#%%
one_hot_body_type = pd.get_dummies(fifa['Body Type'].replace('Normal',np.nan), prefix='Body_Type')
fifa = pd.concat([fifa, one_hot_body_type], axis=1)
fifa.drop('Body Type', axis =1, inplace=True)


fifa_data = fifa.copy()
fifa_data.drop(['Wage','Name'], axis=1, inplace=True)
fifa_data['Value'] = np.log(fifa_data['Value'])


X = fifa_data.drop('Value', axis=1)
y = fifa_data['Value']

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size = 0.2, random_state = 123)
print('train x:', train_x.shape)
print('train y:', train_y.shape)
print('train x:', valid_x.shape)
print('train y:', valid_y.shape)


xgb_model = XGBRegressor(n_estimators= 100, random_state= 50)
xgb_model.fit(train_x, train_y)

train_y_predict = xgb_model.predict(train_x)
MAE_train = mean_absolute_error(train_y, train_y_predict)
print('MAE_train: ', MAE_train)

valid_y_predict = xgb_model.predict(valid_x)
MAE_valid = mean_absolute_error(valid_y, valid_y_predict)
print('MAE_valid: ', MAE_valid)

# %%
