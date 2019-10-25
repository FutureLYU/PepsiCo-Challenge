#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# In[2]:


data = pd.read_excel('shelf-life-study-data-for-analytics-challenge_prediction.xlsx')


# In[8]:


data_process = data.drop(data[data['Storage Conditions'].isna() & data['Packaging Stabilizer Added'].isna() & data[
    'Transparent Window in Package'].isna()
    & data['Preservative Added'].isna() & data['Moisture (%)'].isna() & data['Residual Oxygen (%)'
                                                                            ].isna() & data[
        'Hexanal (ppm)'].isna()].index,0)


# In[10]:


data_process = data_process.drop(columns=['Study Number','Sample ID','Prediction'])


# In[12]:


data_process.loc[data_process['Base Ingredient'].isna(),'Base Ingredient'] = 'Unknown'
encoder = LabelEncoder()
encoded_BPP = data_process[['Base Ingredient','Product Type','Process Type']].apply(encoder.fit_transform)


# In[13]:


Y_Base = encoded_BPP[encoded_BPP['Base Ingredient'] != 6]['Base Ingredient'].values

X_Base = encoded_BPP[encoded_BPP['Base Ingredient'] != 6].drop(columns=['Base Ingredient']).values

model_rf_Base = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

model_rf_Base.fit(X_Base,Y_Base)

Xp_Base = encoded_BPP[encoded_BPP['Base Ingredient'] == 6].drop(columns=['Base Ingredient']).values
Yp_Base = model_rf_Base.predict(Xp_Base)

encoded_BPP.loc[encoded_BPP['Base Ingredient']==6,'Base Ingredient']=Yp_Base

data_process.loc[:,['Base Ingredient','Product Type','Process Type']] = encoded_BPP


# In[15]:


data_2c = data_process.drop(data_process[data_process['Moisture (%)'].isna() | data_process[
    'Residual Oxygen (%)'].isna()].index,0)


# In[16]:


data_2c.loc[data_2c['Packaging Stabilizer Added'].isna(),'Packaging Stabilizer Added'] = 'Unknown'
encoded_PSA = data_2c[['Packaging Stabilizer Added']].apply(encoder.fit_transform)


# In[17]:


Y_PSA = encoded_PSA[encoded_PSA['Packaging Stabilizer Added'] != 1]['Packaging Stabilizer Added'].values

X_PSA = data_2c[encoded_PSA['Packaging Stabilizer Added'] != 1][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values

model_rf_PSA = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

model_rf_PSA.fit(X_PSA,Y_PSA)

Xp_PSA = data_2c[encoded_PSA['Packaging Stabilizer Added'] == 1][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values
Yp_PSA = model_rf_PSA.predict(Xp_PSA)

encoded_PSA.loc[encoded_PSA['Packaging Stabilizer Added']==1,'Packaging Stabilizer Added']=Yp_PSA

data_2c.loc[:,['Packaging Stabilizer Added']] = encoded_PSA


# In[19]:


data_2c.loc[data_2c['Preservative Added'].isna(),'Preservative Added'] = 'Unknown'
encoded_PA = data_2c[['Preservative Added']].apply(encoder.fit_transform)


# In[20]:


Y_PA = encoded_PA[encoded_PA['Preservative Added'] != 1]['Preservative Added'].values

X_PA = data_2c[encoded_PA['Preservative Added'] != 1][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values

model_rf_PA = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

model_rf_PA.fit(X_PA,Y_PA)

Xp_PA = data_2c[encoded_PA['Preservative Added'] == 1][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values
Yp_PA = model_rf_PA.predict(Xp_PA)

encoded_PA.loc[encoded_PA['Preservative Added']==1,'Preservative Added']=Yp_PA

data_2c.loc[:,['Preservative Added']] = encoded_PA


# In[23]:


data_2c.loc[data_2c['Storage Conditions'].isna(),'Storage Conditions'] = 'Unknown'
encoded_SC = data_2c[['Storage Conditions']].apply(encoder.fit_transform)


# In[24]:


Y_SC = encoded_SC[encoded_SC['Storage Conditions'] != 2]['Storage Conditions'].values

X_SC = data_2c[encoded_SC['Storage Conditions'] != 2][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values

model_rf_SC = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

model_rf_SC.fit(X_SC,Y_SC)

Xp_SC = data_2c[encoded_SC['Storage Conditions'] == 2][
    ['Product Type','Base Ingredient','Process Type','Sample Age (Weeks)',
     'Difference From Fresh','Moisture (%)','Residual Oxygen (%)']].values
Yp_SC = model_rf_SC.predict(Xp_SC)

encoded_SC.loc[encoded_SC['Storage Conditions']==2,'Storage Conditions']=Yp_SC

data_2c.loc[:,['Storage Conditions']] = encoded_SC


# In[25]:


X_final = np.array(data_2c.drop(columns=['Sample Age (Weeks)','Transparent Window in Package',
                                'Moisture (%)','Residual Oxygen (%)','Hexanal (ppm)']))
Y_final = np.array(data_2c['Sample Age (Weeks)'])


# In[26]:


data_test = data.copy()
data_test = data_test.drop(columns=['Study Number','Sample ID','Prediction'])
data_test.loc[data_test['Base Ingredient'].isna(),'Base Ingredient'] = 'Unknown'
data_test[['Base Ingredient','Product Type',
           'Process Type']] = data_test[['Base Ingredient','Product Type',
                                         'Process Type']].apply(encoder.fit_transform)
data_test.loc[data_test['Packaging Stabilizer Added'].isna(),'Packaging Stabilizer Added'] = 'Unknown'
data_test[['Packaging Stabilizer Added']] = data_test[['Packaging Stabilizer Added']].apply(encoder.fit_transform)
data_test.loc[data_test['Preservative Added'].isna(),'Preservative Added'] = 'Unknown'
data_test[['Preservative Added']] = data_test[['Preservative Added']].apply(encoder.fit_transform)
data_test.loc[data_test['Storage Conditions'].isna(),'Storage Conditions'] = 'Unknown'
data_test[['Storage Conditions']] = data_test[['Storage Conditions']].apply(encoder.fit_transform)


# In[30]:


X_test = np.array(data_test.drop(columns=['Transparent Window in Package',
                                 'Moisture (%)','Residual Oxygen (%)','Hexanal (ppm)','Sample Age (Weeks)']))


# In[33]:


xgbr=xgb.XGBRegressor(learning_rate=0.08,n_estimators=700)
xgbr.fit(X_final,Y_final)
Yp_final=xgbr.predict(X_test)#预测


# In[38]:


Prediction = np.round(Yp_final).astype(int)


# In[43]:


data.loc[:,'Prediction'] = Prediction


# In[53]:


data.to_excel('HanxiLyu.xlsx',sheet_name='Shelf Life Study Data for Analy')


# In[ ]:




