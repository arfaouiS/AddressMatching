#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from gensim.models.fasttext import load_facebook_model
import math
import re


# In[4]:


def cosine_similarity11(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


# In[2]:


def loadAddressPair(file):
    
    df = pd.read_csv(file)
    
    Add1_split = []
    Add2_split = []
    Add_split = []
    a = 0
    b = 0
    c = 0
    key = ['LA', 'LE', 'LES', 'DU', 'DES', 'DE']
    for i in range(0, len(df)):
        
        L = df['INBUILDING'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'INBUILDING'] = ' '.join(str(x) for x in L)
        df.loc[i,'INBUILDING'] = re.sub("\s\s+", " ", df['INBUILDING'][i])
        df.loc[i,'INBUILDING'] = df['INBUILDING'][i].strip()

        L = df['INBUILDINGK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'INBUILDINGK'] = ' '.join(str(x) for x in L)
        df.loc[i,'INBUILDINGK'] = re.sub("\s\s+", " ", df['INBUILDINGK'][i])
        df.loc[i,'INBUILDINGK'] = df['INBUILDINGK'][i].strip()

        L = df['EXTBUILDING'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'EXTBUILDING'] = ' '.join(str(x) for x in L)
        df.loc[i,'EXTBUILDING'] = re.sub("\s\s+", " ", df['EXTBUILDING'][i])
        df.loc[i,'EXTBUILDING'] = df['EXTBUILDING'][i].strip()

        L = df['EXTBUILDINGK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'EXTBUILDINGK'] = ' '.join(str(x) for x in L)
        df.loc[i,'EXTBUILDINGK'] = re.sub("\s\s+", " ", df['EXTBUILDINGK'][i])
        df.loc[i,'EXTBUILDINGK'] = df['EXTBUILDINGK'][i].strip()

        L = df['POILOGISTIC'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'POILOGISTIC'] = ' '.join(str(x) for x in L)
        df.loc[i,'POILOGISTIC'] = re.sub("\s\s+", " ", df['POILOGISTIC'][i])
        df.loc[i,'POILOGISTIC'] = df['POILOGISTIC'][i].strip()

        L = df['POILOGISTICK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'POILOGISTICK'] = ' '.join(str(x) for x in L)
        df.loc[i,'POILOGISTICK'] = re.sub("\s\s+", " ", df['POILOGISTICK'][i])
        df.loc[i,'POILOGISTICK'] = df['POILOGISTICK'][i].strip()

        L = df['ZONE'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'ZONE'] = ' '.join(str(x) for x in L)
        df.loc[i,'ZONE'] = re.sub("\s\s+", " ", df['ZONE'][i])
        df.loc[i,'ZONE'] = df['ZONE'][i].strip()

        L = df['ZONEK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'ZONEK'] = ' '.join(str(x) for x in L)
        df.loc[i,'ZONEK'] = re.sub("\s\s+", " ", df['ZONEK'][i])
        df.loc[i,'ZONEK'] = df['ZONEK'][i].strip()

        L = df['ROADNAME'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'ROADNAME'] = ' '.join(str(x) for x in L)
        df.loc[i,'ROADNAME'] = re.sub("\s\s+", " ", df['ROADNAME'][i])
        df.loc[i,'ROADNAME'] = df['ROADNAME'][i].strip()

        L = df['ROADNAMEK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'ROADNAMEK'] = ' '.join(str(x) for x in L)
        df.loc[i,'ROADNAMEK'] = re.sub("\s\s+", " ", df['ROADNAMEK'][i])
        df.loc[i,'ROADNAMEK'] = df['ROADNAMEK'][i].strip()

        L = df['CITY'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'CITY'] = ' '.join(str(x) for x in L)
        df.loc[i,'CITY'] = re.sub("\s\s+", " ", df['CITY'][i])
        df.loc[i,'CITY'] = df['CITY'][i].strip()

        L = df['CITYK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'CITYK'] = ' '.join(str(x) for x in L)
        df.loc[i,'CITYK'] = re.sub("\s\s+", " ", df['CITYK'][i])
        df.loc[i,'CITYK'] = df['CITYK'][i].strip()
        
        L = df['DISTRICT'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'DISTRICT'] = ' '.join(str(x) for x in L)
        df.loc[i,'DISTRICT'] = re.sub("\s\s+", " ", df['DISTRICT'][i])
        df.loc[i,'DISTRICT'] = df['DISTRICT'][i].strip()

        L = df['DISTRICTK'][i].split()
        for k in range(0, len(L)):
            if L[k] in key:
                L[k] = ' '

        df.loc[i,'DISTRICTK'] = ' '.join(str(x) for x in L)
        df.loc[i,'DISTRICTK'] = re.sub("\s\s+", " ", df['DISTRICTK'][i])
        df.loc[i,'DISTRICTK'] = df['DISTRICTK'][i].strip()


    #########base d'apprentissage###############

    Add1_split = []
    Add2_split = []
    Add_split = []
    a = 0
    b = 0
    c = 0

    for i in range(0, len(df)):
        I1 = []
        I2 = []
        I1.insert(0, df['INBUILDING'][i])
        I1.insert(1, df['EXTBUILDING'][i])
        I1.insert(2, df['POILOGISTIC'][i])
        I1.insert(3, df['ZONE'][i])
        I1.insert(4, df['HOUSENUM'][i])
        I1.insert(5, df['ROADNAME'][i])
        I1.insert(6, df['CITY'][i])
        I1.insert(7, df['DISTRICT'][i])

        I2.insert(0, df['INBUILDINGK'][i])
        I2.insert(1, df['EXTBUILDINGK'][i])
        I2.insert(2, df['POILOGISTICK'][i])
        I2.insert(3, df['ZONEK'][i])
        I2.insert(4, df['HOUSENUMK'][i])
        I2.insert(5, df['ROADNAMEK'][i])
        I2.insert(6, df['CITYK'][i])
        I2.insert(7, df['DISTRICTK'][i])

        Add1_split.insert(a, I1)
        Add2_split.insert(b, I2)
        Add_split.insert(c, I1)
        Add_split.insert(c + 1, I2)
        a = a + 1
        b = b + 1
        c = c + 1

    model9 = load_facebook_model('fasttext.bin')

    cos0L = []
    cos1L = []
    cos2L = []
    cos3L = []
    cos4L = []
    cos5L = []
    cos6L = []
    cos7L = []
    cos8L = []
    cosALLlist = []
    LabelL = []
    for i in range(0, len(df)):
        #print(i)

        v1 = Add1_split[i][0]
        v2 = Add2_split[i][0]
        if v1 != 'NONE' and v2 != 'NONE':
            vectorW1 = model9.wv[v1]
            vectorW2 = model9.wv[v2]
            cos0 = cosine_similarity11(vectorW1, vectorW2)
        else:
            cos0 = 0
        cos0L.insert(i, cos0)

        v3 = Add1_split[i][1]
        v4 = Add2_split[i][1]
        if v3 != 'NONE' and v4 != 'NONE':
            vectorW3 = model9.wv[v3]
            vectorW4 = model9.wv[v4]
            cos1 = cosine_similarity11(vectorW3, vectorW4)
        else:
            cos1 = 0
        cos1L.insert(i, cos1)

        v5 = Add1_split[i][2]
        v6 = Add2_split[i][2]
        if v5 != 'NONE' and v6 != 'NONE':
            vectorW5 = model9.wv[v5]
            vectorW6 = model9.wv[v6]
            cos2 = cosine_similarity11(vectorW5, vectorW6)
        else:
            cos2 = 0
        cos2L.insert(i, cos2)

        v7 = Add1_split[i][3]
        v8 = Add2_split[i][3]
        if v7 != 'NONE' and v8 != 'NONE':
            vectorW7 = model9.wv[v7]
            vectorW8 = model9.wv[v8]
            cos3 = cosine_similarity11(vectorW7, vectorW8)
        else:
            cos3 = 0
        cos3L.insert(i, cos3)

        v9 = Add1_split[i][4]
        v10 = Add2_split[i][4]
        if v9 != 'NONE' and v10 != 'NONE':
            vectorW9 = model9.wv[v9]
            vectorW10 = model9.wv[v10]
            cos4 = cosine_similarity11(vectorW9, vectorW10)
        else:
            cos4 = 0
        cos4L.insert(i, cos4)

        v11 = Add1_split[i][5]
        v12 = Add2_split[i][5]
        if v11 != 'NONE' and v12 != 'NONE':
            vectorW11 = model9.wv[v11]
            vectorW12 = model9.wv[v12]
            cos5 = cosine_similarity11(vectorW11, vectorW12)
        else:
            cos5 = 0
        cos5L.insert(i, cos5)

        v13 = Add1_split[i][6]
        v14 = Add2_split[i][6]
        if v13 != 'NONE' and v14 != 'NONE':
            vectorW13 = model9.wv[v13]
            vectorW14 = model9.wv[v14]
            cos6 = cosine_similarity11(vectorW13, vectorW14)
        else:
            cos6 = 0
        cos6L.insert(i, cos6)
        
        v15 = Add1_split[i][7]
        v16 = Add2_split[i][7]
        if v15 != 'NONE' and v16 != 'NONE':
            vectorW15 = model9.wv[v15]
            vectorW16 = model9.wv[v16]
            cos7 = round(cosine_similarity11(vectorW15, vectorW16),2)
        else:
            cos7 = 0
        cos7L.insert(i, cos7)


    cosALL1 = []
    cosALL2 = []
    cosALL3 = []
    cosALL4 = []
    cosALL5 = []
    cosALL6 = []
    cosALL7 = []
    cosALL8 = []
    cosALL9 = []
    cosALL10 = []
    cosALL11 = []
    cosALL12 = []
    cosALL13 = []
    cosALL14 = []
    cosALL15 = []
    cosALL16 = []
    for i in range(0, len(df)):
        v40 = Add1_split[i][1]
        v41 = Add2_split[i][2]
        if v40 != 'NONE' and v41 != 'NONE':
            vectorW40 = model9.wv[v40]
            vectorW41 = model9.wv[v41]
            cos40 = cosine_similarity11(vectorW40, vectorW41)
        else:
            cos40 = 0
        cosALL5.insert(i, cos40)

        v42 = Add1_split[i][1]
        v43 = Add2_split[i][3]
        if v42 != 'NONE' and v43 != 'NONE':
            vectorW42 = model9.wv[v42]
            vectorW43 = model9.wv[v43]
            cos42 = cosine_similarity11(vectorW42, vectorW43)
        else:
            cos42 = 0
        cosALL6.insert(i, cos42)

        v44 = Add1_split[i][1]
        v45 = Add2_split[i][5]
        if v44 != 'NONE' and v45 != 'NONE':
            vectorW44 = model9.wv[v44]
            vectorW45 = model9.wv[v45]
            cos44 = cosine_similarity11(vectorW44, vectorW45)
        else:
            cos44 = 0
        cosALL7.insert(i, cos44)

        v46 = Add1_split[i][2]
        v47 = Add2_split[i][1]
        if v46 != 'NONE' and v47 != 'NONE':
            vectorW46 = model9.wv[v46]
            vectorW47 = model9.wv[v47]
            cos46 = cosine_similarity11(vectorW46, vectorW47)
        else:
            cos46 = 0
        cosALL8.insert(i, cos46)

        v48 = Add1_split[i][2]
        v49 = Add2_split[i][3]
        if v48 != 'NONE' and v49 != 'NONE':
            vectorW48 = model9.wv[v48]
            vectorW49 = model9.wv[v49]
            cos48 = cosine_similarity11(vectorW48, vectorW49)
        else:
            cos48 = 0
        cosALL9.insert(i, cos48)

        v50 = Add1_split[i][2]
        v51 = Add2_split[i][5]
        if v50 != 'NONE' and v51 != 'NONE':
            vectorW50 = model9.wv[v50]
            vectorW51 = model9.wv[v51]
            cos50 = cosine_similarity11(vectorW50, vectorW51)
        else:
            cos50 = 0
        cosALL10.insert(i, cos50)

        v52 = Add1_split[i][3]
        v53 = Add2_split[i][1]
        if v52 != 'NONE' and v53 != 'NONE':
            vectorW52 = model9.wv[v52]
            vectorW53 = model9.wv[v53]
            cos52 = cosine_similarity11(vectorW52, vectorW53)
        else:
            cos52 = 0
        cosALL11.insert(i, cos52)

        v54 = Add1_split[i][3]
        v55 = Add2_split[i][2]
        if v54 != 'NONE' and v55 != 'NONE':
            vectorW54 = model9.wv[v54]
            vectorW55 = model9.wv[v55]
            cos54 = cosine_similarity11(vectorW54, vectorW55)
        else:
            cos54 = 0
        cosALL12.insert(i, cos54)

        v56 = Add1_split[i][3]
        v57 = Add2_split[i][5]
        if v56 != 'NONE' and v57 != 'NONE':
            vectorW56 = model9.wv[v56]
            vectorW57 = model9.wv[v57]
            cos56 = cosine_similarity11(vectorW56, vectorW57)
        else:
            cos56 = 0
        cosALL13.insert(i, cos56)

        v58 = Add1_split[i][5]
        v59 = Add2_split[i][1]
        if v58 != 'NONE' and v59 != 'NONE':
            vectorW58 = model9.wv[v58]
            vectorW59 = model9.wv[v59]
            cos58 = cosine_similarity11(vectorW58, vectorW59)
        else:
            cos58 = 0
        cosALL14.insert(i, cos58)

        v60 = Add1_split[i][5]
        v61 = Add2_split[i][2]
        if v60 != 'NONE' and v61 != 'NONE':
            vectorW60 = model9.wv[v60]
            vectorW61 = model9.wv[v61]
            cos60 = cosine_similarity11(vectorW60, vectorW61)
        else:
            cos60 = 0
        cosALL15.insert(i, cos60)

        v62 = Add1_split[i][5]
        v63 = Add2_split[i][3]
        if v62 != 'NONE' and v63 != 'NONE':
            vectorW62 = model9.wv[v62]
            vectorW63 = model9.wv[v63]
            cos62 = cosine_similarity11(vectorW62, vectorW63)
        else:
            cos62 = 0
        cosALL16.insert(i, cos62)
        
        
        v36 = Add1_split[i][7]
        v37 = Add2_split[i][5]
        if v36 != 'NONE' and v37 != 'NONE':
            vectorW36 = model9.wv[v36]
            vectorW37 = model9.wv[v37]
            cos36 = round(cosine_similarity11(vectorW36, vectorW37),2)
        else:
            cos36 = 0
        cosALL3.insert(i, cos36)

        v38 = Add1_split[i][5]
        v39 = Add2_split[i][7]
        if v38 != 'NONE' and v39 != 'NONE':
            vectorW38 = model9.wv[v38]
            vectorW39 = model9.wv[v39]
            cos38 = round(cosine_similarity11(vectorW38, vectorW39),2)
        else:
            cos38 = 0
        cosALL4.insert(i, cos38)

    Label2L = []
    for i in range(0, len(df)):
        if df['NewLabel'][i] == 'NoMatch':
            Label2L.insert(i, 0)
        elif df['NewLabel'][i] == 'PartialMatch':
            Label2L.insert(i, 1)
        elif df['NewLabel'][i] == 'Match':
            Label2L.insert(i, 2)

    df2 = pd.DataFrame()

    se0 = pd.Series(cos0L)
    df2['cos0'] = se0.values

    se1 = pd.Series(cos1L)
    df2['cos1'] = se1.values

    se2 = pd.Series(cos2L)
    df2['cos2'] = se2.values

    se3 = pd.Series(cos3L)
    df2['cos3'] = se3.values

    se4 = pd.Series(cos4L)
    df2['cos4'] = se4.values

    se5 = pd.Series(cos5L)
    df2['cos5'] = se5.values

    se6 = pd.Series(cos6L)
    df2['cos6'] = se6.values
    
    se7= pd.Series(cos7L)
    df2['cos7'] = se7.values


    se12= pd.Series(cosALL3)
    df2['cosExtra3'] = se12.values

    se13= pd.Series(cosALL4)
    df2['cosExtra4'] = se13.values

    se13 = pd.Series(cosALL5)
    df2['cos9'] = se13.values

    se14 = pd.Series(cosALL6)
    df2['cos10'] = se13.values

    se15 = pd.Series(cosALL7)
    df2['cos11'] = se15.values

    se16 = pd.Series(cosALL8)
    df2['cos12'] = se16.values

    se17 = pd.Series(cosALL9)
    df2['cos13'] = se17.values

    se18 = pd.Series(cosALL10)
    df2['cos14'] = se18.values

    se19 = pd.Series(cosALL11)
    df2['cos15'] = se19.values

    se20 = pd.Series(cosALL12)
    df2['cos16'] = se20.values

    se21 = pd.Series(cosALL13)
    df2['cos17'] = se21.values

    se22 = pd.Series(cosALL14)
    df2['cos18'] = se22.values

    se23 = pd.Series(cosALL15)
    df2['cos19'] = se23.values

    se24 = pd.Series(cosALL16)
    df2['cos20'] = se24.values
    
    df2['INBUILDING'] = df['INBUILDING']
    df2['EXTBUILDING'] = df['EXTBUILDING']
    df2['POILOGISTIC'] = df['POILOGISTIC']
    df2['ZONE'] = df['ZONE']
    df2['HOUSENUM'] = df['HOUSENUM']
    df2['ROADNAME'] = df['ROADNAME']
    df2['CITY'] = df['CITY']
    df2['DISTRICT'] = df['DISTRICT']

    
    df2['INBUILDINGK'] = df['INBUILDINGK']
    df2['EXTBUILDINGK'] = df['EXTBUILDINGK']
    df2['POILOGISTICK'] = df['POILOGISTICK']
    df2['ZONEK'] = df['ZONEK']
    df2['HOUSENUMK'] = df['HOUSENUMK']
    df2['ROADNAMEK'] = df['ROADNAMEK']
    df2['CITYK'] = df['CITYK']
    df2['DISTRICTK'] = df['DISTRICTK']

    se9 = pd.Series(Label2L)
    df2['Label'] = se9.values

    return df2

