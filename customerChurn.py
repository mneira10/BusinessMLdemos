import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

def customerChurn():

    st.title('Ejemplo de Fuga de Cientes')

    st.write('Usaremos un [dataset obtenido de kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) para esta demostracion. Los datos son los siguientes:')

    #load csv data
    data = pd.read_csv('./customerChurn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data = data[data['TotalCharges']!=' ']

    inputVars = list(data.columns)
    inputVars.remove('Churn')
    inputVars.remove('customerID')

    def encodeUser(user,hasChurn=True):
        usercolumns = list(data.columns)
        if not hasChurn:
            usercolumns.remove('Churn')
        for col in usercolumns:
            if col in mappings:
                print(col)
                user[col] = mappings[col][user[col]]
        return user

    #load dict mappings
    with open('./customerChurn/mappings.pkl', 'rb') as fp:
        mappings = pickle.load(fp)

    #load classifier
    clf = load('./customerChurn/simpleRF.clf') 

    st.write(data)

    st.markdown('''
    Hay un total de {} columnas el el dataset: 
    '''.format(len(data.columns)))

    listCols = ''
    for i,col in enumerate(data.columns):
        listCols += str(i+1)+'. '+col + '\n'

    st.write(listCols)

    st.write('''
    En este caso, la columna a predecir es **Churn**, donde Yes indica que el cliente se ha fugado y No de lo contrario.

    Con los atributos de cada usuario, se puede predecir (hasta cierto punto), que usuarios se van a fugar. Hemos implementado un modelo de machine learning con una **precisión de 52.85%**, un **recall de 79.44%** y un **F1-score de 63.47%**. Veamos un ejemplo de uso con este usuario:
    ''')

    user = data.iloc[0]

    st.write(user)
    user = encodeUser(user)
    wellFormatedUser = user[inputVars].values.reshape(1,-1)
    pred = clf.predict_proba(wellFormatedUser)
    st.write('''Nuestro algoritmo predice que el usuario se fugará con una probabilidad del: 
    **{:.2f}%**'''.format(pred[0][1]*100))

    cat_cols = list(data.select_dtypes(include=['object']))
    cat_cols.remove('customerID')
    cat_cols.remove('Churn')
    cat_cols.remove('TotalCharges')

    st.write('## Predicción dinámica')
    st.write('Seleccione las características del usuario para tener una predicción en tiempo real:')

    options = []

    for col in cat_cols:
        option = st.selectbox(
            col,
            tuple(data[col].unique()))

        options.append(option)

    numOptionNames = ['MonthlyCharges', 'SeniorCitizen', 'tenure','TotalCharges']
    numOptionVals = []
    for name in numOptionNames:
        numVal = st.text_input(name, 23.3)
        numOptionVals.append(numVal)

    dfVals = options+numOptionVals
    dfCols = cat_cols+numOptionNames

    newUser = pd.DataFrame([dfVals], 
                columns =dfCols) 

    st.write('Para el usuario:')
    st.write(newUser)
    wellFormatedNewUser = encodeUser(newUser.iloc[0],hasChurn=False)
    wellFormatedNewUser = wellFormatedNewUser.values.reshape(1,-1)
    pred = clf.predict_proba(wellFormatedNewUser)
    st.write('La probabilidad de que se fugue el cliente es de: {:.2f}%'.format(pred[0][1]*100))

