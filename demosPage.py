import streamlit as st
import numpy as np
import pandas as pd
from demandPrediction import productPredDemand


def customerChurn():
    st.title('Fuga de clientes')
    
def sentimentAnalisis():
    st.title('Análisis de sentimientos')
    
def main():
    st.sidebar.title("""Demos algoritmos predictivos""")

    modes = ["Fuga de clientes", "Predicción de demanda", "Análisis de sentimientos"]
    apps = [customerChurn,productPredDemand,sentimentAnalisis]

    app_mode = st.sidebar.selectbox("Escoja el modelo a visualizar",
        modes)

    chosenIndex= modes.index(app_mode)
    apps[chosenIndex]()

if __name__ == "__main__":
    main()