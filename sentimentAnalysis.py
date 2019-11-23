import streamlit as st 
from classifier import *


def sentimentAnalisis():


    st.title('Análisis de sentimientos')

    st.write('En muchos casos es útil saber, de forma automatizada, el sentimiento de el texto. Ya sea de tweets de la empresa, retroalimentacion de los clientes, opiniones de los empleados, etc. El siguiente modelo toma texto como input y retorna un número entre -1 y 1 reflejando el sentimiento del text. -1 corresponde a un comentario extremadamente negativo y 1 a un comentario extremadamente positivo. Veamos algunos ejemplos:')

    clf = SentimentClassifier()
    def predictVal(text):
        return clf.predict(text)*2 -1
    
    sampletexts = ['Muy buen producto! Lo recomendaría.','Pésima atención. No volvería nunca.']

    for t in sampletexts:

        st.code(t)
        st.write('**Sentimiento: {:.4f}**'.format(predictVal(t)))

    st.write('## Predicción dinámica')

    userInput = st.text_input('Pruebe usted con su propio texto:','Esto funciona muy bien!')

    st.write('**Sentimiento: {:.4f}**'.format(predictVal(userInput)))
    