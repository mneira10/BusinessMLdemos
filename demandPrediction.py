import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


@st.cache
def load_data():
    data = pd.read_csv('./demandForecasting/train.csv',
                       parse_dates=['date'], index_col=['date'])

    store_item_table = pd.read_csv('./demandForecasting/store_item_table.csv')
    month_table = pd.read_csv('./demandForecasting/month_table.csv')
    dow_table = pd.read_csv('./demandForecasting/dow_table.csv')
    year_table = pd.read_csv('./demandForecasting/year_table.csv')
    return data, store_item_table, month_table, dow_table, year_table


def productPredDemand():
    st.title('Predicción de demanda')
    st.write(
        'Tomamos los datos de una [competencia de kaggle](https://www.kaggle.com/c/demand-forecasting-kernels-only/data).')

    data_load_state = st.text('Cargando datos...')
    data, store_item_table, month_table, dow_table, year_table = load_data()
    data_load_state.empty()

    st.write('Los datos tienen la información de las ventas diarias de {} productos para {} tiendas desde Enero del 2013 hasta Diciembre del 2017. Veamos un par de entradas: '.format(
        len(data.item.unique()), len(data.store.unique())))

    st.write(data.head(20))

    st.write('## Distribución de los datos')

    st.write('Miremos el comportamiento de las ventas de 10 productos de una tienda:')

    dataplot = []
    for i in range(10):
        dataplot.append(go.Scatter(
            x=data[(data.store == 1) & (data.item == i+1)].index,
            y=data[(data.store == 1) & (data.item == i+1)].sales,
            mode='lines',
            name='Product {}'.format(i+1)
        ))
    fig = go.Figure(dataplot)
    fig.update_layout(
        xaxis=go.layout.XAxis(
            title_text="Fecha",
            title_font={"size": 20},
        ),
        yaxis=go.layout.YAxis(
            title_text="Ventas",
            title_font={"size": 20},

        )
    )
    st.plotly_chart(fig)

    st.write('Notemos la periodicidad anual. Cada año hay una "montaña" y esa "montaña" crece año a año.')

    st.write('Veamos un producto en particular:')

    singleFig = go.Figure([go.Scatter(
        x=data[(data.store == 1) & (data.item == 35)].index,
        y=data[(data.store == 1) & (data.item == 35)].sales,
        mode='lines',
        name='Product {}'.format(35)
    )])
    singleFig.update_layout(
        xaxis=go.layout.XAxis(
            title_text="Fecha",
            title_font={"size": 20},
        ),
        yaxis=go.layout.YAxis(
            title_text="Ventas",
            title_font={"size": 20},

        )
    )

    st.plotly_chart(singleFig)

    st.write('## Predicciones de compras')

    actualTrain = data.loc[:'2016-01-01']

    grand_avg = 48.6017700729927

    years = np.arange(2013, 2018)

    year_table = year_table.set_index('year')
    month_table = month_table.set_index('month')
    dow_table = dow_table.set_index('dayofweek')
    store_item_table = store_item_table.set_index('store')

    annual_sales_avg = year_table.values.squeeze()

    p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg[:-1], 2))

    annual_growth = p2

    predDates = []
    predSales = []

    item = st.number_input('Predecir ventas para el producto:',
                           min_value=1, max_value=50, value=35)

    # item = 35
    store = 1
    # st.write(data.loc['2016-01-01':].index)

    for row in data.loc['2016-01-01':].index.unique():

        dow, month, year = row.dayofweek, row.month, row.year+1
        base_sales = store_item_table[str(item)].iloc[store-1]
        mul = month_table.at[month, 'sales'] * dow_table.at[dow, 'sales']
        pred_sales = base_sales * mul * annual_growth(year)

        predDates.append(row+pd.DateOffset(days=365))
        predSales.append(pred_sales)

    predPlot = []
    predPlot.append(go.Scatter(x=data[(data.item == item) & (data.store == store)].index, y=data[(data.item == item) & (data.store == store)].sales,
                               name='Data'))

    predPlot.append(go.Scatter(x=predDates, y=predSales,
                               #                     mode='lines',
                               name='Predicción'))

    predFig = go.Figure(predPlot)

    predFig.update_layout(
        title={
            'text': 'Producto {}'.format(item),
            'xanchor': 'center',
            # 'y': 0.9,
            'x': 0.5,
            'yanchor': 'top',
            'font':{'size':25}
            },
        xaxis=go.layout.XAxis(
            title_text="Fecha",
            title_font={"size": 20},
        ),
        yaxis=go.layout.YAxis(
            title_text="Ventas",
            title_font={"size": 20},

        )

    )

    st.write('Entrenamos con los datos desde Enero del 2013 hasta Diciembre del 2016. Le pedimos al algoritmo que prediga valores para el periodo de Enero 2016-Diciembre 2018:')

    st.plotly_chart(predFig)
