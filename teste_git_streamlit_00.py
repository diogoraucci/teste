

import streamlit as st
import pandas as pd
import openpyxl
from math import sqrt
import numpy as np
from datetime import date

import plotly.graph_objects as go

import math
from sklearn.linear_model import LinearRegression

#df = pd.read_excel(r'./DF_Cotacoes.xlsx', sheet_name='Cotacoes', index_col=0)
url = "https://github.com/diogoraucci/teste/raw/main/DF_Cotacoes.xlsx"
df = pd.read_excel(url, sheet_name='Cotacoes', index_col=0, engine='openpyxl')
st.dataframe(df)

# Obter dados do Bitcoin usando o símbolo "BTC-USD"
tickList = df.columns.to_list()
select_tickers = tickList[0]
select_MM = 90

lineColor=  'black'
horizontalLineColor = 'black'
dotColor1 = 'black'
dotColor2 = 'black'
dotColor3 = 'black'
MMlineColor = 'blue'

# Coleta das cotações
ts = pd.DataFrame(df.iloc[:, 0])

# Cálculo da média móvel
ts['MM'] = ts.iloc[:,0].rolling(select_MM).mean()
ts.fillna(method='bfill', inplace=True)

# Calcular Retorno Logaritmico da Média Móvel
ts['mm%'] = ts.apply(lambda x: math.log(x[0] / x[1]), axis=1)  # /mm

# Calcuular Variação Diária
ts['pct'] = np.log(ts.iloc[:, [0]].pct_change() + 1)
ts['pct'].fillna(method='bfill', inplace=True)

# REGRESSÃO LINEAR
X_independent = ts['pct'].values.reshape(-1, 1)
Y_dependent = ts['mm%'].values.reshape(-1, 1)

reg = LinearRegression().fit(X_independent, Y_dependent)

# Gerando Reta da regressao------------------------------------------------------->>>>>>>>>>>
Y_predict = reg.predict(X_independent);
# Calculando residuos
ts['Resíduos'] = (Y_dependent - Y_predict)

# Gráfico dos Resíduos
mean = ts['Resíduos'].mean()
std = ts['Resíduos'].std()

ts['1std+'] = std
ts['1std-'] = std * -1
ts['2std+'] = std * 2
ts['2std-'] = std * -2
ts['3std+'] = std * 3
ts['3std-'] = std * -3
ts['zero'] = mean

# Plot do gráfico Precos
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts.iloc[:,0], name='BTC', mode='lines',
                         line=dict(color=lineColor, width=2)))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['MM'], name=f'Média Móvel {select_MM} períodos', mode='lines',
               line=dict(color=MMlineColor, width=2)))
# Incluir Pontos

fig.add_trace(go.Scatter(
    x=ts[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-'])].index,
    y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-']), select_tickers],
    mode='markers', marker=dict(color=dotColor1, size=5), showlegend=False
))

fig.add_trace(go.Scatter(
    x=ts[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-'])].index,
    y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-']), select_tickers],
    mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
))

fig.add_trace(go.Scatter(
    x=ts[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-'])].index,
    y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-']), select_tickers],
    mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
))

fig.update_layout(title=f'Cotações de {select_tickers} Média Móvel de {select_MM} períodos')

# Define a legenda na parte interna do gráfico
fig.update_layout(legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'), autosize=True,
                  height=500)

# Exibir 8 datas no eixo x
num_dates = 5
tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
tick_values = [ts.index[0]] + tick_values.tolist() + [
    ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

# Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Sinalizando Entradas ============================================================================
# VENDAS
# Criando uma nova coluna que indica se o valor é maior que o limite superior de 3 desvios padrão
ts['acima_3std'] = ts['Resíduos'] >= ts['3std+']
ts['acima_2std'] = (ts['Resíduos'] >= ts['2std+']) & (ts['Resíduos'] < ts['3std+'])
ts['acima_1std'] = (ts['Resíduos'] >= ts['1std+']) & (ts['Resíduos'] < ts['2std+'])
# COMPRAS
ts['abaixo_3std'] = ts['Resíduos'] <= ts['3std-']
ts['abaixo_2std'] = (ts['Resíduos'] <= ts['2std-']) & (ts['Resíduos'] > ts['3std-'])
ts['abaixo_1std'] = (ts['Resíduos'] <= ts['1std-']) & (ts['Resíduos'] > ts['2std-'])

# ==================================================
# Plot do gráfico de Resíduos
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['Resíduos'], name='Resíduos', mode='lines',
               line=dict(color=lineColor, width=2)))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['zero'], mode='lines', line=dict(color=MMlineColor, width=1, dash='solid')))

fig.add_trace(go.Scatter(x=ts.index, y=ts['1std+'], mode='lines',
                         line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['2std+'], mode='lines',
               line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['3std+'], mode='lines',
               line=dict(color=horizontalLineColor, width=0.2, dash='dot')))
fig.add_trace(go.Scatter(x=ts.index, y=ts['1std-'], mode='lines',
                         line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['2std-'], mode='lines',
               line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
fig.add_trace(
    go.Scatter(x=ts.index, y=ts['3std-'], mode='lines',
               line=dict(color=horizontalLineColor, width=0.2, dash='dot')))


fig.add_trace(go.Scatter(x=ts[ts['abaixo_1std']].index, y=ts.loc[ts['abaixo_1std'], 'Resíduos'], mode='markers',
                         marker=dict(color=dotColor1, size=5)))

fig.add_trace(go.Scatter(x=ts[ts['abaixo_2std']].index, y=ts.loc[ts['abaixo_2std'], 'Resíduos'], mode='markers',
                         marker=dict(color=dotColor2, size=10)))

fig.add_trace(go.Scatter(x=ts[ts['abaixo_3std']].index, y=ts.loc[ts['abaixo_3std'], 'Resíduos'], mode='markers',
                         marker=dict(color=dotColor3, size=15)))

fig.update_layout(title=f'Gráfico Normalizado {select_tickers} Média Móvel de {select_MM} períodos')
fig.update_layout(showlegend=False)  # Remove as legendas
# Remover o eixo Y
fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False))
fig.update_layout(yaxis=dict(showline=False, zeroline=False))
# Define a legenda na parte interna do gráfico
fig.update_layout(legend=dict(x=0, y=1.1, orientation='h', traceorder='normal'), autosize=True, height=400)

# Exibir 8 datas no eixo x
num_dates = 5
tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
tick_values = [ts.index[0]] + tick_values.tolist() + [
    ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

# Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
st.plotly_chart(fig, theme="streamlit", use_container_width=True)