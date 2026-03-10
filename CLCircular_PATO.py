# Librerías
import pdb
from pdb import pm as pdb_pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.datasets import grunfeld
from linearmodels.panel import PanelOLS
from linearmodels . panel import RandomEffects
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import compare
from linearmodels.panel import PooledOLS
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statistics as stats # estadística
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pylab import info, rcParams
from statsmodels.tsa.arima.model import ARIMA
from numpy._core.fromnumeric import transpose
import itertools
from semopy import Model, Optimizer, calc_stats
import openpyxl
import os
from pathlib import Path
import streamlit as st
import plotly.express as px
import glob
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent

### Carga Data Market Size Categorías
df_mst = pd.read_excel(BASE_DIR / 'db1_internacional_pharma.xlsx', 'Mkt.Size', header = 5)
df_mst.tail()

### Limpieza y proceso de Data
df_mst.isnull().sum()

df_mst.dropna(inplace=True)
df_mst.isnull().sum()

df_mstg = df_mst[df_mst['Category'] != 'Consumer Health']
df_mstg.head()

### Data a Serie de Tiempo
years = [str(year) for year in range(2011, 2026)]

df_mstj = df_mst.melt(id_vars=['Geography', 'Category', 'Data Type', 'Unit', 'Current Constant'],
                            value_vars=years,
                            var_name='Year',
                            value_name='Value')

### Indexamos Year
df_mstj['Year'] = pd.to_numeric(df_mstj['Year'])
df_mstj

df_mstj = df_mstj.set_index('Year')
df_mstj.head()

### Ajustamos los valores a USD
df_mstj['Value'] = pd.to_numeric(df_mstj['Value'], errors='coerce')

df_mstj['Value'] = np.where(df_mstj['Unit'] == 'MXN million', df_mstj['Value']/17.7, df_mstj['Value'])
df_mstj
df_mstj.head()

df_mstj['Unit'] = 'Million USD'
df_mstj.head()

print(df_mst['Category'].unique())

### Seleccionamos las categorías de interés para el análisis
selected_categories = ['OTC', 'Sports Nutrition', 'Vitamins and Dietary Supplements', 'Weight Management and Wellbeing', 'Herbal/Traditional Products', 'Allergy Care']
df_prog = df_mstj[df_mstj['Category'].isin(selected_categories)]
df_prog

df_mexusa_mstj = df_prog[df_prog['Geography'].isin(['Mexico', 'USA'])]
print(df_mexusa_mstj)

### Gráfica Valor categoría por país
df_plot = df_mexusa_mstj.reset_index()

plt.figure(figsize=(16, 9))
sns.lineplot(data=df_plot, x='Year', y='Value', hue='Category', marker='o')
plt.title('Evolución del Valor por Categoría y País (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_plot['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### Evolución Categoría por país (Sin Background)
plt.figure(figsize=(16, 9))
sns.lineplot(data=df_mexusa_mstj, x='Year', y='Value', hue='Category', style='Geography', marker='o')
plt.title('Evolución del Valor por Categoría en México y USA (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_mexusa_mstj['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### Evolución México
df_mex = df_mexusa_mstj[df_mexusa_mstj['Geography'] == 'Mexico']
df_mex = df_mex.reset_index()
df_mex.head()

plt.figure(figsize=(16, 9))
sns.lineplot(data=df_mex, x='Year', y='Value', hue='Category', style='Geography', marker='o')
plt.title('Evolución del Valor por Categoría en México (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_mex['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### Evolución USA
df_usa = df_mexusa_mstj[df_mexusa_mstj['Geography'] == 'USA']
df_usa = df_usa.reset_index()
df_usa.head()

plt.figure(figsize=(16, 9))
sns.lineplot(data=df_usa, x='Year', y='Value', hue='Category', style='Geography', marker='o')
plt.title('Evolución del Valor por Categoría en USA (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_usa['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
### Dentro de la industria farmacéutica en el mercado estadounidense, notamos que los principales secores son 
# OTC, Sports Nutrition, Vitamins and Dietary Supplements y Herbal/Traditional Products.

### ARIMA - SARIMA para OTC en USA
df_otc_usa = df_usa[df_usa['Category'] == 'OTC']
df_otc_usa = df_otc_usa.set_index('Year')
df_otc_usa.head()

df_otc_usa.index = pd.to_datetime(df_otc_usa.index.astype(str), format='%Y')

plt.figure(figsize=(12, 6))
plt.plot(df_otc_usa.index, df_otc_usa['Value'])
plt.title('Evolución de Viajes Potenciales a lo largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Viajes Potenciales')
plt.grid(True)
plt.show()

dec = sm.tsa.seasonal_decompose(df_otc_usa['Value'],period = 4, model = 'additive').plot()
plt.show()

### Prueba ACF y PACF
### ACF
plot_acf(df_otc_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación (ACF) para Ventas')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('ACF')
plt.show()
acf_valores = acf(df_otc_usa['Value'], nlags=7)
print(acf_valores) ### q = 0

### PACF
plot_pacf(df_otc_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación Parcial (PACF) para Value')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('PACF')
plt.show()
pacf_valores = pacf(df_otc_usa['Value'], nlags=7)
print(pacf_valores) ### p = 1

### Arima (1, 1, 1)
model = ARIMA(df_otc_usa['Value'], order=(1, 1, 1))
model_fit = model.fit()
# Pronóstico
print(model_fit.summary())
forecast = model_fit.forecast(steps=10)

last_date = df_otc_usa.index.max()
forecast_index = pd.date_range(start=pd.to_datetime(last_date) + pd.DateOffset(years=1), periods=len(forecast), freq='YS')
forecast.index = forecast_index

print("Forecast Series with new DatetimeIndex:")
forecast

forecast_results_obj = model_fit.get_forecast(steps=len(forecast))
conf_int = forecast_results_obj.conf_int()

conf_int['Esperado'] = forecast
conf_int = conf_int[['lower Value', 'Esperado', 'upper Value']]
conf_int.head()

conf_int.info()
print("\nIndex type:", conf_int.index)
print("Index Dtype:", conf_int.index.dtype)

plt.figure(figsize=(10,6))

# dibujo de la serie histórica
plt.plot(df_otc_usa.index, df_otc_usa['Value'],
         label='Observado', marker='o')

# dibujo del pronóstico
plt.plot(forecast.index, forecast,
         label='Forecast', marker='o', linestyle='--', color='C1')

plt.title('Pronóstico OTC‑USA')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

model = ARIMA(df_otc_usa['Value'], order=(1, 1, 1))
model_fit = model.fit()

# obtener forecast + banda de confianza 95 %
fc_res = model_fit.get_forecast(steps=10)
forecast = fc_res.predicted_mean            # Series
conf_int = fc_res.conf_int(alpha=0.05)      # DF con columnas ‘lower Value’ y ‘upper Value’

# convertir los índices a años (ver mensaje anterior)
last_date = df_otc_usa.index.max()
forecast_index = pd.date_range(start=pd.to_datetime(last_date) + pd.DateOffset(years=1),
                               periods=len(forecast),
                               freq='YS')
forecast.index = forecast_index
conf_int.index = forecast_index

# trazo
plt.figure(figsize=(10,6))
plt.plot(df_otc_usa.index, df_otc_usa['Value'], label='Observado', marker='o')
plt.plot(forecast.index, forecast, label='Pronóstico', linestyle='--', color='C1')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:,0],          # límite inferior
                 conf_int.iloc[:,1],          # límite superior
                 color='C1', alpha=0.3,
                 label='IC 95%')
plt.title('Modelo ARIMA – Observado y pronóstico')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### SARIMA
serie_temporal = df_otc_usa['Value']
model_SARIMA = sm.tsa.SARIMAX(serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

# Train the model
resultado = model_SARIMA.fit(disp=False)

# See results
print(resultado.summary())

### Forecast SARIMA
SARIMA_pred = resultado.get_forecast(steps=10)

# Print the forecasted values
print("Forecasted Values:")
print(SARIMA_pred.predicted_mean)

SARIMA_pred = resultado.get_forecast(steps=10)
ic = SARIMA_pred.conf_int()

# Create a proper DatetimeIndex for the forecast
last_date = df_otc_usa.index.max()
forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=10, freq='YS')

# Assign the new DatetimeIndex to the confidence interval DataFrame
ic.index = forecast_index

print("DataFrame 'ic' with updated DatetimeIndex:")
print(ic)
### Gráfica forecast SARIMA con IC
forecast_sarima = SARIMA_pred.predicted_mean
forecast_sarima.index = ic.index  

plt.figure(figsize=(10,6))

# serie histórica
plt.plot(df_otc_usa.index, df_otc_usa['Value'],
         label='Observado', marker='o')

plt.plot(forecast_sarima.index, forecast_sarima,
         label='Forecast SARIMA', linestyle='--', color='C1', marker='o')

# banda de confianza (IC) usando el DataFrame ic
plt.fill_between(ic.index,
                 ic.iloc[:,0],      # lower Value
                 ic.iloc[:,1],      # upper Value
                 color='C1', alpha=0.25,
                 label='IC 95%')

plt.title('SARIMA – Observado y pronóstico con IC')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### SARIMA - Vitamins and Dietary Supplements en USA
df_vds_usa = df_usa[df_usa['Category'] == 'Vitamins and Dietary Supplements']
df_vds_usa = df_vds_usa.set_index('Year')
df_vds_usa.head()

df_vds_usa.index = pd.to_datetime(df_vds_usa.index.astype(str), format='%Y')

plt.figure(figsize=(12, 6))
plt.plot(df_vds_usa.index, df_vds_usa['Value'])
plt.title('Evolución de Viajes Potenciales a lo largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Viajes Potenciales')
plt.grid(True)
plt.show()

dec = sm.tsa.seasonal_decompose(df_vds_usa['Value'],period = 5, model = 'additive').plot()
plt.show()

### Prueba ACF y PACF
### ACF
plot_acf(df_vds_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación (ACF) para Ventas')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('ACF')
plt.show()
acf_valores = acf(df_vds_usa['Value'], nlags=7)
print(acf_valores) ### q = 0

### PACF
plot_pacf(df_vds_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación Parcial (PACF) para Value')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('PACF')
plt.show()
pacf_valores = pacf(df_vds_usa['Value'], nlags=7)
print(pacf_valores) ### p = 1

### SARIMA
serie_temporal = df_vds_usa['Value']
model_SARIMA = sm.tsa.SARIMAX(serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

# Train the model
resultado = model_SARIMA.fit(disp=False)

# See results
print(resultado.summary())

### Forecast SARIMA
SARIMA_pred = resultado.get_forecast(steps=10)

# Print the forecasted values
print("Forecasted Values:")
print(SARIMA_pred.predicted_mean)

SARIMA_pred = resultado.get_forecast(steps=10)
ic = SARIMA_pred.conf_int()

# Create a proper DatetimeIndex for the forecast
last_date = df_vds_usa.index.max()
forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=10, freq='YS')

# Assign the new DatetimeIndex to the confidence interval DataFrame
ic.index = forecast_index

print("DataFrame 'ic' with updated DatetimeIndex:")
print(ic)
### Gráfica forecast SARIMA con IC
forecast_sarima = SARIMA_pred.predicted_mean
forecast_sarima.index = ic.index  

plt.figure(figsize=(10,6))

# serie histórica
plt.plot(df_vds_usa.index, df_vds_usa['Value'],
         label='Observado', marker='o')

plt.plot(forecast_sarima.index, forecast_sarima,
         label='Forecast SARIMA', linestyle='--', color='C1', marker='o')

# banda de confianza (IC) usando el DataFrame ic
plt.fill_between(ic.index,
                 ic.iloc[:,0],      # lower Value
                 ic.iloc[:,1],      # upper Value
                 color='C1', alpha=0.25,
                 label='IC 95%')

plt.title('SARIMA – Observado y pronóstico con IC')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### SARIMA - Sports Nutrition en USA
df_sn_usa = df_usa[df_usa['Category'] == 'Sports Nutrition']
df_sn_usa = df_sn_usa.set_index('Year')
df_sn_usa.head()

df_sn_usa.index = pd.to_datetime(df_sn_usa.index.astype(str), format='%Y')

plt.figure(figsize=(12, 6))
plt.plot(df_sn_usa.index, df_sn_usa['Value'])
plt.title('Evolución de Value Sports Nutrition')
plt.xlabel('Fecha')
plt.ylabel('Value')
plt.grid(True)
plt.show()

dec = sm.tsa.seasonal_decompose(df_sn_usa['Value'],period = 4, model = 'additive').plot()
plt.show()

### Prueba ACF y PACF
### ACF
plot_acf(df_sn_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación (ACF) para Ventas')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('ACF')
plt.show()
acf_valores = acf(df_sn_usa['Value'], nlags=7)
print(acf_valores) ### q = 0

### PACF
plot_pacf(df_sn_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación Parcial (PACF) para Value')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('PACF')
plt.show()
pacf_valores = pacf(df_sn_usa['Value'], nlags=7)
print(pacf_valores) ### p = 1

### SARIMA
serie_temporal = df_sn_usa['Value']
model_SARIMA = sm.tsa.SARIMAX(serie_temporal, order=(0, 1, 1), seasonal_order=(0, 1, 1, 9))

# Train the model
resultado = model_SARIMA.fit(disp=False)

# See results
print(resultado.summary())

### Forecast SARIMA
SARIMA_pred = resultado.get_forecast(steps=10)

# Print the forecasted values
print("Forecasted Values:")
print(SARIMA_pred.predicted_mean)

SARIMA_pred = resultado.get_forecast(steps=10)
ic = SARIMA_pred.conf_int()

# Create a proper DatetimeIndex for the forecast
last_date = df_sn_usa.index.max()
forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=10, freq='YS')

# Assign the new DatetimeIndex to the confidence interval DataFrame
ic.index = forecast_index

print("DataFrame 'ic' with updated DatetimeIndex:")
print(ic)
### Gráfica forecast SARIMA con IC
forecast_sarima = SARIMA_pred.predicted_mean
forecast_sarima.index = ic.index  

plt.figure(figsize=(10,6))

# serie histórica
plt.plot(df_sn_usa.index, df_sn_usa['Value'],
         label='Observado', marker='o')

plt.plot(forecast_sarima.index, forecast_sarima,
         label='Forecast SARIMA', linestyle='--', color='C1', marker='o')

# banda de confianza (IC) usando el DataFrame ic
plt.fill_between(ic.index,
                 ic.iloc[:,0],      # lower Value
                 ic.iloc[:,1],      # upper Value
                 color='C1', alpha=0.25,
                 label='IC 95%')

plt.title('SARIMA – Observado y pronóstico con IC')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### Buscamos entender el comportamiento de los subsegmentos de OTC
df_mexusa_mstj
print(df_mexusa_mstj['Category'].unique())
print(df_mstj['Category'].unique())

selected_categories = ['Adult Mouth Care', 'Analgesics', 'Sleep Aids', 'Cough, Cold and Allergy (Hay Fever) Remedies', 'Dermatologicals',
                       'Digestive Remedies', 'Emergency Contraception', 'Eye Care', 'NRT Smoking Cessation Aids', 'Wound Care']

df_usa = df_mstj[df_mstj['Geography'] == 'USA']
df_usa = df_usa.reset_index()
df_usa.head()

df_prog = df_usa[df_usa['Category'].isin(selected_categories)]
df_prog

### Gráfica Categorías OTC en USA
plt.figure(figsize=(16, 9))
sns.lineplot(data=df_prog, x='Year', y='Value', hue='Category', style='Geography', marker='o')
plt.title('Evolución del Valor por Categoría en USA (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_prog['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

selected_categories = ['Cough, Cold and Allergy (Hay Fever) Remedies', 'Digestive Remedies', 'Analgesics', 'Adult Mouth Care', 'Sleep Aids']

df_usa = df_prog[df_prog['Geography'] == 'USA']
df_usa = df_usa.reset_index()
df_usa.head()

df_prog = df_usa[df_usa['Category'].isin(selected_categories)]
df_prog

### Gráfica Categorías OTC en USA - Principales subsegmentos
plt.figure(figsize=(16, 9))
sns.lineplot(data=df_prog, x='Year', y='Value', hue='Category', style='Geography', marker='o')
plt.title('Evolución del Valor por Categoría en USA (2011-2025)')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_prog['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

### Sarima para Cough, Cold and Allergy (Hay Fever) Remedies
df_ccar_usa = df_usa[df_usa['Category'] == 'Cough, Cold and Allergy (Hay Fever) Remedies']
df_ccar_usa = df_ccar_usa.set_index('Year')
df_ccar_usa.head()

df_ccar_usa.index = pd.to_datetime(df_ccar_usa.index.astype(str), format='%Y')

plt.figure(figsize=(12, 6))
plt.plot(df_ccar_usa.index, df_ccar_usa['Value'])
plt.title('Evolución de Value Cough, Cold and Allergy (Hay Fever) Remedies')
plt.xlabel('Fecha')
plt.ylabel('Value')
plt.grid(True)
plt.show()

dec = sm.tsa.seasonal_decompose(df_ccar_usa['Value'],period = 4, model = 'additive').plot()
plt.show()

### Prueba ACF y PACF
### ACF
plot_acf(df_ccar_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación (ACF) para Ventas')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('ACF')
plt.show()
acf_valores = acf(df_ccar_usa['Value'], nlags=7)
print(acf_valores) ### q = 0

### PACF
plot_pacf(df_ccar_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación Parcial (PACF) para Value')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('PACF')
plt.show()
pacf_valores = pacf(df_ccar_usa['Value'], nlags=7)
print(pacf_valores) ### p = 1

### SARIMA
serie_temporal = df_ccar_usa['Value']
model_SARIMA = sm.tsa.SARIMAX(serie_temporal, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))

# Train the model
resultado = model_SARIMA.fit(disp=False)

# See results
print(resultado.summary())

### Forecast SARIMA
SARIMA_pred = resultado.get_forecast(steps=10)

# Print the forecasted values
print("Forecasted Values:")
print(SARIMA_pred.predicted_mean)

SARIMA_pred = resultado.get_forecast(steps=10)
ic = SARIMA_pred.conf_int()

# Create a proper DatetimeIndex for the forecast
last_date = df_ccar_usa.index.max()
forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=10, freq='YS')

# Assign the new DatetimeIndex to the confidence interval DataFrame
ic.index = forecast_index

print("DataFrame 'ic' with updated DatetimeIndex:")
print(ic)
### Gráfica forecast SARIMA con IC
forecast_sarima = SARIMA_pred.predicted_mean
forecast_sarima.index = ic.index  

plt.figure(figsize=(10,6))

# serie histórica
plt.plot(df_ccar_usa.index, df_ccar_usa['Value'],
         label='Observado', marker='o')

plt.plot(forecast_sarima.index, forecast_sarima,
         label='Forecast SARIMA', linestyle='--', color='C1', marker='o')

# banda de confianza (IC) usando el DataFrame ic
plt.fill_between(ic.index,
                 ic.iloc[:,0],      # lower Value
                 ic.iloc[:,1],      # upper Value
                 color='C1', alpha=0.25,
                 label='IC 95%')

plt.title('SARIMA – Observado y pronóstico con IC')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### Analgesics
df_a_usa = df_usa[df_usa['Category'] == 'Analgesics']
df_a_usa = df_a_usa.set_index('Year')
df_a_usa.head()

df_a_usa.index = pd.to_datetime(df_a_usa.index.astype(str), format='%Y')

plt.figure(figsize=(12, 6))
plt.plot(df_a_usa.index, df_a_usa['Value'])
plt.title('Evolución de Value Analgesics')
plt.xlabel('Fecha')
plt.ylabel('Value')
plt.grid(True)
plt.show()

dec = sm.tsa.seasonal_decompose(df_a_usa['Value'],period = 4, model = 'additive').plot()
plt.show()

### Prueba ACF y PACF
### ACF
plot_acf(df_a_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación (ACF) para Ventas')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('ACF')
plt.show()
acf_valores = acf(df_a_usa['Value'], nlags=7)
print(acf_valores) ### q = 0

### PACF
plot_pacf(df_a_usa['Value'], lags=7, alpha=0.05)
plt.title('Gráfico de Autocorrelación Parcial (PACF) para Value')
plt.xlabel('Rezagos (Lags)')
plt.ylabel('PACF')
plt.show()
pacf_valores = pacf(df_a_usa['Value'], nlags=7)
print(pacf_valores) ### p = 1

### SARIMA
serie_temporal = df_a_usa['Value']
model_SARIMA = sm.tsa.SARIMAX(serie_temporal, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))

# Train the model
resultado = model_SARIMA.fit(disp=False)

# See results
print(resultado.summary())

### Forecast SARIMA
SARIMA_pred = resultado.get_forecast(steps=10)

# Print the forecasted values
print("Forecasted Values:")
print(SARIMA_pred.predicted_mean)

SARIMA_pred = resultado.get_forecast(steps=10)
ic = SARIMA_pred.conf_int()

# Create a proper DatetimeIndex for the forecast
last_date = df_ccar_usa.index.max()
forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=10, freq='YS')

# Assign the new DatetimeIndex to the confidence interval DataFrame
ic.index = forecast_index

print("DataFrame 'ic' with updated DatetimeIndex:")
print(ic)
### Gráfica forecast SARIMA con IC
forecast_sarima = SARIMA_pred.predicted_mean
forecast_sarima.index = ic.index  

plt.figure(figsize=(10,6))

# serie histórica
plt.plot(df_a_usa.index, df_a_usa['Value'],
         label='Observado', marker='o')

plt.plot(forecast_sarima.index, forecast_sarima,
         label='Forecast SARIMA', linestyle='--', color='C1', marker='o')

# banda de confianza (IC) usando el DataFrame ic
plt.fill_between(ic.index,
                 ic.iloc[:,0],      # lower Value
                 ic.iloc[:,1],      # upper Value
                 color='C1', alpha=0.25,
                 label='IC 95%')

plt.title('SARIMA – Observado y pronóstico con IC')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
### ____________________________________
### Importaciones específicas
us_imports = pd.read_excel(BASE_DIR / 'exportaciones_farmaceuticas_identificadas.xlsx')
us_imports.head()

us_imports['HS4 4 Digit'].unique()

hs4_unique_table = pd.DataFrame({
    "HS4 4 Digit": sorted(us_imports["HS4 4 Digit"].unique())
})
hs4_unique_table["ID"] = range(1, len(hs4_unique_table) + 1)
hs4_unique_table = hs4_unique_table[["ID", "HS4 4 Digit"]]

hs4_unique_table.tail()

plantas_file = BASE_DIR / 'empresas_con_plantas_en_mexico_y_exportadores_MX_USA.xlsx'
plantas_80 = pd.read_excel(plantas_file, 'Producto_80', header = 3)
plantas_80.head()

plantas_20 = pd.read_excel(plantas_file, 'Exportadores_20', header = 3)
plantas_20.head()

integrado_u = pd.read_excel(plantas_file, 'Integrado_unico', header = 3)
integrado_u.head()

em_int = pd.read_excel(BASE_DIR / 'exportaciones_equipo_medico_empresas_sectores_urls.xlsx')
em_int.tail()

market_share_empresas_raw = [
    11.0,  # Medtronic
    9.0,   # GE HealthCare
    8.5,   # Siemens Healthineers
    7.0,   # Philips Healthcare
    8.0,   # Abbott
    6.5,   # Johnson & Johnson MedTech
    5.5,   # Stryker
    5.5,   # Becton Dickinson
    4.5,   # Boston Scientific
    3.5,   # 3M Health Care
    3.5,   # Cardinal Health
    4.5,   # Thermo Fisher Scientific
    3.5,   # Fresenius Medical Care
    3.5,   # Zimmer Biomet
    3.5,   # Baxter International
    2.5,   # Hologic
    2.5,   # Intuitive Surgical
    2.5,   # Danaher
    1.5,   # ResMed
    2.5    # Olympus Medical
]

market_share_total = sum(market_share_empresas_raw)
market_share_empresas = [value * (100.0 / market_share_total) for value in market_share_empresas_raw]

df_hubs = pd.read_excel(DATA_ROOT / 'hubs_y_organizaciones_adicionales_farmaceuticas_mexico_urls.xlsx')
df_hubs

df_el_mexico = pd.DataFrame([
    {"empresa":"World Courier","hub_mexico":"Tlalnepantla (Estado de México)","latitud":19.5407,"longitud":-99.1950},
    {"empresa":"Biocair","hub_mexico":"Ciudad de México","latitud":19.4326,"longitud":-99.1332},
    {"empresa":"Marken","hub_mexico":"Ciudad de México","latitud":19.4326,"longitud":-99.1332},
    {"empresa":"Cryoport","hub_mexico":"Guadalajara","latitud":20.6597,"longitud":-103.3496},
    {"empresa":"Movianto","hub_mexico":"Ciudad de México","latitud":19.4326,"longitud":-99.1332},
    {"empresa":"Bomi Group","hub_mexico":"Querétaro","latitud":20.5888,"longitud":-100.3899},
    {"empresa":"FIEGE Pharma Logistics","hub_mexico":"Monterrey","latitud":25.6866,"longitud":-100.3161},
    {"empresa":"JAS Worldwide","hub_mexico":"Guadalajara","latitud":20.6597,"longitud":-103.3496},
    {"empresa":"GEODIS Pharma Healthcare","hub_mexico":"Guadalajara","latitud":20.6597,"longitud":-103.3496},
    {"empresa":"Almac Group","hub_mexico":"Ciudad de México","latitud":19.4326,"longitud":-99.1332}
])

print(df_el_mexico)

delitos_dir = DATA_ROOT / 'Municipal-Delitos-2015-2025_ene2026'
files = [
    delitos_dir / '2015.xlsx', delitos_dir / '2016.xlsx', delitos_dir / '2017.xlsx',
    delitos_dir / '2018.xlsx', delitos_dir / '2019.xlsx', delitos_dir / '2020.xlsx',
    delitos_dir / '2021.xlsx', delitos_dir / '2022.xlsx', delitos_dir / '2023.xlsx',
    delitos_dir / '2024_ene2026.xlsx', delitos_dir / '2025_ene2026.xlsx'
]

dfs = []

for f in files:
    df = pd.read_excel(f)
    df["source_file"] = f
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

combined



tipos_objetivo = [
    "Homicidio",
    "Narcomenudeo",
    "Trata de personas"
]

# 2) Subtipos de delito directos (sin depender de modalidad)
subtipos_directos = [
    "Robo a negocio",
    "Robo de maquinaria",
    "Robo a transportista",
    "Extorsión",
]

# 3) Normalización para evitar problemas de mayúsculas/espacios
tipo_s = combined["Tipo de delito"].astype(str).str.strip().str.casefold()
subtipo_s = combined["Subtipo de delito"].astype(str).str.strip().str.casefold()
modalidad_s = combined["Modalidad"].astype(str).str.strip().str.casefold()

tipos_objetivo_s = [t.casefold() for t in tipos_objetivo]
subtipos_directos_s = [s.casefold() for s in subtipos_directos]

# 4) Máscaras por reglas solicitadas
mask_tipos = tipo_s.isin(tipos_objetivo_s)

# Robo a negocio (con y sin violencia)
mask_robo_negocio = (
    (subtipo_s == "robo a negocio".casefold()) &
    modalidad_s.isin(["con violencia".casefold(), "sin violencia".casefold()])
)

# Extorsión, Fraude, Daño a la propiedad
mask_subtipos_directos = subtipo_s.isin(subtipos_directos_s)

# Robo de maquinaria -> Modalidad: Robo de herramienta industrial o agrícola (con/sin violencia)
mask_robo_maquinaria = (
    (subtipo_s == "robo de maquinaria".casefold()) &
    modalidad_s.isin([
        "robo de herramienta industrial o agrícola con violencia".casefold(),
        "robo de herramienta industrial o agrícola sin violencia".casefold()
    ])
)

# Robo a transportista (con/sin violencia)
mask_robo_transportista = (
    (subtipo_s == "robo a transportista".casefold()) &
    modalidad_s.isin(["con violencia".casefold(), "sin violencia".casefold()])
)

# 5) Filtro final
mask_final = (
    mask_tipos |
    mask_robo_negocio |
    mask_subtipos_directos |
    mask_robo_maquinaria |
    mask_robo_transportista
)

df_delitos_objetivo = combined.loc[mask_final].copy()

# (Opcional) ver tamaño y primeras filas
print(df_delitos_objetivo.shape)
df_delitos_objetivo.head()

df_delitos_objetivo['Subtipo de delito'].unique()

# Suma anual (enero-diciembre) por registro
meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

df_plot_estado = df_delitos_objetivo.copy()
df_plot_estado[meses] = df_plot_estado[meses].apply(pd.to_numeric, errors="coerce").fillna(0)
df_plot_estado["Total_delitos"] = df_plot_estado[meses].sum(axis=1)

# Agregado por estado
pie_estado = (
    df_plot_estado
    .groupby("Entidad", as_index=False)["Total_delitos"]
    .sum()
    .sort_values("Total_delitos", ascending=False)
)

# Pie chart
fig = px.pie(
    pie_estado,
    names="Entidad",
    values="Total_delitos",
    title="Participación de delitos objetivo por estado"
)
fig.update_traces(textinfo="percent+label")
fig.show()


ultimo_anio = int(df_delitos_objetivo["Año"].max())
anio_inicio = ultimo_anio - 4

df_5y = df_delitos_objetivo[
    (df_delitos_objetivo["Año"] >= anio_inicio) &
    (df_delitos_objetivo["Año"] <= ultimo_anio)
].copy()

meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

df_5y[meses] = df_5y[meses].apply(pd.to_numeric, errors="coerce").fillna(0)
df_5y["Total_delitos"] = df_5y[meses].sum(axis=1)

pie_estado_5y = (
    df_5y
    .groupby("Entidad", as_index=False)["Total_delitos"]
    .sum()
    .sort_values("Total_delitos", ascending=False)
)

fig = px.pie(
    pie_estado_5y,
    names="Entidad",
    values="Total_delitos",
    title=f"Participación por estado (últimos 5 años: {anio_inicio}-{ultimo_anio})"
)
fig.update_traces(textinfo="percent+label")
fig.show()



# Base
df_tipos = df_delitos_objetivo.copy()

# Suma anual por registro
meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]
df_tipos[meses] = df_tipos[meses].apply(pd.to_numeric, errors="coerce").fillna(0)
df_tipos["Total_delitos"] = df_tipos[meses].sum(axis=1)

# Agregado nacional por tipo de delito
participacion_tipo = (
    df_tipos
    .groupby("Tipo de delito", as_index=False)["Total_delitos"]
    .sum()
    .sort_values("Total_delitos", ascending=False)
)

# Participación (%) opcional en tabla
participacion_tipo["Participacion_%"] = (
    participacion_tipo["Total_delitos"] / participacion_tipo["Total_delitos"].sum() * 100
)

print(participacion_tipo)

# Gráfica de participación por tipo
fig = px.pie(
    participacion_tipo,
    names="Tipo de delito",
    values="Total_delitos",
    title="Participación de cada tipo de delito en México"
)
fig.update_traces(textinfo="percent+label")
fig.show()



df_estado = df_delitos_objetivo.copy()

meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

# Total anual por registro
df_estado[meses] = df_estado[meses].apply(pd.to_numeric, errors="coerce").fillna(0)
df_estado["Total_anual_registro"] = df_estado[meses].sum(axis=1)

# Total por estado y año
estado_anio = (
    df_estado
    .groupby(["Entidad", "Año"], as_index=False)["Total_anual_registro"]
    .sum()
)

# Promedio anual por estado (promedio entre años)
pie_estado_promedio = (
    estado_anio
    .groupby("Entidad", as_index=False)["Total_anual_registro"]
    .mean()
    .rename(columns={"Total_anual_registro": "Promedio_anual_delitos"})
    .sort_values("Promedio_anual_delitos", ascending=False)
)

fig = px.pie(
    pie_estado_promedio,
    names="Entidad",
    values="Promedio_anual_delitos",
    title="Participación por estado (promedio anual)"
)
fig.update_traces(textinfo="percent+label")
fig.show()

# Quitar filas de la entidad "México" y "Ciudad de México"
df_sin_mx_cdmx = df_delitos_objetivo[
    ~df_delitos_objetivo["Entidad"].astype(str).str.strip().isin(["México", "Ciudad de México"])
].copy()

# Verificación rápida
print(df_sin_mx_cdmx["Entidad"].unique())

# Usa el nuevo dataframe sin México y CDMX
df_estado = df_sin_mx_cdmx.copy()

meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

# Total anual por registro
df_estado[meses] = df_estado[meses].apply(pd.to_numeric, errors="coerce").fillna(0)
df_estado["Total_anual_registro"] = df_estado[meses].sum(axis=1)

# Total por estado y año
estado_anio = (
    df_estado
    .groupby(["Entidad", "Año"], as_index=False)["Total_anual_registro"]
    .sum()
)

# Promedio anual por estado
pie_estado_promedio = (
    estado_anio
    .groupby("Entidad", as_index=False)["Total_anual_registro"]
    .mean()
    .rename(columns={"Total_anual_registro": "Promedio_anual_delitos"})
    .sort_values("Promedio_anual_delitos", ascending=False)
)

fig = px.pie(
    pie_estado_promedio,
    names="Entidad",
    values="Promedio_anual_delitos",
    title="Participación por estado (promedio anual, sin México y CDMX)"
)
fig.update_traces(textinfo="percent+label")
fig.show()

df_estado['Entidad'].unique()


# ============================================
# Punto estratégico con Decision Tree Regressor
# ============================================



def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_distance_km(lat, lon, points_df):
    if points_df.empty:
        return np.inf
    dists = haversine_km(lat, lon, points_df["lat"].values, points_df["lon"].values)
    return float(np.nanmin(dists))


# 1) Fuentes geográficas
plants_80_geo = plantas_80[["empresa", "estado", "latitud_aprox", "longitud_aprox"]].rename(
    columns={"latitud_aprox": "lat", "longitud_aprox": "lon"}
).dropna(subset=["lat", "lon"])
plants_20_geo = plantas_20[["empresa", "estado", "latitud_aprox", "longitud_aprox"]].rename(
    columns={"latitud_aprox": "lat", "longitud_aprox": "lon"}
).dropna(subset=["lat", "lon"])
plants_geo = pd.concat([plants_80_geo, plants_20_geo], ignore_index=True).drop_duplicates()

hubs_geo = df_hubs[["nombre", "estado", "latitud", "longitud"]].rename(
    columns={"nombre": "empresa", "latitud": "lat", "longitud": "lon"}
).dropna(subset=["lat", "lon"])

dist_geo = df_el_mexico[["empresa", "hub_mexico", "latitud", "longitud"]].rename(
    columns={"hub_mexico": "estado", "latitud": "lat", "longitud": "lon"}
).dropna(subset=["lat", "lon"])

# 2) Riesgo por estado (promedio anual de delitos)
risk_df = pie_estado_promedio.copy()
risk_df["estado"] = risk_df["Entidad"].astype(str).str.strip()
risk_df["risk_base"] = pd.to_numeric(risk_df["Promedio_anual_delitos"], errors="coerce").fillna(0.0)

# Multiplicadores de riesgo solicitados
risk_multiplier = {
    "Guanajuato": 2.5,
    "Baja California": 2.5,
    "Michoacán de Ocampo": 3.4,
    "Jalisco": 1.9,
    "Chihuahua": 2.5,
    "Sonora": 3.4,
    "Guerrero": 2.5,
    "Zacatecas": 2.5,
    "Colima": 2.5,
    "Sinaloa": 3.0
}
risk_df["multiplier"] = risk_df["estado"].map(risk_multiplier).fillna(1.0)
risk_df["risk_weighted"] = risk_df["risk_base"] * risk_df["multiplier"]

# Coordenadas por estado para capa de riesgo (basadas en plantas/hubs/distribuidores + Sinaloa fijo)
state_points = pd.concat([
    plants_geo[["estado", "lat", "lon"]],
    hubs_geo[["estado", "lat", "lon"]],
], ignore_index=True)
state_centroids = (
    state_points.dropna(subset=["estado"])
    .groupby("estado", as_index=False)[["lat", "lon"]]
    .mean()
)

# Coordenada fija para Sinaloa
state_centroids = pd.concat([
    state_centroids[state_centroids["estado"] != "Sinaloa"],
    pd.DataFrame([{"estado": "Sinaloa", "lat": 25.1721, "lon": -107.4795}])
], ignore_index=True)

risk_points = state_centroids.merge(
    risk_df[["estado", "risk_weighted"]],
    on="estado",
    how="inner"
).dropna(subset=["risk_weighted"])

# 3) Malla de candidatos (México aprox)
lat_grid = np.linspace(14.5, 32.5, 70)
lon_grid = np.linspace(-117.0, -86.0, 90)
candidates = pd.DataFrame(
    [(la, lo) for la in lat_grid for lo in lon_grid],
    columns=["lat", "lon"]
)

# 4) Feature engineering y función objetivo
features = []
targets = []
for _, row in candidates.iterrows():
    lat, lon = row["lat"], row["lon"]

    d_plants = nearest_distance_km(lat, lon, plants_geo)
    d_hubs = nearest_distance_km(lat, lon, hubs_geo)
    d_dist = nearest_distance_km(lat, lon, dist_geo)

    if risk_points.empty:
        risk_penalty = 0.0
    else:
        d_risk = haversine_km(lat, lon, risk_points["lat"].values, risk_points["lon"].values)
        # Peso inverso a distancia para penalizar cercanía a estados de riesgo
        risk_penalty = float(np.sum(risk_points["risk_weighted"].values / (d_risk + 25.0)))

    prox_plants = np.exp(-d_plants / 220.0)
    prox_hubs = np.exp(-d_hubs / 250.0)
    prox_dist = np.exp(-d_dist / 250.0)

    score = (0.45 * prox_plants) + (0.25 * prox_hubs) + (0.20 * prox_dist) - (0.75 * risk_penalty)

    features.append([lat, lon, d_plants, d_hubs, d_dist, risk_penalty])
    targets.append(score)

X = pd.DataFrame(features, columns=["lat", "lon", "d_plants", "d_hubs", "d_dist", "risk_penalty"])
y = np.array(targets)

# 5) Modelo Decision Tree
tree = DecisionTreeRegressor(
    max_depth=8,
    min_samples_leaf=25,
    random_state=42
)
tree.fit(X, y)
candidates["pred_score"] = tree.predict(X)

best_idx = candidates["pred_score"].idxmax()
best_point = candidates.loc[best_idx, ["lat", "lon", "pred_score"]]

print("PUNTO ESTRATEGICO RECOMENDADO")
print(best_point)

# 6) Visualización rápida
fig_strategy = go.Figure()
fig_strategy.add_trace(go.Scattergeo(
    lat=plants_geo["lat"],
    lon=plants_geo["lon"],
    mode="markers",
    marker=dict(size=5, color="#4b5563", opacity=0.55),
    name="Plantas"
))
fig_strategy.add_trace(go.Scattergeo(
    lat=hubs_geo["lat"],
    lon=hubs_geo["lon"],
    mode="markers",
    marker=dict(size=8, color="#2563eb", opacity=0.8),
    name="Hubs"
))
fig_strategy.add_trace(go.Scattergeo(
    lat=dist_geo["lat"],
    lon=dist_geo["lon"],
    mode="markers",
    marker=dict(size=8, color="#f59e0b", opacity=0.8),
    name="Distribuidores"
))
fig_strategy.add_trace(go.Scattergeo(
    lat=[best_point["lat"]],
    lon=[best_point["lon"]],
    mode="markers",
    marker=dict(size=14, color="#16a34a", symbol="star"),
    name="Punto estratégico"
))
fig_strategy.update_geos(
    scope="north america",
    lataxis_range=[14, 33],
    lonaxis_range=[-119, -86],
    showcountries=True,
    showcoastlines=True,
    showsubunits=True
)
fig_strategy.update_layout(
    title="Decision Tree - Punto estratégico sugerido",
    height=650
)
fig_strategy.show()
