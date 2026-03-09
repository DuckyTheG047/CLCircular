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
import streamlit as st

### Carga Data Market Size Categorías
df_mst = pd.read_excel('/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/Code/db1_internacional_pharma.xlsx', 'Mkt.Size', header = 5)
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
us_imports = pd.read_excel('/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/Exports---November-2025-Click-on-the-Visualization-to-Select.xlsx')
us_imports.head()
print(us_imports.columns)

us_imports["HS4 4 Digit"] = us_imports["HS4 4 Digit"].astype(str).str.lower()

# Palabras clave relacionadas con la industria farmacéutica
keywords = [
    "pharmaceutical",
    "medicament",
    "medicine",
    "drug",
    "vaccine",
    "antibiotic",
    "vitamin",
    "hormone",
    "enzyme",
    "alkaloid",
    "culture",
    "diagnostic",
    "laboratory",
    "reagent",
    "blood",
    "organ",
    "therapeutic",
    "medical",
    "surgical",
    "dental",
    "hospital",
    "bandage",
    "gauze",
    "x-ray",
    "respiration",
    "therapy"
]

def is_pharma(text):
    return any(keyword in text for keyword in keywords)

# Aplicar filtro
pharma_df = us_imports[us_imports["HS4 4 Digit"].apply(is_pharma)]

pharma_df[['Year', 'Chapter 4 Digit', 'HS2 4 Digit', 'HS4 4 Digit', 'Trade Value', 'Share']]





st.set_page_config(
    page_title="Dashboard Pharma Market Size",
    layout="wide"
)

st.title("Dashboard interactivo - Pharma Market Size")
st.markdown("Filtra por país y categoría para explorar la evolución del valor de mercado.")

# ---------------------------
# Carga de datos
# ---------------------------
file = '/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/db1_internacional_pharma.xlsx'
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Mkt.Size", header=5)
    return df

uploaded_file = st.file_uploader(
    "Sube el archivo Excel de market size",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Sube el archivo `db1_internacional_pharma.xlsx` para visualizar el dashboard.")
    st.stop()

df_mst = load_data(uploaded_file)

# ---------------------------
# Limpieza y transformación
# ---------------------------
df_mst = df_mst.dropna().copy()

years = [str(year) for year in range(2011, 2026)]

df_long = df_mst.melt(
    id_vars=["Geography", "Category", "Data Type", "Unit", "Current Constant"],
    value_vars=years,
    var_name="Year",
    value_name="Value"
)

df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

# Conversión MXN million -> USD million
df_long["Value"] = np.where(
    df_long["Unit"].eq("MXN million"),
    df_long["Value"] / 17.7,
    df_long["Value"]
)

df_long["Unit"] = "Million USD"

df_long = df_long.dropna(subset=["Year", "Value", "Geography", "Category"]).copy()
df_long["Year"] = df_long["Year"].astype(int)

# ---------------------------
# Sidebar filtros
# ---------------------------
st.sidebar.header("Filtros")

all_geographies = sorted(df_long["Geography"].unique().tolist())
all_categories = sorted(df_long["Category"].unique().tolist())

selected_geographies = st.sidebar.multiselect(
    "Selecciona país / geografía",
    options=all_geographies,
    default=["Mexico", "USA"] if all(x in all_geographies for x in ["Mexico", "USA"]) else all_geographies[:2]
)

selected_categories = st.sidebar.multiselect(
    "Selecciona categoría",
    options=all_categories,
    default=[
        c for c in [
            "OTC",
            "Sports Nutrition",
            "Vitamins and Dietary Supplements",
            "Weight Management and Wellbeing",
            "Herbal/Traditional Products",
            "Allergy Care"
        ] if c in all_categories
    ] or all_categories[:6]
)

year_range = st.sidebar.slider(
    "Rango de años",
    min_value=int(df_long["Year"].min()),
    max_value=int(df_long["Year"].max()),
    value=(int(df_long["Year"].min()), int(df_long["Year"].max()))
)

# ---------------------------
# Filtrado
# ---------------------------
df_filtered = df_long[
    (df_long["Geography"].isin(selected_geographies)) &
    (df_long["Category"].isin(selected_categories)) &
    (df_long["Year"].between(year_range[0], year_range[1]))
].copy()

if df_filtered.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada.")
    st.stop()

# ---------------------------
# KPIs
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros", f"{len(df_filtered):,}")
col2.metric("Categorías", df_filtered["Category"].nunique())
col3.metric("Geografías", df_filtered["Geography"].nunique())
col4.metric("Valor total", f"{df_filtered['Value'].sum():,.2f} {df_filtered['Unit'].iloc[0]}")

# ---------------------------
# Gráfica principal
# Basada en la lógica de Year vs Value con hue=Category
# ---------------------------
st.subheader("Evolución del valor por categoría")

plot_mode = st.radio(
    "Modo de visualización",
    ["Colorear por categoría", "Colorear por país"],
    horizontal=True
)

fig, ax = plt.subplots(figsize=(14, 7))

if plot_mode == "Colorear por categoría":
    sns.lineplot(
        data=df_filtered,
        x="Year",
        y="Value",
        hue="Category",
        style="Geography",
        marker="o",
        ax=ax
    )
else:
    sns.lineplot(
        data=df_filtered,
        x="Year",
        y="Value",
        hue="Geography",
        style="Category",
        marker="o",
        ax=ax
    )

ax.set_title("Evolución del Valor por Categoría")
ax.set_xlabel("Año")
ax.set_ylabel("Valor (Million USD)")
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df_filtered["Year"].unique()))
plt.xticks(rotation=45)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

st.pyplot(fig)

# ---------------------------
# Tabla resumen
# ---------------------------
st.subheader("Tabla resumen")

summary = (
    df_filtered.groupby(["Geography", "Category", "Year"], as_index=False)["Value"]
    .sum()
    .sort_values(["Geography", "Category", "Year"])
)

st.dataframe(summary, use_container_width=True)

# ---------------------------
# Tabla pivote
# ---------------------------
st.subheader("Tabla pivote")

pivot_table = summary.pivot_table(
    index=["Geography", "Category"],
    columns="Year",
    values="Value",
    aggfunc="sum"
)

st.dataframe(pivot_table, use_container_width=True)

# ---------------------------
# Descarga
# ---------------------------
csv_data = summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar datos filtrados (CSV)",
    data=csv_data,
    file_name="market_size_filtrado.csv",
    mime="text/csv"
)