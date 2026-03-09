# Librerías
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
from pylab import rcParams
from statsmodels.tsa.arima.model import ARIMA
from numpy._core.fromnumeric import transpose
import itertools
from semopy import Model, Optimizer, calc_stats
import openpyxl
import os

### Carga Data Market Size Categorías
df_mst = pd.read_excel('/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/db1_internacional_pharma.xlsx', 'Mkt.Size', header = 5)
df_mst.head()

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
plt.xticks(df_mexusa['Year'].unique(), rotation=45)
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

### Buscamos entender el comportamiento de los subsegmentos de OTC
df_mexusa_mstj.head()
print(df_mexusa_mstj['Category'].unique())
print(df_mstj['Category'].unique())

selected_categories = ['Adult Mouth Care', 'Analgesics', 'Sleep Aids', 'Cough, Cold and Allergy (Hay Fever) Remedies', 'Dermatologicals',
                       'Digestive Remedies', 'Emergency Contraception', 'Eye Care', 'NRT Smoking Cessation Aids', 'Wound Care']

df_usa = df_prog[df_prog['Geography'] == 'USA']
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

### Entendimiento correlación estatal México (Exportaciones farmacéuticas)
df_estados_farma = pd.read_excel('/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/Evolution-of-the-Quarterly-Market-Concentration-of-Pharmaceutical-Products.xlsx')
df_estados_farma.head()

df_estados_corr = df_estados_farma[['State ID', 'Trade Value', 'Year']]
correlation_matrix = df_estados_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Pearson Correlation Frutas")
plt.show()

plt.figure(figsize=(16, 9))
sns.lineplot(data=df_estados_farma, x='Year', y='Trade Value', hue='State ID', marker='o')
plt.title('Estados Exportación')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.xticks(df_estados_farma['Year'].unique(), rotation=45)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

df_estados_farma['State ID'].unique()

df_estados_complejo = pd.read_excel('/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/Especializacion-y-complejidad-en-Productos-Farmaceuticos.xlsx')
df_estados_complejo.head()
df_estados_farma.head()

db1['value_mean'] = ]