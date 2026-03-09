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
import plotly.express as px

st.set_page_config(
    page_title="CLCircular - Dashboard",
    layout="wide"
)

st.title("CLCircular Dashboard")
st.markdown("Explorando oportunidades de mercado")

