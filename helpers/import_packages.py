"""
This file imports all required packages
"""
import os
from contextlib import contextmanager
import platform
import errno
import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.colors  as colors
import scipy.stats        as st
import matplotlib.cm      as cmx
import pickle             as pk
import matplotlib.patches as patch
import patsy              as pat
import copy
import time
import random 
import yaml
import tscv
from monthdelta import monthdelta


# machine learning (from scikit-learn)

import lightgbm
import sklearn.base             as skl_base
import sklearn.ensemble         as skl_ens
import sklearn.neural_network   as skl_nn
import sklearn.tree             as skl_tree
import sklearn.linear_model     as skl_lin
import sklearn.neighbors        as skl_neigh
import sklearn.svm              as skl_svm
import sklearn.naive_bayes      as skl_NB
import sklearn.metrics          as skl_metrics
import sklearn.model_selection  as skl_slct
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance, partial_dependence
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# some extras
from sklearn.preprocessing import StandardScaler
import hashlib
import multiprocessing
import statsmodels.api as sm
import statsmodels.tsa.api         as sm_ts
import statsmodels.tsa.arima_model as arima
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.formula.api     as smf
from statsmodels.regression.linear_model import OLS
import numpy.polynomial.polynomial as poly
import statsmodels.tsa.stattools   as stat_tool
from statsmodels.tsa.stattools    import adfuller, coint
import itertools
from itertools import chain, combinations
import collections
from itertools import combinations, chain
from math import sqrt
# from math import comb

import re
import time
import os
import shap
import warnings
import math
import csv
from collections.abc import Iterable 
import pathos.pools as pp
