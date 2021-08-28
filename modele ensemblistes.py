# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:52:29 2019

@author: iamadou
"""

pip install xgboost
import re # Regular expressions package
from sklearn import linear_model, ensemble, cluster, neural_network # Will be with dir()
from sklearn.linear_model import * # Making models avalaible to access _get_param_names() function
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.neural_network import *
#import xgboost
#from xgboost import XGBClassifier, XGBRegressor, XGBRanker #Importing packages not included in sklearn
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams 
import xgboost as xgb