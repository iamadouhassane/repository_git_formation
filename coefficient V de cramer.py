# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:38:24 2019

@author: iamadou
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from dython.model_utils import roc_graph
from dython.nominal import associations
from matplotlib import cm as cm
from pyitlib import discrete_random_variable as drv
from collections import Counter
from pycm import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dython.nominal import conditional_entropy

data = pd.read_csv("C:/Users/iamadou/Desktop/arpu_v2/data_montant_1_v2_2.csv")
data = data.drop(['TRAFFIC_SOURCE_ADWORDS_CLICK_INFO_CRITERIA_PARAMETERS'], axis=1)
data_1 = data.drop(['ODCHA_LABEL','LB_BLCHA'], axis=1).copy()
data = data.drop(['montant', 'ODCHA_LABEL','LB_BLCHA'], axis=1)
for i in data.columns:
    data[i] = data[i].astype('object')

df = data.copy()
cat_list = [] 

for col in df.columns:
  if df[col].dtype == object:
    cat_list.append(col)

df_corr = associations(df,
                       nominal_columns = cat_list, 
                       theil_u=True,
                       plot=False, 
                       return_results = True)


#################################################################################################################
############################################ calcul du coeffcient V de Cramer ###################################
#################################################################################################################
## Ce coefficient mesure l'association entre 2 caracteristqiues categorielles donc une sorte de coeffcient de corrélation
## à la seule différence qu'il est toujours positif compris [0, 1] et qu'il ne permette pas de determiner la nature de cette association
## est -elle positive ou negative ?
## 0 indique aucune association entre les modalités des variables et 1 une forte association 
##Ce coefficient est aussi symetrique 


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


correlation_cramer = pd.DataFrame(0, index=list(data.columns), columns=list(data.columns))

for i in range(len(data.columns)):
    for j in range(len(data.columns)) :
        correlation_cramer.iloc[i, j] = cramers_v(df.iloc[:,i], df.iloc[:,j])
        correlation_cramer.iloc[j, i] = correlation_cramer.iloc[i, j]
        correlation_cramer.iloc[i, i] = 1

correlation_cramer.fillna(0, inplace=True)



fig = plt.figure(1, figsize=(len(correlation_cramer), len(correlation_cramer)))
ax = plt.gca()
ax.set_xticks(np.arange(len(correlation_cramer)))
ax.set_yticks(np.arange(len(correlation_cramer)))
ax.set_xticklabels(correlation_cramer)
ax.set_yticklabels(correlation_cramer)
im = ax.imshow(correlation_cramer)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
for i in range(len(correlation_cramer)):
    for j in range(len(correlation_cramer)):
        text = ax.text(j, i, np.round(correlation_cramer.iloc[i, j],2),ha="center", va="center", color="w")
plt.colorbar(im, cax=cax)
ax.set_title("Cramer correlation matrix")
plt.show()

ax.get_figure().savefig('C:/Users/iamadou/Desktop/Cramer_correlation.png')


####################################################################################################################
#Conclusion1 :Il suffit de regarder cette carte thermique pour constater que l’ AD_NETWORK est étroitement associée à la classe 
#(DEVICE_LABEL / AD_TYPE / SERVICE_NAME) de l'ARPU et que AD_TYPE est étroitement associée à trois autres : NETCH_LABEL,DEVICE_LABEL,SERVICE_NAME, PROMO

# conclusion2 : La malédiction de la symétrie : Soient X et Y 2 variables catégorielles avec des nombres de modalités Nx et Ny .
# On remarque que si la valeur de X est connue , la valeur de Y ne peut toujours pas être déterminée, mais si la valeur de Y est connue, la valeur de X est garantie
# Cette information précieuse est perdue lors de l'utilisation de Cramer V en raison de sa symétrie. 
# Par conséquent, pour la préserver, nous avons besoin d'une mesure asymétrique de l'association entre les caractéristiques catégorielles.
# Pour se faire, on utilise U de theil , aussi appelé le coefficient d'incertitude


################################################################################################################
########################## calcul du coefficient d'incertitude ou Theil's U  ###################################
################################################################################################################
## Ce coefficient est basée sur l' entropie conditionnelle
## Tout comme le V de Cramer, la valeur de sortie est sur la plage de [0,1], avec les mêmes interprétations que précédemment
## mais contrairement à V de Cramer, il est asymétrique ie  U (x, y) ≠ U (y, x) ( tandis que V (x, y) = V (y, x))


def theils_u(x, y):
    x_counter = Counter(x)
    x = list(x) ## actual_vector 
    y = list(y) ## predict_vector
    cm = ConfusionMatrix(x, y,digit=5)
    total_occurrences = sum(list(x_counter.values()))
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    sx_x_log = sum(-log(p_x)*p_x)
    s_x = ss.entropy(p_x)
    I_xy = cm.overall_stat['Mutual Information']
    if s_x == 0:
        return 1
    else:
        return (I_xy / s_x)


def theils_u_1(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(list(x_counter.values()))
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


correlation_theil = pd.DataFrame(0, index=list(data.columns), columns=list(data.columns))

correlation_theil_1 = pd.DataFrame(0, index=list(data.columns), columns=list(data.columns))

for i in range(len(data.columns)):
    for j in range(len(data.columns)) :
        correlation_theil_1.iloc[i, j] = theils_u_1(df.iloc[:,i], df.iloc[:,j])
        correlation_theil_1.iloc[j, i] = theils_u_1(df.iloc[:,j], df.iloc[:,i])
        correlation_theil_1.iloc[i, i] = 1
        
theils_u_1(df.AD_NETWORK, df.AD_TYPE)
theils_u(df.AD_NETWORK, df.AD_TYPE)

for i in range(len(data.columns)):
    for j in range(len(data.columns)) :
        correlation_theil.iloc[i, j] = theils_u(df.iloc[:,i], df.iloc[:,j])
        correlation_theil.iloc[j, i] = theils_u(df.iloc[:,j], df.iloc[:,i])
        correlation_theil.iloc[i, i] = 1

correlation_theil[correlation_theil>1] = .9

theils_u(df.AD_NETWORK, df.AD_TYPE)
theils_u(df.AD_TYPE, df.AD_NETWORK)


fig = plt.figure(1, figsize=(len(correlation_theil), len(correlation_theil)))
ax = plt.gca()
ax.set_xticks(np.arange(len(correlation_theil)))
ax.set_yticks(np.arange(len(correlation_theil)))
ax.set_xticklabels(correlation_theil)
ax.set_yticklabels(correlation_theil)
im = ax.imshow(correlation_theil)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
for i in range(len(correlation_theil)):
    for j in range(len(correlation_theil)):
        text = ax.text(j, i, np.round(correlation_theil.iloc[i, j],2),ha="center", va="center", color="w")
plt.colorbar(im, cax=cax)
ax.set_title("Theil correlation matrix")
plt.show()
ax.get_figure().savefig('C:/Users/iamadou/Desktop/Theil_correlation.png')


fig = plt.figure(1, figsize=(len(correlation_theil_1), len(correlation_theil_1)))
ax = plt.gca()
ax.set_xticks(np.arange(len(correlation_theil_1)))
ax.set_yticks(np.arange(len(correlation_theil_1)))
ax.set_xticklabels(correlation_theil_1)
ax.set_yticklabels(correlation_theil_1)
im = ax.imshow(correlation_theil_1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
for i in range(len(correlation_theil_1)):
    for j in range(len(correlation_theil_1)):
        text = ax.text(j, i, np.round(correlation_theil_1.iloc[i, j],2),ha="center", va="center", color="w")
plt.colorbar(im, cax=cax)
ax.set_title("Theil correlation matrix")
plt.show()

################################################################################################################
###################################### rapport de correlation  #################################################
################################################################################################################
# mesure statistique entre une variable continue et une variable categorielle 
#il est défini comme la variance pondérée de la moyenne de chaque catégorie divisée par la variance de tous les échantillons
#Il permet de repondre a la question :
#À partir d'un nombre continu, dans quelle mesure pouvez-vous savoir à quelle catégorie il appartient



def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return (eta)


ratio_correlation = pd.DataFrame(0, index=list(data.columns), columns=list(data.columns))
ratio_correlation = ratio_correlation.iloc[1,:]

for i in data.columns:
    ratio_correlation.iloc[ratio_correlation.index==i] = correlation_ratio(data_1[i], data_1.montant)

ratio_correlation = pd.DataFrame(ratio_correlation.reset_index().rename(columns={'NETCH_LABEL' : 'coefficient', 'index':'variable'}))
ratio_correlation.set_index(['variable'])




import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax1.bar(list(ratio_correlation.variable.unique()),list(ratio_correlation.coefficient.unique()))
ax1.set_title("Analyse univariée : variables   Vs   montant")
plt.setp(ax1.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor", visible=True)
for j in range(len(correlation_theil)):
    #print(np.round(ratio_correlation.iloc[j, 1],2))
    text = ax1.text(1,j, np.round(ratio_correlation.iloc[j, 1],2))

plt.show()






