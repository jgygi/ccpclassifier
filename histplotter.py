import numpy as np
import trackloader as tl
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import math

def plotHistograms(df, df_norm, plotA, plotB):
    Tot = len(df.columns) - 1
    Cols = 4 #NOTE: using 4 columns, change for more
    Rows = Tot // Cols 
    Rows += Tot % Cols
    Position = range(1,Tot + 1)

    df = df.reset_index()
    df = df.drop(labels = 'trackNum', axis = 1)
    print(df)

    df_norm = df_norm.reset_index()
    df_norm = df_norm.drop(labels = 'trackNum', axis = 1)
    print(df_norm)



    fig = plt.figure(1, figsize=(60, 35))
    df_comb = pd.DataFrame(columns = ['before', 'after'])
    #for k in range(Tot):
    df_comb['before'] = df[df.columns[0]]
    df_comb['after'] = df_norm[df.columns[0]]

    print(df_comb)

    ax = fig.add_subplot(Rows,Cols,Position[0])
    plt.title(df.columns[0])
    if plotA:
        ax.hist(df_comb['before'], edgecolor='black', alpha=0.5, color='blue')
    if plotB:
        ax.hist(df_comb['after'], edgecolor='black', alpha=0.5, color='red')
    fig.subplots_adjust(hspace=.5)
    plt.show()

    