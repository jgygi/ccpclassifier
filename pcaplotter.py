import scipy.io
import pandas as pd
from itertools import chain
from itertools import groupby
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plotPCA(df):
    #Split into Aux+ (df_aux1) and Aux- (df_aux0)
    df_aux0_reg = df.loc[df['aux'].isin([0])]
    df_aux1_reg = df.loc[df['aux'].isin([1])]

    #Standardize / Normalize:
    df_aux0 = StandardScaler().fit_transform(df_aux0_reg)
    df_aux1 = StandardScaler().fit_transform(df_aux1_reg)

    #PCA
    pca0 = PCA(n_components = len(df_aux0_reg.columns))
    pca0.fit(df_aux0)
    pca1 = PCA(n_components = len(df_aux1_reg.columns))
    pca1.fit(df_aux1)
    plt.grid(True)

    label = ["Lifetime", "Max Intensity", "Background", "Total Displacement", "MSD", "Category", "Aux", "Average Slope Rise", "Average Slope Decay", "Slope Rise / Slope Decay", "Average Moment Rise", "Average Moment Decay", "Moment Rise / Moment Decay"]
    colors = ['#F65314','#7CBB00','#00A1F1','#FFBB00','#7B0099','#221F1F','#F65314','#003399','#003399','#b8a9c9','#5b9aa0','#e06377','#f2e394','#a96e5b']
    plt.axhline(linewidth=2, color='k')
    plt.axvline(linewidth=2, color='k')
    for i in [3,4]:
        plt.scatter(pca0.components_[0][i], pca0.components_[1][i], marker = 'o', s = 1000, color = colors[i], alpha = .5)
        plt.scatter(pca1.components_[0][i], pca1.components_[1][i], marker = 'x', s = 1000, color = colors[i])
        plt.text(pca0.components_[0][i], (pca0.components_[1][i]), label[i], fontsize = 12)
        plt.text(pca1.components_[0][i], (pca1.components_[1][i]), label[i], fontsize = 12)

    #plt.text(0, -.75, "X - Aux+", fontsize = 8)
    #plt.text(0, -.78, "O - Aux-", fontsize = 8)
    #plt.text(0, -.81, "Yellow - A, c, tdisp, msd, category, aux", fontsize = 8)
    #plt.text(0, -.84, "Green - Slope-related", fontsize = 8)
    #plt.text(0, -.87, "Magenta - Moment-related", fontsize = 8)

    #plt.axis([-.75,.75,-.5,.75])

    plt.show()

def plotBoxplot(df, colToPlot):
    #Split into Aux+ (df_aux1) and Aux- (df_aux0)
    df_aux0_reg = df.loc[df['aux'].isin([0])]
    df_aux1_reg = df.loc[df['aux'].isin([1])]

    plt.boxplot([df_aux0_reg[colToPlot], df_aux1_reg[colToPlot]], showfliers = False)

    plt.show()