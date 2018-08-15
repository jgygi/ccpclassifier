import numpy as np
import trackloader as tl
import histplotter as hp
import pcaplotter as pp
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import math

# prepareData Function:
# Called by the main function in masterscript.py, prepareData will return a tensorflow readable dataframe with all of the clathrin-related columns
# The program also converts categorical variables such as category into 'one hot' encoding ([0,1,0,0] for 2...)
# Returns: a pandas DataFrame

def prepareData(numCells, folderPath, dropCols = [], showHist = False, catsToUse = [1,2,3,4,5,6,7,8], pvalCutOff = .005, numConsecPVal = 3, mustBeSecondHalf = True, showPrints = True, showBoxplot = True, boxCol = 'max_msd', checkLifetime = True):
    #Change Parameters for Code: --------------------------------------
    
    continuousLabels = ['aux', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom']
    categoricalFeatures = ['frame', 'catIdx', 'trackNum']

    #Initialize Total DataFrame:
    df = pd.DataFrame(columns=['trackNum', 'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom'])
    df = df.set_index('trackNum')

    #Pull Data into DataFrame:
    for i in range(1,int(numCells) + 1):
        df_temp = tl.pullMatrixData(cellNum = i, parentFolderPath = folderPath, numConsecPVal = numConsecPVal, pvalCutOff = pvalCutOff, mustBeSecondHalf = mustBeSecondHalf, checkLifetime = checkLifetime)
        df_temp = df_temp.set_index('trackNum')
        df = pd.concat([df, df_temp], ignore_index = False)

    #Filter DataFrame:
    df = tl.filterMatrixData(data = df, dropNA = True, catsToUse = catsToUse, colsToDrop = dropCols)

    if showBoxplot:
        pp.plotBoxplot(df, boxCol)



    if showPrints:
        print("Describing Data: ____________________________")
        for colname in df.columns:
            print(colname + ":")
            print(df[colname].describe())
            print()
    if showPrints:
        print("Performing Normalization: ___________________")
        print("--- Continuous Features ---")
    df_norm = df.copy()
    print(df_norm)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_norm), columns = df_norm.columns)
    df_norm[:] = df_scaled[:].values

    #NOTE: Change below to change them individually -------
    df_norm['aux'] = df['aux'].values
    df_norm['lifetime'] = df['lifetime'].values
    #df_norm['max_intensity'] = df_scaled['max_intensity'].values
    #df_norm['background'] = df_scaled['background'].values
    #df_norm['totaldisp'] = df_scaled['totaldisp'].values
    #df_norm['max_msd'] = df_scaled['max_msd'].values
    #df_norm['avg_rise'] = df_scaled['avg_rise'].values
    #df_norm['avg_dec'] = df_scaled['avg_dec'].values
    #df_norm['risevsdec'] = df_scaled['risevsdec'].values
    #df_norm['avg_mom_rise'] = df_scaled['avg_mom_rise'].values
    #df_norm['avg_mom_dec'] = df_scaled['avg_mom_dec'].values
    #df_norm['risevsdec_mom'] = df_scaled['risevsdec_mom'].values
    if showPrints:
        print("--- Categorical Features ---")

    #catIdx
    numbins = 8
    zero_data = np.zeros(shape=(df.shape[0], numbins))
    df_zero = pd.DataFrame(zero_data, columns=['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8'], index=df.index)
    j = 0
    for i in df.index.values:
        val = df['catIdx'][i]
        df_zero.iat[j,int(val) - 1] = 1
        j = j + 1

    continuousLabels = [i for i in continuousLabels if i not in dropCols]

    if showHist:
        if showPrints:
            print("Plotting Histograms: ________________________")
        hp.plotHistograms(df[continuousLabels], df_norm[continuousLabels], True, False)
        hp.plotHistograms(df[continuousLabels], df_norm[continuousLabels], False, True)

    #Combine Continuous w/ Categorical:
    output_df = pd.concat([df_norm[continuousLabels], df_zero], axis = 1, ignore_index = False)

    if showPrints:
        print("FINAL DF")
        print(output_df)

    return(output_df)

