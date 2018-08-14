from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import datapreparation as dp

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def preprocess_features(df):
    """Prepares input features from California housing data set.

    Args:
        df: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = df[
        ['lifetime', 'max_intensity', 'background', 
        'totaldisp', 'max_msd', 
        'avg_rise', 'avg_dec', 'risevsdec', 
        'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom', 
        'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(df):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
        df: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["aux"] = df["aux"]
    return output_targets

def construct_feature_columns(input_features):
    #Construct the TensorFlow Feature Columns.
    #
    #Args:
    #    input_features: The names of the numerical input features to use.
    #Returns:
    #    A set of feature columns
    
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
        learning_rate,
        steps,
        batch_size,
        df_total,
        percentTrain,
        percentTest,
        percentTrainToVal,
        targetCol = 'aux',
        showTrainingPrints = True,
        shuffleTrainingData = True,
        numPeriods = 20):
    #Trains a linear regression model.
    #
    #In addition to training, this function also prints training progress information,
    #as well as a plot of the training and validation loss over time.
    # 
    #Args:
    #    learning_rate: A `float`, the learning rate.
    #    steps: A non-zero `int`, the total number of training steps. A training step
    #    consists of a forward and backward pass using a single batch.
    #    feature_columns: A `set` specifying the input feature columns to use.
    #    training_examples: A `DataFrame` containing one or more columns from
    #    `df` to use as input features for training.
    #    training_targets: A `DataFrame` containing exactly one column from
    #    `df` to use as target for training.
    #    validation_examples: A `DataFrame` containing one or more columns from
    #    `df` to use as input features for validation.
    #    validation_targets: A `DataFrame` containing exactly one column from
    #    `df` to use as target for validation.
    #    
    #Returns:
    #    A `DNNClassifier` object trained on the training data.

    # Re-Index Randomly:
    
    df_sample = df_total.sample(frac=1)

    # Split into Training and Testing:
    numTotal =  df_sample.shape[0]
    numTotalTrain = math.ceil(percentTrain * numTotal)
    numTotalTest = numTotal - numTotalTrain
    df_train = df_sample.head(numTotalTrain)
    df_test = df_sample.tail(numTotalTest)

    # Choose training.
    numTrain = math.ceil(percentTrainToVal * numTotalTrain)
    training_examples = preprocess_features(df_train.head(numTrain))
    training_targets = preprocess_targets(df_train.head(numTrain))

    # Choose validation.
    numValidation = numTotalTrain - numTrain
    validation_examples = preprocess_features(df_train.tail(numValidation))
    validation_targets = preprocess_targets(df_train.tail(numValidation))

    # Choose testing.
    test_examples = preprocess_features(df_test)
    test_targets = preprocess_targets(df_test)

    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print(training_examples)
    print("Validation examples summary:")
    display.display(validation_examples.describe())
    print(validation_examples)
    print("Testing examples summary:")
    display.display(test_examples.describe())
    print(test_examples)

    print("Training targets summary:")
    display.display(training_targets.describe())
    print(training_targets)
    print("Validation targets summary:")
    display.display(validation_targets.describe())
    print(validation_targets)
    print("Testing targets summary:")
    display.display(test_targets.describe())
    print(test_targets)

    #Get Feature Columns
    feature_columns=construct_feature_columns(training_examples)

    periods = numPeriods
    steps_per_period = steps / periods

    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10,10],
        n_classes=2
    )
    
    training_input_fn = lambda: my_input_fn(training_examples, 
                                            training_targets[targetCol], 
                                            batch_size=batch_size)
    eval_input_fn = lambda: my_input_fn(validation_examples,
                                            validation_targets[targetCol], 
                                            num_epochs = 1,
                                            shuffle = False)
    test_input_fn = lambda: my_input_fn(test_examples, 
                                            test_targets[targetCol], 
                                            num_epochs = 1,
                                            shuffle = False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("Accuracy: (on Validation data):")
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Evaluate Validation Set:
        eval_result = dnn_classifier.evaluate(
            input_fn= eval_input_fn)
    
        print('\nValidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        if shuffleTrainingData:
            # Shuffle the Training and Validation Data for the next run
            # df_train has ALL of the data for the training and validation sets
            df_train_sample = df_train.sample(frac=1)

            # Choose new training.
            numTrain = math.ceil(percentTrainToVal * numTotalTrain)
            training_examples = preprocess_features(df_train_sample.head(numTrain))
            training_targets = preprocess_targets(df_train_sample.head(numTrain))

            # Choose new validation.
            numValidation = numTotalTrain - numTrain
            validation_examples = preprocess_features(df_train_sample.tail(numValidation))
            validation_targets = preprocess_targets(df_train_sample.tail(numValidation))

        if showTrainingPrints:
            print("Training examples summary:")
            display.display(training_examples.describe())
            print(training_examples)
            print("Validation examples summary:")
            display.display(validation_examples.describe())
            print(validation_examples)

            print("Training targets summary:")
            display.display(training_targets.describe())
            print(training_targets)
            print("Validation targets summary:")
            display.display(validation_targets.describe())
            print(validation_targets)

    print("Model training finished.")

    # Evaluate Test Set:
    test_result = dnn_classifier.evaluate(
        input_fn= test_input_fn)
    expected = [0,1]
    predictions = dnn_classifier.predict(
        input_fn= test_input_fn)
    pred_list = list()
    for i in predictions:
        pred_list.append(i['class_ids'][0])
    
    cell_indexes = test_targets.index.to_series()
    cell_nums = [ x.split("-")[0] for x in cell_indexes ]
    track_nums = [ x.split("-")[1] for x in cell_indexes ]
    cell_num_df = pd.DataFrame(cell_nums, index = test_targets.index)
    track_num_df = pd.DataFrame(track_nums, index = test_targets.index)
    pred_df = pd.DataFrame(pred_list, index = test_targets.index)
    
    total_pred = pd.concat([cell_num_df, track_num_df, pred_df, test_targets], axis = 1, ignore_index = False)
    total_pred.columns = ['cell', 'track', 'predictions', 'values']
    
    pred_correct = total_pred.loc[total_pred['predictions'] == total_pred['values']]
    pred_incorrect = total_pred.loc[total_pred['predictions'] != total_pred['values']]

    print("Predictions:")
    total_pred.to_csv('predictions/total_pred.csv', sep=',', encoding='utf-8')
    print(total_pred)

    print("Correct:")
    pred_correct.to_csv('predictions/pred_correct.csv', sep=',', encoding='utf-8')
    print(pred_correct)

    print("Incorrect:")
    pred_incorrect.to_csv('predictions/pred_incorrect.csv', sep=',', encoding='utf-8')
    print(pred_incorrect)

    print(' --- FINAL Test set accuracy: {accuracy:0.3f}\n'.format(**test_result))

    return dnn_classifier



# ___________________________________________________________________________
#
#                              DATA PREPARATION                              
# ___________________________________________________________________________



#Parameters
numCells = 8
folderPath = '/home/gokul/Documents/PythonScripts/KangminData/'
dropCols = ['frame']            #Use arrays like the format below to do multiple runs for comparison:
catsToUse = [[1,2,3,4]]         #[[1,2,3,4,5,6,7,8], [1], [1], [1]]
pvalCutOff = [.005]             #[1, 1, .005, .005]
numConsecPVal = [3]             #[1, 1, 3, 3]
mustBeSecondHalf = [True]       #[False, False, False, True]
checkLifetime = [True]          #[False, False, False, True]

#TensorFlow Parameters:
#percentTrain - the amount of total data to use for training
#percentTest - the amount of total data to use for testing (defaults to 1 - percentTrain)
#percentTrainToVal - the subset of Training to Validation data of percentTrain
#showTrainingPrints - will print out the training and validation data descriptions after each period
#shuffleTrainingData - whether or not to shuffle training and validation between periods (defaults to True)
#numperiods - how many times to repeat training and validation
percentTrain = .8               
percentTest = 1 - percentTrain
percentTrainToVal = .8          #This value is a subset of percentTrain (ex. [percentTrainToVal = .5] of [percentTrain = .8] would be .4 of the total tracks)
showTrainingPrints = False
shuffleTrainingData = True
numPeriods = 20

#Options
#ShowHist: whether or not to pause the program to show normalized and unnormalized histograms of the features
#ShowPrints: used for debugging, will show more inforamtion throughout the loading process
#ShowBoxplot: used for showing the Boxplot to pause the program
showHist = False
showPrints = False
showBoxplot = False


for i in range(0,len(catsToUse)):
    df_total = dp.prepareData(numCells = numCells, folderPath = folderPath, dropCols = dropCols, showHist = showHist, catsToUse = catsToUse[i], pvalCutOff = pvalCutOff[i], numConsecPVal = numConsecPVal[i], mustBeSecondHalf = mustBeSecondHalf[i], showPrints = showPrints, showBoxplot = showBoxplot, checkLifetime = checkLifetime[i])
    
    df_aux0 = df_total.loc[df_total['aux'].isin([0])]
    df_aux1 = df_total.loc[df_total['aux'].isin([1])]

    print("Aux + and - info:")
    print(df_aux0.describe())
    print(df_aux1.describe())

    print("RUN " + str(i) + ": Cats Used: " + str(catsToUse[i]) + " | pval: " + str(pvalCutOff[i]) + " | #ofPVal: " + str(numConsecPVal[i]) + " | Half? " + str(mustBeSecondHalf[i]) + " | Apply LT? " + str(checkLifetime[i]))
    print("# Qualifying Tracks: " + str(len(df_total.values)) + " | # Aux+: " + str(len(df_aux1.values))  + " | # Aux-: " + str(len(df_aux0.values)))
    print(df_aux1)
    print()

print("\nEnd of Data Collection / Filtering -----------------------------\n")

# ___________________________________________________________________________
#
#                              BEGIN TENSORFLOW                              
# ___________________________________________________________________________
# 
# Data: df_total - All Continuous Variables: lifetime, intensity_max, bkground, tdisp, msd, avgRise, avgDecay, riseVsDecay, avgRiseMom, avgDecayMom, riseVsDecayMom
#       NOTE:   - Category converted into OneHot (i.e. category 4 - [0,0,0,1,0,0,0,0])
df_total = df_total.astype(float)
#df_total = df_total.reset_index(drop=True)
numTracks =  df_total.shape[0]
index_dict = pd.Series(range(numTracks), index = df_total.index)
print("INDEX!!!")
print(index_dict)

_ = train_model(
    learning_rate=.0001,
    steps=1000,
    batch_size=100,
    df_total=df_total,
    percentTrain=percentTrain,
    percentTest=percentTest,
    percentTrainToVal=percentTrainToVal,
    shuffleTrainingData=shuffleTrainingData,
    showTrainingPrints=showTrainingPrints,
    numPeriods=numPeriods)




