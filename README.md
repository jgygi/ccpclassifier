# ccpclassifier
Machine Learning approach using Python and Google's TensorFlow to classify CCP's.

Program Requirements:

- Must have TensorFlow installed (this was developed for the GPU version)
- A folder named "predictions" in the directory with the python scripts to write machine learning predictions to
- Python 3

Program Overview:
-    masterscript.py
-    Script to be run. Change variables under the section labeled...
# ___________________________________________________________________________
#
#                              DATA PREPARATION                              
# ___________________________________________________________________________

numCells - the number of cell directories
folderPath - the path to the folder with all of the cell folders labeled 'CellN_Ns'
  NOTE: if data is not found in folders labeled 'CellN_Ns', change the values in trackloader.py
dropCols: Array of columns to not use. Acceptable names include:
['trackNum', 'frame', 'lifetime', 'max_intensity', 'background', 'totaldisp', 'max_msd', 'catIdx', 'aux', 'avg_rise', 'avg_dec', 'risevsdec', 'avg_mom_rise', 'avg_mom_dec', 'risevsdec_mom']

