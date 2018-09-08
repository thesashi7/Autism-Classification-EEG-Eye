import pandas as pd
from config.config import *
import glob
import numpy as np

def getSubjectFileName(sub_id):
    print main_path
    file_name = str(sub_id)
    print file_name
    while (len(file_name) < 3):
        file_name = "0" + file_name
    file_name = glob.glob("".join([main_path, file_name + "*.xlsx"]))
    print file_name
    return file_name[0]


def diagnosticFile(type="train"):
    df = pd.read_excel(main_path + main_file[type])
    dg_info = df.values
    # deleting unnecessary rows for diagnosis
    if(type == "train"):
        dg_info = np.delete(dg_info, [34, 35, 36, 37], axis=0)
    return dg_info

"""
	@file: excel files path and name
	@returns the time row and the signals. First value is the array with timestamps and the second value is a 2D array with signals
"""
def readEEG(file_name, sheet_name="sheet1"):
   print "readEEG"
   df = pd.read_excel(file_name)
   time_x = df.index.values
   signals = df.values
   #print df.columns
   return time_x, signals

